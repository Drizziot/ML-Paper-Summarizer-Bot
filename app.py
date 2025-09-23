import streamlit as st
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document
import arxiv
import re


# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="ML Paper Summarizer Bot",
    page_icon="📚",
    layout="wide"
)


# Check for API key
def check_api_key():
    """Check if Groq API key is available"""
    # First check environment variable
    api_key = os.getenv("GROQ_API_KEY")

    # If not in environment, check Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            api_key = None

    # If still no key, ask user to input it
    if not api_key:
        st.warning("🔑 Groq API key required!")
        st.markdown("""
        **To get started:**
        1. Get your free API key from [Groq Console](https://console.groq.com/)
        2. Enter it below or set it as an environment variable `GROQ_API_KEY`
        """)

        api_key = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Your API key will not be stored and is only used for this session"
        )

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("✅ API key set successfully!")
            st.rerun()
        else:
            st.stop()

    return api_key


# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'paper_loaded' not in st.session_state:
    st.session_state.paper_loaded = False
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None


# Initialize the language model and embeddings
@st.cache_resource
def initialize_llm(_api_key):
    return ChatGroq(
        api_key=_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )


@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def create_qa_chain(vectordb, llm, memory):
    """Create the QA chain with custom prompt template for paper analysis"""
    template = """
    You are an AI Research Assistant specializing in machine learning and AI papers. Your task is to help users understand academic papers by providing clear, accurate summaries and answering questions about the research.


    When summarizing or answering questions:
    - Focus on key contributions, methodology, and results
    - Explain complex concepts in accessible language
    - Highlight novel techniques or approaches
    - Mention limitations and future work when relevant
    - Be precise about technical details when asked


    Conversation History:
    {chat_history}


    Paper Context:
    {context}


    User Question: {question}


    Response:
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


def extract_arxiv_id(input_text):
    """Extract arXiv ID from various input formats"""
    # Remove whitespace
    input_text = input_text.strip()

    # Pattern for arXiv IDs (new format: YYMM.NNNNN, old format: subject-class/YYMMnnn)
    arxiv_patterns = [
        r'(\d{4}\.\d{4,5})',  # New format: 2301.12345
        r'([a-z-]+/\d{7})',   # Old format: cs/0123456
        r'arXiv:(\d{4}\.\d{4,5})',  # With arXiv prefix
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',  # From URL
    ]

    for pattern in arxiv_patterns:
        match = re.search(pattern, input_text)
        if match:
            return match.group(1)

    return input_text  # Return as-is if no pattern matches


def fetch_arxiv_paper(paper_id, api_key):
    """Fetch paper from arXiv and create vector database"""
    try:
        # Clean the paper ID
        clean_id = extract_arxiv_id(paper_id)

        # Search for the paper
        client = arxiv.Client()
        search = arxiv.Search(id_list=[clean_id])

        papers = list(client.results(search))
        if not papers:
            st.error(f"No paper found with ID: {clean_id}")
            return None, None, False

        paper = papers[0]

        # Create document from paper content
        paper_content = f"""
        Title: {paper.title}
       
        Authors: {', '.join([author.name for author in paper.authors])}
       
        Published: {paper.published.strftime('%Y-%m-%d')}
       
        Categories: {', '.join(paper.categories)}
       
        Abstract:
        {paper.summary}
       
        PDF URL: {paper.pdf_url}
        """

        # Create document object
        doc = Document(
            page_content=paper_content,
            metadata={
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime('%Y-%m-%d'),
                "categories": paper.categories,
                "arxiv_id": clean_id,
                "pdf_url": paper.pdf_url
            }
        )

        # Create vector database
        embedding = initialize_embeddings()
        vectordb = DocArrayInMemorySearch.from_documents(
            documents=[doc],
            embedding=embedding
        )

        # Create QA chain
        llm = initialize_llm(api_key)
        qa_chain = create_qa_chain(vectordb, llm, st.session_state.memory)

        return qa_chain, paper, True

    except Exception as e:
        st.error(f"Error fetching paper: {str(e)}")
        return None, None, False


def main():
    # Check API key first
    api_key = check_api_key()

    st.title("📚 ML Paper Summarizer Bot")
    st.markdown(
        "Enter an arXiv paper ID to get AI-powered summaries and ask questions about research papers!")

    # Sidebar for paper input
    with st.sidebar:
        st.header("Load Paper")
        st.markdown("**Input formats supported:**")
        st.markdown("• `2301.12345` (arXiv ID)")
        st.markdown("• `arXiv:2301.12345`")
        st.markdown("• `https://arxiv.org/abs/2301.12345`")

        paper_input = st.text_input(
            "Enter arXiv Paper ID or URL:",
            placeholder="e.g., 2301.12345",
            help="Enter the arXiv ID, URL, or any format containing the ID"
        )

        if paper_input:
            if st.button("Load Paper", type="primary"):
                with st.spinner("Fetching paper from arXiv..."):
                    qa_chain, paper, success = fetch_arxiv_paper(
                        paper_input, api_key)
                    if success:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.current_paper = paper
                        st.session_state.paper_loaded = True
                        st.success("✅ Paper loaded successfully!")
                        st.rerun()

        # Clear conversation button
        if st.session_state.paper_loaded:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.session_state.memory.clear()
                st.success("Conversation cleared!")
                st.rerun()

        # Reset application button
        if st.button("Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application reset!")
            st.rerun()

    # Main interface
    if not st.session_state.paper_loaded:
        st.info("👈 Enter an arXiv paper ID in the sidebar to get started!")

        # Show example paper IDs and questions
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Example Paper IDs:")
            example_papers = [
                "2017.14426 (BERT paper)",
                "1706.03762 (Transformer)",
                "2005.14165 (GPT-3)",
                "2010.11929 (Vision Transformer)",
                "2203.02155 (ChatGPT/InstructGPT)"
            ]

            for paper in example_papers:
                st.markdown(f"• `{paper}`")

        with col2:
            st.subheader("❓ What you can ask:")
            sample_questions = [
                "Summarize this paper in simple terms",
                "What are the key contributions?",
                "How does the methodology work?",
                "What are the main results?",
                "What are the limitations?",
                "How does this compare to previous work?"
            ]

            for question in sample_questions:
                st.markdown(f"• {question}")

    else:
        # Display paper information
        paper = st.session_state.current_paper

        with st.expander("📄 Paper Information", expanded=False):
            st.markdown(f"**Title:** {paper.title}")
            st.markdown(
                f"**Authors:** {', '.join([author.name for author in paper.authors])}")
            st.markdown(
                f"**Published:** {paper.published.strftime('%Y-%m-%d')}")
            st.markdown(f"**Categories:** {', '.join(paper.categories)}")
            st.markdown(f"**PDF:** [View on arXiv]({paper.pdf_url})")

            with st.expander("Abstract", expanded=False):
                st.write(paper.summary)

        # Quick action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📝 Get Summary", use_container_width=True):
                st.session_state.chat_history.append(
                    ("Provide a comprehensive summary of this paper", ""))
                st.rerun()

        with col2:
            if st.button("🔬 Key Contributions", use_container_width=True):
                st.session_state.chat_history.append(
                    ("What are the key contributions and novel aspects of this paper?", ""))
                st.rerun()

        with col3:
            if st.button("📊 Results & Impact", use_container_width=True):
                st.session_state.chat_history.append(
                    ("What are the main results and their significance?", ""))
                st.rerun()

        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {question}")

                if answer == "":  # If answer is empty, generate it
                    with st.spinner("Analyzing the paper..."):
                        try:
                            result = st.session_state.qa_chain.invoke(
                                {"question": question})
                            answer = result['answer']
                            st.session_state.chat_history[i] = (
                                question, answer)
                        except Exception as e:
                            answer = f"Error generating response: {str(e)}"
                            st.session_state.chat_history[i] = (
                                question, answer)

                st.markdown(f"**AI Research Assistant:** {answer}")
                st.divider()

        # Chat input
        user_question = st.chat_input("Ask a question about this paper...")

        if user_question and st.session_state.qa_chain:
            # Add user message to chat
            with st.container():
                st.markdown(f"**You:** {user_question}")

                # Get AI response
                with st.spinner("Analyzing the paper..."):
                    try:
                        result = st.session_state.qa_chain.invoke(
                            {"question": user_question})
                        answer = result['answer']

                        # Display AI response
                        st.markdown(f"**AI Research Assistant:** {answer}")

                        # Add to chat history
                        st.session_state.chat_history.append(
                            (user_question, answer))

                        # Update memory
                        st.session_state.memory.chat_memory.add_user_message(
                            user_question)
                        st.session_state.memory.chat_memory.add_ai_message(
                            answer)

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** Ask specific questions about methodology, results, or implications. "
        "The AI can explain complex concepts in simpler terms or provide detailed technical analysis."
    )
    st.markdown(
        "🤖 **Powered by Groq's LLaMA 3.1** - Fast and intelligent paper analysis!")


if __name__ == "__main__":
    main()
