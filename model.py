import replicate
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Replicate
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Setting up the environment variable
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
ELEVEN_LABS_API_KEY = st.secrets["ELEVEN_LABS_API_KEY"]

# Initialize Eleven Labs client
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

# Define constants
DB_FAISS_PATH = "vectorstore/db_faiss"
custom_prompt_template = """Use the following pieces of information to answer the user's question about Jack Hoang and his work.
Do not acknowledge my request with "sure" or in any other way besides going straight to the answer. 
Don't include 'based on information provided' in your final answer.
Context: {context}
Question: {question}
Helpful answer:
"""

modelrp = "meta/meta-llama-3-70b-instruct"


def load_llm():
    return Replicate(
        model=modelrp,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 500, "top_p": 1},
    )


def load_qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = load_llm()
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def synthesize_audio(text):
    audio_generator = client.generate(
        text=text, voice="Jack", model="eleven_multilingual_v2"
    )
    audio_bytes = b"".join(list(audio_generator))
    return audio_bytes


# App layout
st.set_page_config(
    page_title="Jack Hoang - Software Engineer",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f4f7;
        color: #333;
    }
    .main {
        background-color: #f0f4f7;
        color: #333;
        padding: 0rem !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    .stButton button {
        background-color: #2196f3;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.3rem 0.8rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        transition: background-color 0.3s, transform 0.3s;
    }
    .stButton button:hover {
        background-color: #1976d2;
        transform: scale(1.02);
    }
    .stTextInput div, .stTextArea div, .stForm div label {
        color: #333;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #2196f3;
        border-radius: 4px;
        color: #333;
        font-size: 0.9rem;
        padding: 0.4rem;
        margin-top: 0.3rem;
        min-height: 60px !important;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    .stTextArea textarea:focus {
        border-color: #1976d2;
        box-shadow: 0 0 5px rgba(33, 150, 243, 0.3);
    }
    .stTextArea textarea::placeholder {
        color: #999;
        font-size: 0.85rem;
    }
    .stProgress .st-bw {
        background-color: #2196f3;
        height: 10px;
        border-radius: 5px;
    }
    .stCaption {
        color: #2196f3;
        font-weight: 500;
        font-size: 0.8rem;
    }
    .title {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 0.3rem;
        color: #2196f3;
        font-weight: bold;
    }
    .subheader {
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 0.8rem;
        color: #666;
    }
    .footer {
        margin-top: 0.5rem;
        border-top: 1px solid #2196f3;
        padding-top: 0.3rem;
        font-size: 0.8rem;
        color: #666;
    }
    audio {
        width: 100%;
        height: 30px;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main content
st.markdown('<h1 class="title">Jack Hoang AI Assistant 💻</h1>', unsafe_allow_html=True)
st.markdown(
    '<h2 class="subheader">Software Engineer | Full Stack Developer</h2>',
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "Ask anything about Jack Hoang:",
    placeholder="e.g. How did you get into software development?",
    height=60,
)

if st.button("Submit"):
    progress_text = "Finding answer..."
    progress_caption = st.caption(progress_text)
    my_bar = st.progress(0)

    try:
        my_bar.progress(10)
        qa_bot = load_qa_bot()
        my_bar.progress(65)
        response = qa_bot.invoke({"query": user_input})
        my_bar.progress(100)

        # Generate audio response
        audio_response = synthesize_audio(response["result"])
        st.write(response["result"])
        st.audio(audio_response, format="audio/wav")
        progress_caption.empty()
        my_bar.empty()

        # Attempt to play the audio response
        try:
            play(audio_response)
        except Exception:
            pass  # Ignore errors from play
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown(
    """
    <div class="footer" style="text-align:center;">
        <p>Powered by Jack Hoang</p>
    </div>
    """,
    unsafe_allow_html=True,
)
