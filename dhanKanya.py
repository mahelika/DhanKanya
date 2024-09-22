import streamlit as st
import anthropic
import pandas as pd
import datetime
import speech_recognition as sr
import re 
import os
import logging
from dotenv import load_dotenv
import traceback
import json

from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = 'chroma'
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY') # Get the API key from .env file using dotenv library
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

PROMPT_TEMPLATE = """
Answer the question so that it is easily understandable. The context is provided so that you can take reference from this. Please take inspiration from the context. You can also add things that you think are helpful for girls out there. Do not mention about the context provided. Answer as you usually answer.

{context}

---

{question}
"""

INTRODUCTION_PROMPTS = [
    r'introduce yourself',
    r'tell me about yourself',
    r'who are you',
    r'what can you do',
    r'how can you help me',
    r'what are your capabilities',
    r'what kind of tasks can you assist with',
    r'what are you capable of',
    r'what can i ask you',
    r'what are you good at?',
    r'what are your specialties?',
    r'what is your purpose?',
    r'what are you designed for?',
    r'how do i get started with you?',
    r'how should i interact with you?',
    r'who created you'
    r'hello',
    r'hey',
    r'namaste'
]

def main():
    st.set_page_config(page_title="Financial Advisor for Your Dreams", page_icon=":moneybag:", layout="wide")

    # Create a navigation bar
    menu = ["Home", "Templates", "Expense Tracker"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Create the Anthropic client with the API key
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        st.sidebar.success("AI assistant initialized successfully!")
    except Exception as e:
        logger.error(f"Error creating Anthropic client: {e}")
        st.sidebar.error(f"Failed to initialize the AI assistant. Error: {str(e)}")
        return

    # Display the selected page
    if choice == "Home":
        home_page(client)
    elif choice == "Templates":
        templates_page(client)
    elif choice == "Expense Tracker":
        expense_tracker_page()

def home_page(client):
    st.title("DhanKanya: Financial Empowerment for Girls in India")

    # Center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=200)

    st.write("""
    ### Welcome to our AI-powered financial literacy application!

    Our mission is to empower girls in India with the knowledge and tools they need to achieve financial independence and success.

    With our user-friendly app, you'll have access to:
    """)

    st.write("- **Interactive Budgeting Tools** to help you track your income and expenses.")
    st.write("- **Educational Resources** on essential financial literacy concepts like saving and investing.")
    st.write("- **Goal Setting Functionality** to plan and save for specific educational milestones.")

    st.markdown("---")

    if "claude_model" not in st.session_state:
        st.session_state["claude_model"] = "claude-3-haiku-20240307"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add voice input functionality
    st.write("You can ask questions in Hindi using your voice or type them in English.")
    voice_input = st.button("🎙️ Use voice input")

    if voice_input:
        prompt = get_voice_input()
    else:
        prompt = st.chat_input("Ask a question in English")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = get_response(prompt, client)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def load_templates():
    try:
        with open('state_templates.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("state_templates.json file not found")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding state_templates.json")
        return {}

def templates_page(client):
    st.title("Sample Prompts depending on the state that you reside in.")
    
    # Templates for different states
    templates  = load_templates()
    
    states = list(templates.keys())
    if not states:
        st.error("No templates available. Please check the configuration file.")
        return

    selected_state = st.selectbox("Select your state in India", states)

    if selected_state:
        st.write(f"Selected State: {selected_state}")
        st.write(f"Choose from the following prompt templates related to {selected_state}:")
        
        state_templates = templates.get(selected_state, {})
        for category, prompts in state_templates.items():
            st.subheader(category)
            for prompt in prompts:
                if st.button(prompt):
                    response = get_response(prompt, client)
                    st.markdown(response)

    logger.info(f"User selected state: {selected_state}")

def expense_tracker_page():
    st.title("Expense Tracker")
    
    # Initialize session state attributes if not already initialized
    if "expenses" not in st.session_state:
        st.session_state.expenses = {}
        st.session_state.total_expenses = {"total": 0, "necessary": 0, "avoidable": 0}
        st.session_state.selected_month = datetime.datetime.now().strftime("%Y-%m")

    # Create a form for entering new expenses
    with st.form("expense_form"):
        expense_date = st.date_input("Date")
        expense_description = st.text_input("Description")
        expense_amount = st.number_input("Amount (in ₹)", min_value=0.0, step=1.00)
        expense_necessity = st.selectbox("Necessity", ["Necessary", "Could've been avoided"])
        submitted = st.form_submit_button("Add Expense")
        
        if submitted:
            # Store the expense in a dictionary or database
            # For example, you can use a dictionary with the date as the key
            date_str = expense_date.strftime("%Y-%m-%d")
            if date_str not in st.session_state.expenses:
                st.session_state.expenses[date_str] = []

            category = "Necessary" if expense_necessity == "Necessary" else "Avoidable"
            st.session_state.expenses[date_str].append({
                "description": expense_description,
                "amount": expense_amount,
                "category": category
            })

            st.session_state.total_expenses[category.lower()] += expense_amount
            st.session_state.total_expenses["total"] += expense_amount
            st.success("Expense added successfully!")

    # Display the total expenses for each category and total for selected month
    st.subheader("Total Expenses")
    st.write("Total: ₹<span style='color:green'>{:.2f}</span>".format(st.session_state.total_expenses["total"]), unsafe_allow_html=True)
    st.write("Necessary: ₹<span style='color:green'>{:.2f}</span>".format(st.session_state.total_expenses["necessary"]), unsafe_allow_html=True)
    st.write("Avoidable: ₹<span style='color:red'>{:.2f}</span>".format(st.session_state.total_expenses["avoidable"]), unsafe_allow_html=True)
    
    # Display the expenses in a table organized by date
    all_expenses = []
    for date, expenses in sorted(st.session_state.expenses.items()):
        for expense in expenses:
            all_expenses.append({
                "Date": date,
                "Description": expense['description'],
                "Amount": "₹ {:.2f}".format(expense['amount']),  
                "Category": expense['category']
            })

    if all_expenses:
        st.subheader("Expenses")
        st.table(pd.DataFrame(all_expenses).set_index("Date"))


def get_voice_input():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)

        try:
            # Urdu language support
            try:
                urdu_text = r.recognize_google(audio, language='ur-PK')
            except sr.UnknownValueError:
                urdu_text = None

            # Telugu language support
            try:
                telugu_text = r.recognize_google(audio, language='te-IN')
            except sr.UnknownValueError:
                telugu_text = None

            # Hindi language support
            hindi_text = r.recognize_google(audio, language='hi-IN')

            # Return the recognized text in the correct language
            if hindi_text or telugu_text or urdu_text:
                if hindi_text:
                    return hindi_text
                elif telugu_text:
                    return telugu_text
                elif urdu_text:
                    return urdu_text
            else:
                st.error("Sorry, I could not understand your voice input in any supported language. Please try again.")

        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your voice input. Please try again.")

        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")

    return None

def query(query_text, client):
    try:
        # Prepare the DB with HuggingFaceBgeEmbeddings.
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            logger.info("Direct query to Claude")
            answer = get_response(query_text, client)
            return answer

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        logger.info("Query with context")
        response = get_response(prompt, client)
        return response
    except Exception as e:
        logger.error(f"Error in query function: {e}")
        logger.error(traceback.format_exc())
        return "I'm sorry, but I encountered an error while processing your query. Please try again later."


def get_response(prompt, client):
    """
    Retrieves response from the Anthropic model based on the prompt.
    """
    try:
        prompt_lower = prompt.lower()
        for pattern in INTRODUCTION_PROMPTS:
            if re.search(pattern, prompt_lower):
                return "Namaste! I'm your financial assistant, developed by the Finance team at 100GIGA and powered by Anthropic's Claude AI model. My purpose is to provide you with expert financial guidance, enhancing your financial literacy and addressing your needs. Feel free to ask me anything related to finance, and I'll be here to assist you every step of the way."
        
        message = client.messages.create(
            model=st.session_state["claude_model"],
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return ''.join(block.text for block in message.content)
    except Exception as e:
        logger.error(f"Error getting response from Claude: {e}")
        logger.error(traceback.format_exc())
        return f"I apologize, but I'm having trouble generating a response at the moment. \nError details: {str(e)}"


if __name__ == "__main__":
    main()