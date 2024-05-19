import streamlit as st
import anthropic
import clipboard
import pandas as pd
import datetime
import speech_recognition as sr
from speech_recognition import Recognizer, AudioFile
import pyttsx3
import re 
import os

from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    from keys import ANTHROPIC_API_KEY


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
]

def main():
    st.set_page_config(page_title="Financial Advisor for Your Dreams", page_icon=":moneybag:", layout="wide")

    # Create a navigation bar
    menu = ["Home", "Templates", "Expense Tracker"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Create the Anthropic client with the API key
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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

    st.markdown("""
    ---
    """)

    if "claude_model" not in st.session_state:
        st.session_state["claude_model"] = "claude-3-haiku-20240307"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if ANTHROPIC_API_KEY:
        # Create the Anthropic client
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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

def templates_page(client):
    st.title("Sample Prompts depending on the state that you reside in.")
    
    # Ask the user to select their state
    selected_state = st.selectbox("Select your state in India", ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"])
    
    # Templates for different states
    state_templates = {
        "Andhra Pradesh": {
            "Sample Prompts": [
                "I would like to inquire about financial aid programs offered by the Andhra Pradesh government specifically for female students.",
                "With the intention of pursuing higher education, I am interested in learning about relevant scholarships and schemes available in Andhra Pradesh.",
                "As a student belonging to the SC/ST/BC/EBC community in Andhra Pradesh, I would like to know about any scholarships designated for this category.",
                "In my current situation as an orphan/destitute girl residing in Andhra Pradesh, I would appreciate information on scholarship programs that can offer financial assistance for my education.",
                "I would like to gain comprehensive information on educational programs specifically catered towards female students in Andhra Pradesh, with a particular focus on those offering scholarships.",
                "I am interested in pursuing an ITI course in Andhra Pradesh. Could you please advise on any financial assistance programs available for female students enrolled in such programs?",
                "To effectively manage my finances for college, I would appreciate any guidance on budgeting or financial planning strategies.",
                "I'm considering opening a bank account for the first time. What documents do I need, and what kind of accounts are suitable for saving goals?",
            ]
        },
        "Tamil Nadu": {
            "Sample Prompts": [
                "College fees are a challenge! Can you help me find scholarships for girls pursuing higher education in Tamil Nadu?",
                "The government offers schemes to support girls' education in Tamil Nadu. Can you guide me to the most suitable one?",
                "I belong to the Scheduled Caste community in Tamil Nadu. Are there scholarships specifically offered to students from this category?",
                "As a girl facing financial hardship in Tamil Nadu [explain your situation - orphan, economically disadvantaged], are there special scholarship programs available to support me?",
                "Beyond scholarships, are there any interesting programs or opportunities designed to empower girls in Tamil Nadu?",
                "I'm curious about educational options for girls in Tamil Nadu, particularly those that might include scholarships.",
                "Is there any financial support available for girls pursuing vocational courses (ITI) in Tamil Nadu?"
            ]
        },
        "Arunachal Pradesh": {
            "Sample Prompts for Arunachal Pradesh": [
                "College fees in Arunachal Pradesh are giving me a headache! Are there scholarships I can apply for to help out?",
                "I heard about government schemes for girls' education in Arunachal Pradesh. Can you help me find the best one for me?",
                "Tuition fees are a barrier to my education. Are there any government schemes in Arunachal Pradesh that can help me overcome this challenge?",
                "I'm an Arunachal Pradesh resident belonging to the backward community. Are there scholarships specifically for students like me?",
                "Due to my financial situation, I need financial aid for my education. Are there special scholarship programs available in Arunachal Pradesh?",
                "I'd love to learn more about educational options for girls in Arunachal Pradesh, especially those with scholarships.",
                "I'm looking to advance my education! Tell me everything about educational opportunities for girls in Arunachal Pradesh.",
                "Is there any financial assistance available for girls pursuing courses at Arunachal University?",
                "Budgeting and managing my finances for college feels overwhelming. Do you have any tips to help me out?"
            ]
        },
        "Bihar": {
            "Prompts for Bihar": [
                "Due to financial limitations, pursuing higher education seems difficult. Are there any government schemes in Bihar that can empower girls like me to achieve our academic goals?",
                "I'm interested in learning more about educational options for girls in Bihar, particularly those that might offer financial assistance through scholarships.",
                "I'm eager to further my education! Can you provide me with comprehensive details on educational programs available for girls in Bihar?",
                "Managing college finances feels overwhelming. Do you have any tips on budgeting or financial planning that can help me?",
                "I'm a student in Patna and receive a monthly stipend. How can I best manage my expenses to save for further education?",
                "Can you tell me more about the Chief Minister Kanya Utthan Yojana for female students?",
                "Are there any government schemes in Bihar that provide financial assistance for vocational training for women?",
                "Are there any scholarships specifically for graduation or post-graduation studies in Bihar?",
                "When is the application deadline for the Bihar scholarships for female students?",
                "What is the official website to apply for scholarships in Bihar?",
                "I belong to a reservation category (SC, ST, OBC, EBC), are there any additional scholarships available for me in Bihar?",  
            ]
        },
        "Chhattisgarh": {
            "Sample Prompts" :[
                "What scholarships are offered by the Chhattisgarh government specifically for female students?",
                "Are there any merit-based scholarships available for higher education in Chhattisgarh?",
                "When do scholarship applications typically open in Chhattisgarh?",
                "Where can I find the official application portal for Chhattisgarh scholarships?",
                "I belong to a reservation category (SC, ST, OBC), are there additional scholarships I qualify for in Chhattisgarh?",
                "Do reservation category scholarships have different application processes or eligibility criteria compared to general scholarships?",
                "Are there any government schemes in Chhattisgarh that support female students from underprivileged families pursuing professional courses?",
                "Can you provide details on the Rajya Chatra Vritti Scheme for girls in classes 3-8?",
                "What is the Medhavi Chhatr/Chhaatra Shiksha Protsahan Yojana and is it applicable to female students?",
            ]
        },
        "Goa": {
            "Sample Prompts": [
                "What are the scholarships offered by the Goa government for female students?",
                "Are there any scholarships specifically for undergraduate or postgraduate studies in Goa?",
                "I'm interested in learning new skills to boost my career options. Does the Goa government offer any skill development programs specifically for women?",
                "When is the application deadline for scholarships offered by the Goa government?",
                "Does the Goa government have an online portal for scholarship applications?",
                "I belong to a reservation category (SC, ST, OBC), are there any additional scholarships available for me in Goa?",
                "Can you tell me more about the Post Matric Scholarship Scheme for Minorities in Goa?",
                "Are there any government schemes in Goa that provide financial assistance for women pursuing technical education?",
                "I'm a woman entrepreneur in Goa. Are there any government schemes or loans available to support women-led businesses?",
                "What are some government schemes in Goa that focus on empowering women?",
                "Are there any online portals or resources where I can find information about government benefits for women in Goa?",
            ]
        },
        "Gujarat": {
            "Sample Prompts": [
                "शुभचिंतन! मेरे पास डिग्री है, लेकिन नौकरी नहीं मिलती. क्या कोई योजना है जो स्नातक डिग्री धारक महिलाओं को रोजगार पाने में मदद करती है?",
                "मेरे परिवार में आर्थिक तंगी चल रही है. क्या महिलाओं के लिए कोई आर्थिक सहायता योजना उपलब्ध है?",
                "मुझे स्वास्थ्य संबंधी समस्या है. क्या महिलाओं के लिए कोई नि:शुल्क इलाज योजना उपलब्ध है?",
                "I'm planning to start a small business in Gujarat. Are there any government loans available for women entrepreneurs?",
                "I've heard about the Mukhyamantri Bal Shakti Yojana. Can you provide details on eligibility criteria and the application process for this scholarship?"
                "I'd like to learn more about government schemes in Gujarat that offer financial assistance for women pursuing higher education",
                "Are there any online resources or government websites where I can find a comprehensive list of scholarships and schemes for women in Gujarat?",
                "I'd appreciate any information on workshops or seminars focused on scholarships and financial aid specifically for female students in Gujarat.",
            ]
        },
        "Haryana": {
            "Sample Prompts": [
                "Can you provide details on the Apni Beti Apna Dhan scheme offered by the Haryana government? I'd like to learn more about the eligibility criteria and benefits.",
                "I'm interested in pursuing higher education in Haryana. What are the top scholarship programs available for female students in the state?",
                "As a girl belonging to the SC/ST community in Haryana, are there any special scholarship schemes I can apply for to fund my college education?",
                "I'm from an economically weaker section in Haryana. Are there any government initiatives that can support my higher studies through financial assistance?",
                "Can you guide me on the application process and deadlines for the Mukhyamantri Vivah Shagun Yojana in Haryana? I'd like to avail the benefits of this scheme.",
                "What are the vocational training programs offered by the Haryana government specifically for the empowerment of girls and women in the state?",
                "I'm looking to start a small business. Are there any entrepreneurship development schemes run by the Haryana government that provide funding support for women?"
            ]
        },
        "Himachal Pradesh": {
            "Sample Prompts": [
                "I'm a female student in Himachal Pradesh and would like to know more about the Chief Minister's Girl Child Scheme. Can you explain the benefits and eligibility criteria?",
                "As a girl belonging to the SC/ST community in Himachal Pradesh, what are the top scholarship programs I can apply for to fund my higher education?",
                "I'm interested in pursuing a professional course like engineering or medicine in Himachal Pradesh. Are there any state-sponsored scholarship schemes available for female students in these fields?",
                "My family is facing financial difficulties. Are there any government initiatives in Himachal Pradesh that can provide financial assistance for girls from low-income backgrounds to continue their studies?",
                "I want to learn new skills to improve my employability. Can you tell me about any skill development programs run by the Himachal Pradesh government specifically for women?",
                "I'm a girl entrepreneur in Himachal Pradesh. Are there any state schemes that offer funding, mentorship or other support for women-led startups?",
                "Can you guide me on the application process and deadlines for scholarships offered by the Himachal Pradesh government for female students? I'd like to apply for the relevant ones."
            ]
        },
        "Jharkhand": {
            "Sample Prompts": [
                    "I'm a student in Jharkhand facing financial constraints. Can you guide me on the scholarships or financial aid programs available for girls in my state?",
                    "Are there any state-specific schemes in Jharkhand that support girls' education, particularly those from marginalized communities?",
                    "I'm interested in pursuing higher education in Jharkhand. Could you provide details on scholarships or schemes tailored for female students in the state?",
                    "As a girl from Jharkhand, I'm looking for opportunities to enhance my skills. Are there any skill development programs specifically designed for women in the state?",
                    "What are the key government initiatives in Jharkhand that focus on empowering women through education and financial support?",
                    "I'm considering vocational training in Jharkhand. Are there any scholarships or schemes that can assist women like me in pursuing such courses?",
                    "Can you elaborate on the Mukhyamantri Kanya Utthan Yojana or any similar schemes in Jharkhand that aim to uplift female students?",
                    "Where can I find reliable information on scholarships and financial aid programs for girls in Jharkhand?",
                    "I belong to a reserved category in Jharkhand. Are there additional scholarships or benefits available for students like me?",
                    "How can I apply for scholarships in Jharkhand, and what are the deadlines for different scholarship programs?",
                ]
            },
        "Karnataka": {
            "Sample Prompts": [
                    "I'm a female student in Karnataka seeking financial assistance for my education. What are the prominent scholarship programs available for girls in the state?",
                    "Could you provide insights into the state-specific schemes in Karnataka that cater to the educational needs of girls, especially those from disadvantaged backgrounds?",
                    "I'm interested in pursuing higher studies in Karnataka. Can you detail the scholarships or financial aid options specifically for female students?",
                    "As a woman in Karnataka looking to enhance my employability, are there skill development initiatives or programs that I can benefit from?",
                    "What are the key government schemes in Karnataka that focus on promoting education and empowerment among women?",
                    "I'm considering enrolling in a vocational course in Karnataka. Are there scholarships or schemes that support women in vocational training?",
                    "Can you explain the details of the Karnataka government's schemes like the Vidya Siri or any other programs aimed at supporting female students?",
                    "Where can I access comprehensive information on scholarships and financial aid available for girls in Karnataka?",
                    "Do reserved categories in Karnataka have specific scholarships or benefits for female students?",
                    "How can I apply for scholarships in Karnataka, and what are the deadlines for various scholarship applications?",
                ]
            },
            "Kerala": {
                "Sample Prompts": [
                    "I'm a student in Kerala facing financial challenges. What scholarships or financial aid programs are available for girls in the state?",
                    "Are there specific schemes in Kerala that focus on supporting girls' education, especially those from marginalized communities?",
                    "I'm planning to pursue higher education in Kerala. Can you provide details on scholarships or schemes tailored for female students in the state?",
                    "As a girl from Kerala, I'm interested in skill development opportunities. Are there programs designed for women in the state?",
                    "What are the key government initiatives in Kerala that aim to empower women through education and financial support?",
                    "I'm considering vocational training in Kerala. Are there scholarships or schemes that can assist women like me in such courses?",
                    "Can you elaborate on the state schemes like the Vidya Samunnathi or any similar programs in Kerala that uplift female students?",
                    "Where can I find reliable information on scholarships and financial aid programs for girls in Kerala?",
                    "Do reserved categories in Kerala have specific scholarships or benefits for female students?",
                    "How can I apply for scholarships in Kerala, and what are the deadlines for different scholarship programs?",
                ]
            },
        "Madhya Pradesh": {
                "Sample Prompts": [
                    "I'm a female student in Madhya Pradesh looking for financial aid programs. Can you guide me on available scholarships or schemes?",
                    "Are there any scholarships specifically for girls pursuing higher education in Madhya Pradesh?",
                    "As a student from a reserved category in Madhya Pradesh, are there scholarships tailored for students like me?",
                    "I'm facing financial challenges in Madhya Pradesh. Are there special scholarship programs that can assist me?",
                    "Could you provide details on educational programs in Madhya Pradesh that offer scholarships for female students?",
                    "I'm interested in vocational courses in Madhya Pradesh. Are there financial assistance programs available for female students in these fields?",
                    "Managing college finances is a concern. Do you have any advice on budgeting or financial planning for students in Madhya Pradesh?",
                    "What are the key features of the Mukhyamantri Medhavi Vidyarthi Yojana in Madhya Pradesh?",
                    "Are there any government schemes in Madhya Pradesh that support female students from underprivileged backgrounds?",
                    "Can you explain the process of applying for scholarships in Madhya Pradesh and the eligibility criteria?",
                ]
            },
            "Maharashtra": {
                "Sample Prompts": [
                    "I'm a female student in Maharashtra seeking scholarships. Can you provide information on available financial aid programs?",
                    "Are there scholarships specifically for girls pursuing undergraduate or postgraduate studies in Maharashtra?",
                    "As a student from a reserved category in Maharashtra, are there scholarships designed for students like me?",
                    "Financial constraints are hindering my education in Maharashtra. Are there scholarships that can help me overcome this challenge?",
                    "What are the notable educational programs in Maharashtra that offer scholarships for female students?",
                    "I'm interested in vocational courses in Maharashtra. Are there financial assistance programs for female students in these fields?",
                    "Managing college finances is overwhelming. Do you have any tips on budgeting or financial planning for students in Maharashtra?",
                    "Could you explain the Balika Samruddhi Yojana in Maharashtra and how it supports female students?",
                    "Are there government schemes in Maharashtra that provide financial assistance to empower women through education?",
                    "What are the application deadlines and procedures for scholarships in Maharashtra?",
                ]
            },
            "Manipur": {
                "Sample Prompts": [
                    "I'm a female student in Manipur looking for scholarships. Can you guide me on available financial aid programs?",
                    "Are there scholarships specifically for girls pursuing higher education in Manipur?",
                    "As a student from a reserved category in Manipur, are there scholarships tailored for students like me?",
                    "Financial challenges are impacting my education in Manipur. Are there special scholarship programs that can assist me?",
                    "Could you provide details on educational programs in Manipur that offer scholarships for female students?",
                    "I'm interested in vocational courses in Manipur. Are there financial assistance programs available for female students in these fields?",
                    "Managing college finances is a concern. Do you have any advice on budgeting or financial planning for students in Manipur?",
                    "What are the key features of the Chief Minister-gi Hakshelgi Tengbang (CMHT) scheme in Manipur?",
                    "Are there any government schemes in Manipur that support female students from underprivileged backgrounds?",
                    "Can you explain the process of applying for scholarships in Manipur and the eligibility criteria?",
                ]
            },
            "Meghalaya": {
                "Sample Prompts": [
                    "I'm a female student in Meghalaya seeking scholarships. Can you provide information on available financial aid programs?",
                    "Are there scholarships specifically for girls pursuing undergraduate or postgraduate studies in Meghalaya?",
                    "As a student from a reserved category in Meghalaya, are there scholarships designed for students like me?",
                    "Financial constraints are hindering my education in Meghalaya. Are there scholarships that can help me overcome this challenge?",
                    "What are the notable educational programs in Meghalaya that offer scholarships for female students?",
                    "I'm interested in vocational courses in Meghalaya. Are there financial assistance programs for female students in these fields?",
                    "Managing college finances is overwhelming. Do you have any tips on budgeting or financial planning for students in Meghalaya?",
                    "Could you explain the Chief Minister's Career Guidance and Counselling Scheme in Meghalaya and how it supports female students?",
                    "Are there government schemes in Meghalaya that provide financial assistance to empower women through education?",
                    "What are the application deadlines and procedures for scholarships in Meghalaya?",
                ]
            },
            "Mizoram": {
                "Sample Prompts": [
                    "I'm a female student in Mizoram looking for scholarships. Can you guide me on available financial aid programs?",
                    "Are there scholarships specifically for girls pursuing higher education in Mizoram?",
                    "As a student from a reserved category in Mizoram, are there scholarships tailored for students like me?",
                    "Financial challenges are impacting my education in Mizoram. Are there special scholarship programs that can assist me?",
                    "Could you provide details on educational programs in Mizoram that offer scholarships for female students?",
                    "I'm interested in vocational courses in Mizoram. Are there financial assistance programs available for female students in these fields?",
                    "Managing college finances is a concern. Do you have any advice on budgeting or financial planning for students in Mizoram?",
                    "What are the key features of the Chief Minister's Scholarship Scheme in Mizoram?",
                    "Are there any government schemes in Mizoram that support female students from underprivileged backgrounds?",
                    "Can you explain the process of applying for scholarships in Mizoram and the eligibility criteria?",
                ]
            },
        "Nagaland": {
                "Sample Prompts": [
                    "I'm a student in Nagaland facing financial constraints. Are there any existing government scholarships or schemes that can assist me in pursuing higher education?",
                    "As a girl from Nagaland, I'm interested in learning about scholarships or financial aid programs specifically available for female students in the state.",
                    "I belong to a reservation category (SC, ST, OBC) in Nagaland. Are there any scholarships tailored for students like me?",
                    "I've heard about educational schemes in Nagaland. Can you provide detailed information on the most beneficial ones for female students?",
                    "Are there any scholarships in Nagaland that support girls from underprivileged backgrounds or specific communities?",
                    "I'm looking to enhance my skills through vocational training. Are there any government programs in Nagaland that offer financial assistance for women in such courses?",
                    "What are the key eligibility criteria for scholarships in Nagaland, and how can I apply for them?",
                    "Can you guide me on where to find official information about scholarships and financial aid programs for female students in Nagaland?",
                    "I'm interested in pursuing higher education in Nagaland. Are there scholarships available for undergraduate or postgraduate studies?",
                    "What are the deadlines for scholarship applications in Nagaland, and how can I stay updated on new opportunities?",
                ]
            },
    "Odisha": {
        "Prompts for Odisha": [
            "College expenses are a concern for me in Odisha. Are there any government scholarships or financial aid programs that can help alleviate this burden?",
            "I'm a female student in Odisha seeking information on scholarships or schemes specifically designed to support girls in their educational pursuits.",
            "As a student from Odisha belonging to a reservation category (SC, ST, OBC), are there scholarships available that cater to students like me?",
            "I've heard about educational initiatives in Odisha. Can you provide detailed insights into the scholarships and financial aid programs for female students?",
            "Are there scholarships in Odisha that focus on empowering girls from disadvantaged backgrounds or specific communities?",
            "I'm interested in vocational courses in Odisha. Are there government schemes that provide financial assistance for women pursuing such programs?",
            "What are the specific requirements to qualify for scholarships in Odisha, and how can I access the application process?",
            "Where can I find reliable information on scholarships and financial aid programs for female students in Odisha?",
            "I'm considering higher education in Odisha. Are there scholarships available for undergraduate or postgraduate studies?",
            "When do scholarship applications typically open in Odisha, and how can I ensure I don't miss any opportunities?",
        ]
    },
    "Punjab": {
        "Prompts for Punjab": [
            "I'm facing financial challenges in pursuing higher education in Punjab. Are there existing government scholarships or schemes that can support me?",
            "As a female student in Punjab, I'm interested in scholarships or financial aid programs specifically aimed at empowering girls in their academic endeavors.",
            "Being from a reservation category (SC, ST, OBC) in Punjab, are there scholarships tailored to students in similar circumstances?",
            "Could you provide detailed information on the scholarships and financial aid programs available for female students in Punjab?",
            "Are there scholarships in Punjab that target girls from economically disadvantaged backgrounds or specific communities?",
            "I'm keen on vocational training in Punjab. Do government programs offer financial assistance for women pursuing such courses?",
            "What are the eligibility criteria for scholarships in Punjab, and how can I navigate the application process?",
            "Where can I find comprehensive details on scholarships and financial aid programs for female students in Punjab?",
            "I'm exploring higher education opportunities in Punjab. Are there scholarships accessible for undergraduate or postgraduate studies?",
            "When are the scholarship application deadlines in Punjab, and how can I keep track of new opportunities?",
        ]
    },
    "Rajasthan": {
        "Prompts for Rajasthan": [
            "The cost of college education is a concern for me in Rajasthan. Are there government scholarships or financial aid programs that can assist female students like me?",
            "As a girl in Rajasthan, I'm looking for scholarships or schemes specifically tailored to support female students in their educational pursuits.",
            "I belong to a reservation category (SC, ST, OBC) in Rajasthan. Are there scholarships available that cater to students from similar backgrounds?",
            "I've heard about educational schemes in Rajasthan. Can you provide detailed information on scholarships and financial aid programs for female students?",
            "Are there scholarships in Rajasthan that focus on empowering girls from disadvantaged backgrounds or specific communities?",
            "I'm interested in vocational courses in Rajasthan. Are there government programs that offer financial assistance for women pursuing such programs?",
            "What are the specific criteria to qualify for scholarships in Rajasthan, and how can I apply for them?",
            "Where can I find reliable information on scholarships and financial aid programs for female students in Rajasthan?",
            "I'm considering higher education in Rajasthan. Are there scholarships available for undergraduate or postgraduate studies?",
            "When do scholarship applications typically open in Rajasthan, and how can I ensure I don't miss any opportunities?",
        ]
    },
    "Sikkim":
{
"Sample Prompts": [
"I'm interested in pursuing higher education in Sikkim. Can you provide details on the Sikkim Chief Minister's Meritorious Scholarship Scheme for female students?",
"As a girl from Sikkim, I'd like to know more about the Sikkim Government's Post Matric Scholarship Scheme. What are the eligibility criteria and application process?",
"I belong to a marginalized community in Sikkim. Are there any special scholarship programs offered by the state government for SC/ST/OBC female students?",
"I'm a bright student from an underprivileged family in Sikkim. Can you guide me towards any financial assistance programs or scholarships available for girls in my situation?",
"Sikkim has a strong focus on women's empowerment. Can you tell me about any unique scholarship or education schemes specifically designed for the development of girls in the state?",
"I'm interested in pursuing a vocational course in Sikkim. Are there any government scholarships or fee waivers available for female students in this field?",
"As a first-generation learner from Sikkim, I'm worried about managing my college finances. What are some budgeting tips or financial planning resources available for girls like me?"
]
},
"Telangana":
{
"Sample Prompts": [
"I'm a girl from Telangana and want to pursue higher education. Can you provide details on the Telangana State Post-Matric Scholarship Scheme for female students?",
"My family is facing financial constraints in Telangana. Are there any government schemes that offer scholarships or fee reimbursements for girls from low-income backgrounds?",
"I belong to a Scheduled Caste/Scheduled Tribe community in Telangana. What are the scholarship opportunities available specifically for SC/ST female students in the state?",
"I'm an orphan/destitute girl living in Telangana. Can you guide me towards any welfare schemes or educational assistance programs that can support my academic goals?",
"Telangana has a strong focus on women's empowerment. Can you tell me about any unique scholarship or education initiatives designed exclusively for the development of girls in the state?",
"I'm interested in pursuing a technical/vocational course in Telangana. Are there any government scholarships or fee concessions available for female students in these fields?",
"As a college student in Telangana, I'm struggling to manage my finances. Do you have any budgeting tips or financial planning resources that can help girls like me?",
"What is EAMCET/ ECET exam? How can it benefit me?"
]
},
"Tripura":
{
"Sample Prompts": [
"I'm a girl from Tripura and want to pursue higher education. Can you provide details on the Tripura Scholarship Scheme for female students?",
"My family in Tripura is facing financial constraints. Are there any government schemes that offer scholarships or fee reimbursements for girls from low-income backgrounds?",
"I belong to a Scheduled Tribe community in Tripura. What are the scholarship opportunities available specifically for ST female students in the state?",
"I'm an orphan/destitute girl living in Tripura. Can you guide me towards any welfare schemes or educational assistance programs that can support my academic goals?",
"Tripura has a focus on women's empowerment. Can you tell me about any unique scholarship or education initiatives designed exclusively for the development of girls in the state?",
"I'm interested in pursuing a technical/vocational course in Tripura. Are there any government scholarships or fee concessions available for female students in these fields?",
"As a college student in Tripura, I'm struggling to manage my finances. Do you have any budgeting tips or financial planning resources that can help girls like me?"
]
},
"Uttar Pradesh": {
        "Prompts for Uttar Pradesh": [
            "I am a female student in Uttar Pradesh looking for scholarships to pursue higher education. Can you provide information on existing financial aid programs available for girls in the state?",
            "As a student from a marginalized community in Uttar Pradesh, I am interested in scholarships or schemes that cater to students like me. Could you guide me on the options available?",
            "I am an orphan girl residing in Uttar Pradesh and in need of financial assistance for my education. Are there specific scholarship programs in the state that support students in similar situations?",
            "I would like to know about educational programs in Uttar Pradesh that focus on empowering female students through scholarships and financial aid. Can you provide details on such initiatives?",
            "Are there any scholarships or schemes in Uttar Pradesh specifically designed for female students pursuing vocational courses like ITI? I am looking for financial support in this regard.",
            "In Uttar Pradesh, what are the available financial assistance programs for female students interested in ITI courses? I am seeking guidance on funding options for my education.",
            "I am considering budgeting for college in Uttar Pradesh. Can you offer advice on financial planning strategies tailored to students in the state?",
            "As a first-time bank account opener in Uttar Pradesh, I would like to know the required documents and the types of accounts suitable for saving goals. Could you provide guidance on this?",
        ]
    },
    "Uttarakhand": {
        "Prompts for Uttarakhand": [
            "I am a female student in Uttarakhand seeking scholarships for higher education. Can you provide information on the scholarships and financial aid programs available for girls in the state?",
            "As a student from an underprivileged background in Uttarakhand, I am looking for scholarships or schemes that can assist students like me. Could you outline the available options?",
            "I am facing financial challenges as a girl in Uttarakhand and need support for my education. Are there specific scholarship programs in the state that cater to students in difficult circumstances?",
            "I am interested in educational programs in Uttarakhand that focus on empowering female students through scholarships and financial aid. Can you elaborate on such initiatives?",
            "Are there scholarships or schemes in Uttarakhand tailored for female students pursuing vocational courses like ITI? I am in need of financial assistance for such programs.",
            "What financial assistance programs are available in Uttarakhand for female students pursuing ITI courses? I am seeking information on funding opportunities for my education.",
            "I am looking to manage my college finances in Uttarakhand. Can you provide tips on financial planning strategies suitable for students in the state?",
            "As a newcomer to banking in Uttarakhand, I would like to know the necessary documents and the types of accounts suitable for saving goals. Could you offer guidance on this matter?",
        ]
    },
    "West Bengal": {
        "Prompts for West Bengal": [
            "I am a female student in West Bengal searching for scholarships to support my higher education. Can you provide details on the scholarships and financial aid programs available for girls in the state?",
            "As a student from an economically disadvantaged background in West Bengal, I am interested in scholarships or schemes that cater to students facing financial challenges. Could you highlight the available options?",
            "I am a girl in West Bengal facing financial difficulties and require assistance for my education. Are there specific scholarship programs in the state that address the needs of students in similar situations?",
            "I would like to learn about educational programs in West Bengal that focus on empowering female students through scholarships and financial aid. Can you explain such initiatives in the state?",
            "Are there scholarships or schemes in West Bengal specifically designed for female students pursuing vocational courses like ITI? I am seeking financial support for such educational endeavors.",
            "In West Bengal, what financial assistance programs are available for female students interested in ITI courses? I am looking for guidance on funding options for my education.",
            "I am considering budgeting for college in West Bengal. Can you provide advice on financial planning strategies tailored to students in the state?",
            "As a first-time bank account opener in West Bengal, I would like to know the required documents and the types of accounts suitable for saving goals. Could you offer assistance on this matter?",
        ]
    },
    "Assam": {
        "Prompts if you hail from Assam": [
            "Due to financial limitations, continuing my education after school seems difficult. Are there any government schemes in Assam that can help girls like me achieve our academic goals?",
            "I'm interested in learning more about educational options for girls in Assam, particularly those that offer financial assistance through scholarships or grants.",
            "I'm eager to further my education! Can you provide me with details on educational programs available for girls in Assam, including financial aid options?",
            "Managing college finances feels overwhelming. Do you have any tips on budgeting or financial planning that can help a girl like me save for higher education?",
            "I'm a student in Guwahati and receive a monthly stipend. How can I best manage my expenses to stretch it further and save for future educational needs?",
            "Are there any scholarships specifically for graduation or post-graduation studies in Assam, and what are the application deadlines?",
            "What is the official website to apply for scholarships offered by the Assam government?",
            "I belong to a reservation category (SC, ST, OBC, EBC) in Assam. Are there any additional scholarships available for me, or do these categories have different application processes?",
            "Are there any government schemes in Assam that provide financial assistance for skill development programs for women?",
            "Can you tell me more about the Arundhati Kataki Scholarship Scheme and its eligibility criteria for female students?"
        ]
    }

        
    }

    # Display prompt templates based on the selected state
    if selected_state in state_templates:
        st.write(f"Selected State: {selected_state}")
        st.write(f"Choose from the following prompt templates related to {selected_state}:")
        for category, templates in state_templates[selected_state].items():
            st.subheader(category)
            for template_description in templates:
                if st.button(template_description):
                    st.session_state.messages.append({"role": "user", "content": template_description})
                    with st.chat_message("user"):
                        st.markdown(template_description)

                    with st.chat_message("assistant"):
                        response = query(template_description, client)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please select a state to view prompt templates.")


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
        expense_amount = st.number_input("Amount", min_value=0.0, step=1.00)
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
    st.write("Total: <span style='color:green'>{:.2f}</span>".format(st.session_state.total_expenses["total"]), unsafe_allow_html=True)
    st.write("Necessary: <span style='color:green'>{:.2f}</span>".format(st.session_state.total_expenses["necessary"]), unsafe_allow_html=True)
    st.write("Avoidable: <span style='color:red'>{:.2f}</span>".format(st.session_state.total_expenses["avoidable"]), unsafe_allow_html=True)
    
    
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
    # Prepare the DB with HuggingFaceBgeEmbeddings.
    embedding_function = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Direct from Claude")
        answer = get_response(query_text, client)
        return answer

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("From context")
    response = get_response(prompt, client)
    return response

INTRODUCTION_PROMPTS = [
    r'hello',
    r'hey',
    r'namaste'
]


def get_response(prompt, client):
    """
    Retrieves response from the Anthropic model based on the prompt.
    """

    prompt_lower = prompt.lower()
    for pattern in INTRODUCTION_PROMPTS:
        if re.search(pattern, prompt_lower):
            return "Namaste! I'm your financial assistant, developed by the Finance team at 100GIGA and powered by Anthropic's Claude AI model. My purpose is to provide you with expert financial guidance, enhancing your financial literacy and addressing your needs. Feel free to ask me anything related to finance, and I'll be here to assist you every step of the way."
    else:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return ''.join(block.text for block in message.content)

# Rest of the code remains the same...

if __name__ == "__main__":
    main()