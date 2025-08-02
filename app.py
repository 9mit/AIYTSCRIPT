# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# The GOOGLE_API_KEY is now loaded from the .env file
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please add it to your .env file and ensure the file is in the same directory.")
    st.stop()

# --- LangChain Components ---
try:
    # Initialize the LLM - Updated to a newer model to resolve the 404 error
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Define the Prompt Template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="""
    Write a full YouTube video script on the topic: "{topic}".
    Make it informative, engaging, and beginner-friendly.
    Include an intro, main content, and conclusion.
    """
    )

    # Create the LLMChain
    script_chain = LLMChain(llm=llm, prompt=prompt)

except Exception as e:
    st.error(f"Failed to initialize the AI model. Please check your API key and network connection. Error: {e}")
    st.stop()


def generate_script(topic):
    """Generates a script using the LLM chain."""
    # The .invoke method is the recommended way to run chains
    response = script_chain.invoke({"topic": topic})
    return response['text']

# --- Streamlit UI ---
st.title("🎬 YouTube Script Generator")
st.caption("Powered by LangChain and Google Gemini")

user_input = st.text_input("Enter a video topic:", placeholder="e.g., The science of sleep")

if st.button("Generate Script", type="primary"):
    if user_input:
        with st.spinner("Generating your script... 🤖"):
            try:
                script = generate_script(user_input)
                st.subheader("Generated Script:")
                st.write(script)
            except Exception as e:
                st.error(f"An error occurred during script generation: {e}")
    else:
        st.warning("Please enter a topic first.")

