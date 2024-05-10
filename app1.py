import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
import os
import google.generativeai as genai  # Import genai module
import re  # Import the re module for regular expressions

os.environ['ALLOW_DANGEROUS_DESERIALIZATION'] = 'True'
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to preprocess text (remove extra spaces, newlines, etc.)
def preprocess_text(text):
    # Remove extra spaces, newlines, and other special characters
    text = re.sub(r'\s+', ' ', text)  # Use the re module for regular expressions
    return text

# Function to extract question-answer pairs from PDF text
def extract_qa_pairs_from_pdf_text(pdf_text):
    # Preprocess the PDF text
    pdf_text = preprocess_text(pdf_text)
    
    # Regular expression pattern to match questions and answers
    pattern = r"(?P<question>.*?\?)\s*(?P<answer>.*?)(?=\n\n|$)"
    matches = re.finditer(pattern, pdf_text, re.DOTALL)
    qa_pairs = []
    for match in matches:
        qa_pairs.append({
            "question": match.group("question").strip(),
            "answer": match.group("answer").strip()
        })
    return qa_pairs

# Function to get answer from the provided question
def get_answer(question, qa_pairs):
    for pair in qa_pairs:
        if question.lower() in pair['question'].lower():
            return pair['answer']
    return "Answer not found."

# Main Streamlit app
def main():
    st.set_page_config("CHAT BOT")
    st.header("CHAT BOT")

    user_question = st.text_input("Ask a Question from the PDF Files")

    pdf_file ="pdf.pdf"
    if pdf_file:
        ask_button = st.button("Ask Question")
        if ask_button and user_question:
            with st.spinner("Processing..."):
                # Extract text from PDF file
                pdf_text = extract_text_from_pdf([pdf_file])
                
                # Extract question-answer pairs from PDF text
                qa_pairs = extract_qa_pairs_from_pdf_text(pdf_text)
                
                if qa_pairs:
                    st.write("Chatbot Initialized!")

                    # Initialize vector store for similarity search
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector_store = FAISS.from_texts([pair['answer'] for pair in qa_pairs], embedding=embeddings)

                    # Initialize Google Generative AI model
                    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
                    prompt_template = """
                    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
                    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
                    Context:\n {context}?\n
                    Question: \n{question}\n

                    Answer:
                    """

                    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                    response = chain({"input_documents": vector_store.similarity_search(user_question), "question": user_question}, return_only_outputs=True)
                    st.write("Bot Answer from PDF:", response["output_text"])
                else:
                    st.write("No question-answer data found in the PDF.")

if __name__ == "__main__":
    main()
