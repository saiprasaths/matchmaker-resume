import streamlit as st
import os
import tempfile
import uuid
from PyPDF2 import PdfReader
import docx
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {device}")

# Load the embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load the LLM model
@st.cache_resource
@st.cache_resource
def load_llm_model():
    model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" if device == "cpu" else "mistralai/Mistral-7B-Instruct-v0.2"
    if device == "cuda":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
    else:
        from ctransformers import AutoModelForCausalLM as CT_AutoModelForCausalLM
        pipe = pipeline(
            "text-generation",
            model=CT_AutoModelForCausalLM.from_pretrained(
                model_id,
                model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                hf=True,
                context_length=4096
            ),
            max_new_tokens=1024
        )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Initialize the embedding model and LLM
embedding_model = load_embedding_model()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.write(pdf_file.read())
    temp_file.close()
    
    pdf_reader = PdfReader(temp_file.name)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    os.unlink(temp_file.name)
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
    temp_file.write(docx_file.read())
    temp_file.close()
    
    doc = docx.Document(temp_file.name)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    os.unlink(temp_file.name)
    return text

# Function to extract key information from resume using patterns
def extract_resume_info(text):
    # Basic extraction using regex patterns
    contact_pattern = r'(?:\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)|(?:\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b)'
    education_pattern = r'(?i)(?:(?:bachelor|master|phd|doctor|associate|bs|ba|ms|mba|phd|md|degree|diploma)\s+(?:of|in)?\s+[a-z\s]+)|(?:university|college|institute|school)\s+of\s+[a-z\s]+'
    skills_pattern = r'(?i)(?:skills|technical skills|core competencies|expertise)(?:[\s\S]*?)(?:\n\n|\Z)'
    
    contact_info = re.findall(contact_pattern, text)
    education = re.findall(education_pattern, text)
    
    # Extract skills section
    skills_match = re.search(skills_pattern, text)
    skills = []
    if skills_match:
        skills_text = skills_match.group(0)
        # Split by commas, bullets, or new lines to get individual skills
        skills = re.split(r',|\n|•|-|■|●', skills_text)
        skills = [skill.strip() for skill in skills if skill.strip() and len(skill.strip()) > 2]
    
    # Extract work experience with a simple approach
    experience = []
    exp_sections = re.split(r'\n\s*\n', text)  # Split by double newlines
    for section in exp_sections:
        if re.search(r'(?i)(experience|work|employment|career|job)', section) and not re.search(r'(?i)(education|university|college)', section):
            experience.append(section)
    
    return {
        "contact_info": contact_info,
        "education": education,
        "skills": skills,
        "experience": experience
    }

# Function to extract key information from job description
def extract_job_info(text):
    # Look for requirements, responsibilities, and qualifications
    req_pattern = r'(?i)(?:requirements|qualifications|what you need|what we\'re looking for)(?:[\s\S]*?)(?:\n\n|\Z)'
    resp_pattern = r'(?i)(?:responsibilities|duties|what you\'ll do|job description|role)(?:[\s\S]*?)(?:\n\n|\Z)'
    
    requirements = []
    req_match = re.search(req_pattern, text)
    if req_match:
        req_text = req_match.group(0)
        req_items = re.split(r'\n|•|-|■|●', req_text)
        requirements = [req.strip() for req in req_items if req.strip() and len(req.strip()) > 5]
    
    responsibilities = []
    resp_match = re.search(resp_pattern, text)
    if resp_match:
        resp_text = resp_match.group(0)
        resp_items = re.split(r'\n|•|-|■|●', resp_text)
        responsibilities = [resp.strip() for resp in resp_items if resp.strip() and len(resp.strip()) > 5]
    
    # Extract skills from the requirements
    skills = []
    skill_triggers = ['experience with', 'knowledge of', 'familiarity with', 'proficiency in', 'skilled in']
    for req in requirements:
        for trigger in skill_triggers:
            if trigger in req.lower():
                skills.append(req)
                break
    
    return {
        "requirements": requirements,
        "responsibilities": responsibilities,
        "skills": skills
    }

# Function to calculate match score
def calculate_match_score(resume_info, job_info, embedding_model):
    # Convert the resume skills to embeddings
    if not resume_info.get("skills") or not job_info.get("requirements"):
        return 50, "Insufficient data for accurate matching"
    
    resume_skill_embeddings = embedding_model.encode(resume_info["skills"])
    job_req_embeddings = embedding_model.encode(job_info["requirements"])
    
    # Calculate similarity matrix
    similarity_scores = []
    for resume_emb in resume_skill_embeddings:
        scores = []
        for job_emb in job_req_embeddings:
            score = np.dot(resume_emb, job_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(job_emb))
            scores.append(score)
        if scores:
            similarity_scores.append(max(scores))
    
    # Calculate average of top matches
    if similarity_scores:
        match_score = int(np.mean(similarity_scores) * 100)
        return match_score, "Score based on semantic similarity between resume skills and job requirements"
    else:
        return 50, "Unable to calculate match score using embeddings"

# Function to generate detailed analysis and recommendations
# def generate_analysis(resume_text, job_text, match_score, llm):
#     prompt_template = """
#     You are an expert AI career advisor and recruitment specialist. You need to analyze a resume and a job description to provide insights.
    
#     Resume:
#     {resume_text}
    
#     Job Description:
#     {job_text}
    
#     The calculated match score is: {match_score}%
    
#     Please provide:
#     1. An analysis of the match between the candidate and the job (3-5 points)
#     2. Key strengths of the candidate for this position (2-3 points)
#     3. Areas where the candidate could improve to better match the job requirements (2-3 points)
#     4. Specific suggestions to improve the resume for this job application (2-3 actionable tips)
    
#     Format your response with clear headings and bullet points.
#     """
    
#     prompt = PromptTemplate(
#         input_variables=["resume_text", "job_text", "match_score"],
#         template=prompt_template
#     )
    
#     chain = LLMChain(llm=llm, prompt=prompt)
    
#     # Truncate text to avoid context length issues
#     max_length = 2048
#     resume_truncated = resume_text[:max_length] if len(resume_text) > max_length else resume_text
#     job_truncated = job_text[:max_length] if len(job_text) > max_length else job_text
    
#     response = chain.run({
#         "resume_text": resume_truncated,
#         "job_text": job_truncated,
#         "match_score": match_score
#     })
    
#     return response
def generate_analysis(resume_text, job_text, match_score, llm):
    prompt_template = """
    You are an expert AI career advisor and recruitment specialist. You need to analyze a resume and a job description to provide insights.
    Resume:
    {resume_text}
    Job Description:
    {job_text}
    The calculated match score is: {match_score}%
    Please provide:
    1. An analysis of the match between the candidate and the job (3-5 points)
    2. Key strengths of the candidate for this position (2-3 points)
    3. Areas where the candidate could improve to better match the job requirements (2-3 points)
    4. Specific suggestions to improve the resume for this job application (2-3 actionable tips)
    Format your response with clear headings and bullet points.
    """
    prompt = PromptTemplate(
        input_variables=["resume_text", "job_text", "match_score"],
        template=prompt_template
    )
    
    # Use RunnableSequence instead of LLMChain
    chain = prompt | llm
    
    # Truncate text to avoid context length issues
    max_length = 2048
    resume_truncated = resume_text[:max_length] if len(resume_text) > max_length else resume_text
    job_truncated = job_text[:max_length] if len(job_text) > max_length else job_text
    
    # Use invoke() instead of run()
    response = chain.invoke({
        "resume_text": resume_truncated,
        "job_text": job_truncated,
        "match_score": match_score
    })
    return response
# Streamlit UI
st.title("MatchMaker: Resume-Job Matching")
st.write("Upload your resume and job description to see how well they match!")

# File uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume")
    resume_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"], key="resume")
    if resume_file:
        file_details = {"Filename": resume_file.name, "FileType": resume_file.type, "FileSize": f"{resume_file.size / 1024:.2f} KB"}
        st.write(file_details)

with col2:
    st.subheader("Job Description")
    job_desc_choice = st.radio("Choose input method:", ["Upload file", "Paste text"])
    
    if job_desc_choice == "Upload file":
        job_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"], key="job")
        if job_file:
            file_details = {"Filename": job_file.name, "FileType": job_file.type, "FileSize": f"{job_file.size / 1024:.2f} KB"}
            st.write(file_details)
            job_text = ""
    else:
        job_text = st.text_area("Paste job description here:", height=300)

# Process files when the button is clicked
if st.button("Analyze Match"):
    with st.spinner("Processing documents..."):
        # Check if we have both inputs
        if not resume_file:
            st.error("Please upload a resume.")
        elif job_desc_choice == "Upload file" and not job_file:
            st.error("Please upload a job description file or paste text.")
        elif job_desc_choice == "Paste text" and not job_text:
            st.error("Please paste job description text.")
        else:
            # Extract text from resume
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)
            
            # Extract text from job description
            if job_desc_choice == "Upload file":
                if job_file.type == "application/pdf":
                    job_text = extract_text_from_pdf(job_file)
                else:
                    job_text = extract_text_from_docx(job_file)
            
            # Extract information from resume and job description
            resume_info = extract_resume_info(resume_text)
            job_info = extract_job_info(job_text)
            
            # Calculate match score
            match_score, score_explanation = calculate_match_score(resume_info, job_info, embedding_model)
            
            # Display the basic matching results
            st.header("Match Results")
            
            # Match score visualization
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("Match Score")
                st.markdown(f"<h1 style='text-align: center; color: {'green' if match_score >= 70 else 'orange' if match_score >= 50 else 'red'};'>{match_score}%</h1>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Quick Analysis")
                if match_score >= 80:
                    st.success("Excellent match! Your profile aligns very well with the job requirements.")
                elif match_score >= 70:
                    st.success("Good match! Your profile aligns well with many job requirements.")
                elif match_score >= 50:
                    st.warning("Moderate match. There is potential, but some improvements could help.")
                else:
                    st.error("Low match. Consider if this role aligns with your career goals or if significant resume adjustments are needed.")
            
            # Show extracted info for debugging
            with st.expander("View Extracted Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("From Resume")
                    st.write("Skills:", ", ".join(resume_info["skills"][:10]) + ("..." if len(resume_info["skills"]) > 10 else ""))
                    st.write("Education:", ", ".join(resume_info["education"][:3]) + ("..." if len(resume_info["education"]) > 3 else ""))
                
                with col2:
                    st.subheader("From Job Description")
                    st.write("Requirements:", ", ".join(job_info["requirements"][:5]) + ("..." if len(job_info["requirements"]) > 5 else ""))
                    st.write("Skills:", ", ".join(job_info["skills"][:5]) + ("..." if len(job_info["skills"]) > 5 else ""))
            
            # Load LLM for detailed analysis
            with st.spinner("Generating detailed analysis..."):
                try:
                    llm = load_llm_model()
                    analysis = generate_analysis(resume_text, job_text, match_score, llm)
                    
                    st.header("Detailed Analysis & Recommendations")
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Error generating detailed analysis: {str(e)}")
                    st.info("Proceeding with basic analysis only.")
                    
                    # Provide a basic analysis if LLM fails
                    st.header("Basic Analysis")
                    st.subheader("Key Skills Present")
                    matching_skills = []
                    for r_skill in resume_info["skills"]:
                        for j_skill in job_info["skills"]:
                            if r_skill.lower() in j_skill.lower() or j_skill.lower() in r_skill.lower():
                                matching_skills.append(r_skill)
                                break
                    st.write(", ".join(matching_skills[:5]) + ("..." if len(matching_skills) > 5 else ""))
                    
                    st.subheader("Potential Skill Gaps")
                    missing_skills = []
                    for j_skill in job_info["skills"]:
                        found = False
                        for r_skill in resume_info["skills"]:
                            if r_skill.lower() in j_skill.lower() or j_skill.lower() in r_skill.lower():
                                found = True
                                break
                        if not found:
                            missing_skills.append(j_skill)
                    st.write(", ".join(missing_skills[:5]) + ("..." if len(missing_skills) > 5 else ""))