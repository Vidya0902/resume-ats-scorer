# 📄 Resume ATS Scorer

An NLP-powered tool that analyzes how well a resume matches a job description — just like an Applicant Tracking System (ATS) would.

## Features
- 📊 Calculates a match score using TF-IDF and cosine similarity
- ✅ Identifies skills/keywords present in both resume and job description
- ❌ Highlights missing keywords critical to the job
- 💡 Generates personalized rewrite suggestions to improve resume alignment
- 📁 Supports PDF and DOCX resume uploads

## Tech Stack
- Python, scikit-learn (TF-IDF, cosine similarity)
- PyPDF2, python-docx for document parsing
- Gradio for the web interface
- Deployed on Hugging Face Spaces

## Live Demo
🔗 [Try it here](https://huggingface.co/spaces/vidyakv679122/resume-ats-scorer)

## How It Works
1. Upload your resume (PDF/DOCX)
2. Paste the job description
3. Get an instant match score, missing keywords, and improvement tips
