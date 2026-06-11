import PyPDF2
import docx
import re
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_KEYWORDS = [
    "python", "java", "c++", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "nlp", "data analysis", "pandas", "numpy",
    "react", "node.js", "aws", "docker", "kubernetes", "git", "agile",
    "scrum", "fastapi", "flask", "django", "rest api", "html", "css",
    "javascript", "tableau", "power bi", "excel", "communication",
    "leadership", "project management", "linux", "azure", "gcp",
    "computer vision", "data structures", "algorithms", "oop"
]


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_keywords(text, custom_skills=SKILL_KEYWORDS):
    text_clean = clean_text(text)
    found = [skill for skill in custom_skills if skill in text_clean]
    return found


def compute_match_score(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_clean, jd_clean])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)


def find_missing_keywords(resume_text, jd_text):
    resume_clean = clean_text(resume_text)
    jd_keywords = extract_keywords(jd_text)

    missing = [kw for kw in jd_keywords if kw not in resume_clean]
    present = [kw for kw in jd_keywords if kw in resume_clean]

    return present, missing


def generate_suggestions(missing_keywords, match_score):
    suggestions = []

    if match_score < 50:
        suggestions.append("⚠️ Your resume has low alignment with this job. Consider major revisions.")
    elif match_score < 75:
        suggestions.append("Your resume is moderately aligned. A few tweaks can improve your match.")
    else:
        suggestions.append("✅ Strong match! Minor tweaks recommended.")

    if missing_keywords:
        suggestions.append(f"\n📌 Add these missing keywords naturally into your experience/skills section:")
        for kw in missing_keywords[:10]:
            suggestions.append(f"  • {kw}")
        suggestions.append(f"\n💡 Example: 'Utilized {missing_keywords[0]} to develop and optimize project workflows.'")

    suggestions.append("\n📝 General Tips:")
    suggestions.append("  • Use bullet points starting with action verbs (Developed, Implemented, Optimized)")
    suggestions.append("  • Quantify achievements (e.g., 'Improved accuracy by 15%')")
    suggestions.append("  • Mirror exact phrasing from the job description where truthful")

    return "\n".join(suggestions)


def analyze_resume(resume_file_path, jd_text):
    resume_text = extract_text(resume_file_path)

    match_score = compute_match_score(resume_text, jd_text)
    present, missing = find_missing_keywords(resume_text, jd_text)
    suggestions = generate_suggestions(missing, match_score)

    result = f"""
{'='*50}
📊 ATS MATCH SCORE: {match_score}%
{'='*50}

✅ Keywords Found ({len(present)}):
{', '.join(present) if present else 'None'}

❌ Missing Keywords ({len(missing)}):
{', '.join(missing) if missing else 'None'}

{'='*50}
{suggestions}
{'='*50}
"""
    return result


def gradio_interface(resume_file, jd_text):
    if resume_file is None or not jd_text.strip():
        return "Please upload a resume and paste a job description."

    return analyze_resume(resume_file.name, jd_text)


demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload Resume (PDF/DOCX)", file_types=[".pdf", ".docx"]),
        gr.Textbox(label="Paste Job Description", lines=10, placeholder="Paste the job description here...")
    ],
    outputs=gr.Textbox(label="ATS Analysis Result", lines=20),
    title="📄 Resume ATS Scorer",
    description="Upload your resume and a job description to get a match score, missing keywords, and improvement suggestions.",
    theme="soft"
)

demo.launch()