import streamlit as st
from dotenv import load_dotenv
import os, re, json
from openai import OpenAI
import fitz  # PyMuPDF

# ── Setup ─────────────────────────────────────
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OpenRouter_API_KEY")
client = OpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:8501"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Job Application Assistant"),
    },
)
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

application_info: dict[str, str | None] = {"name": None, "email": None, "skills": None}


# ── Core functions ────────────────────────────

def extract_application_info(text: str) -> str:
    name_match   = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE)
    email_match  = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)
    skills_match = re.search(r"(?:skills are|i know|i can use)\s+(.+)", text, re.IGNORECASE)

    if name_match:
        application_info["name"]   = name_match.group(1).title()
    if email_match:
        application_info["email"]  = email_match.group(0)
    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()

    return "Got it. Let me check what else I need."


def check_application_goal() -> str:
    if all(application_info.values()):
        return (
            f"✅ You're ready! "
            f"Name: {application_info['name']}, "
            f"Email: {application_info['email']}, "
            f"Skills: {application_info['skills']}."
        )
    missing = [k for k, v in application_info.items() if not v]
    return f"⏳ Still need: {', '.join(missing)}"


def extract_text_from_pdf(uploaded_file) -> str:
    doc  = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join(str(page.get_text()) for page in doc)
    doc.close()
    return text


def extract_info_from_cv(text: str) -> dict:
    extracted: dict[str, str | None] = {"name": None, "email": None, "skills": None}
    name_match   = re.search(r"(?:Full Name:|Name:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    email_match  = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)
    skills_match = re.search(r"Skills\s*-+\s*(.*?)\n(?:Projects|Certifications|$)", text, re.DOTALL)

    if name_match:
        extracted["name"]  = name_match.group(1).strip()
    if email_match:
        extracted["email"] = email_match.group(0).strip()
    if skills_match:
        raw = skills_match.group(1).replace("\n", ", ").replace("\u2022", "").replace("-", "")
        extracted["skills"] = re.sub(r"\s+", " ", raw.strip())
    return extracted


def run_agent(user_input: str, history: list) -> str:
    """Generate the next assistant reply with OpenRouter."""
    extract_application_info(user_input)

    transcript = []
    for message in history:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            transcript.append(f"User: {content}")
        elif role == "assistant":
            transcript.append(f"Assistant: {content}")

    prompt = (
        "You are a helpful job application assistant. "
        "Collect the user's name, email, and skills. "
        "Ask for missing details one at a time and keep replies concise.\n\n"
        f"Known information:\n"
        f"Name: {application_info['name'] or 'Not provided'}\n"
        f"Email: {application_info['email'] or 'Not provided'}\n"
        f"Skills: {application_info['skills'] or 'Not provided'}\n\n"
        "Conversation so far:\n"
        + ("\n".join(transcript) if transcript else "No prior conversation.")
        + f"\n\nUser: {user_input}\nAssistant:"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful job application assistant. "
                        "Collect the user's name, email, and skills. "
                        "Ask for missing details one at a time and keep replies concise."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        reply = (response.choices[0].message.content or "I'm not sure what to say. Can you try again?").strip()
        if reply:
            return reply
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "quota" in error_text.lower() or "rate limit" in error_text.lower():
            return (
                "OpenRouter quota or rate limit reached for this key/model. "
                "Try another model or check your OpenRouter billing and limits."
            )

    missing = [key for key, value in application_info.items() if not value]
    if missing:
        return f"I couldn't reach OpenRouter right now. Please share your {', '.join(missing)}."
    return check_application_goal()


# ── Streamlit UI ──────────────────────────────

st.set_page_config(page_title="🎯 Job Application Assistant", layout="centered")
st.title("🧠 Goal-Based Agent: Job Application Assistant")
st.markdown("Tell me your **name**, **email**, and **skills** to complete your application!")

# Session state
for key, default in {
    "chat_history":        [],   # list of (role, text) for display
    "openai_history":      [],   # model history for the assistant
    "goal_complete":       False,
    "download_ready":      True,
    "application_summary": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar — resume upload
st.sidebar.header("📤 Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf"])

if resume:
    st.sidebar.success("Resume uploaded!")
    cv_text   = extract_text_from_pdf(resume)
    extracted = extract_info_from_cv(cv_text)
    for key in application_info:
        if extracted[key]:
            application_info[key] = extracted[key]
    st.sidebar.info("🔍 Extracted info from resume:")
    for key, value in extracted.items():
        st.sidebar.markdown(f"**{key.capitalize()}:** {value}")

# Sidebar — reset
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history.clear()
    st.session_state.openai_history.clear()
    st.session_state.goal_complete       = False
    st.session_state.download_ready      = False
    st.session_state.application_summary = ""
    for key in application_info:
        application_info[key] = None
    st.rerun()

# Chat input
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    bot_reply = run_agent(user_input, st.session_state.openai_history)

    # Save to model history for next turn
    st.session_state.openai_history.append({"role": "user",      "content": user_input})
    st.session_state.openai_history.append({"role": "assistant",  "content": bot_reply})

    st.session_state.chat_history.append(("bot", bot_reply))

    goal_status = check_application_goal()
    st.session_state.chat_history.append(("status", goal_status))

    if "you're ready" in goal_status.lower():
        st.session_state.goal_complete = True
        st.session_state.application_summary = (
            f"Name:   {application_info['name']}\n"
            f"Email:  {application_info['email']}\n"
            f"Skills: {application_info['skills']}\n"
        )
        st.session_state.download_ready = True

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "user":
        with st.chat_message("user"):
            st.markdown(message)
    elif sender == "bot":
        with st.chat_message("assistant"):
            st.markdown(message)
    elif sender == "status":
        with st.chat_message("assistant"):
            st.info(message)

# Completion banner
if st.session_state.goal_complete:
    st.success("🎉 All information collected! You're ready to apply!")

# Download button
if st.session_state.download_ready:
    st.download_button(
        label="📥 Download Application Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt",
        mime="text/plain",
    )