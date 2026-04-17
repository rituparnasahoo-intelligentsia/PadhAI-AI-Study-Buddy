🎓 PadhAI — Your AI Study Buddy (NVIDIA NIM Version)
=====================================================
Uses the NVIDIA NIM API with llama-3.1-nemotron-ultra-253b-v1 (or any NIM model)
The NVIDIA NIM API is OpenAI-compatible, so we use the openai SDK.

Get your FREE API key at: https://build.nvidia.com/explore/discover
(Sign up → Get API Key → Free credits included!)

Run:
    pip install openai gradio
    python padhai_nvidia.py
"""

import getpass
import time
import gradio as gr
from openai import OpenAI, RateLimitError, APIError

# ── API Key Setup ─────────────────────────────────────────────
print("=" * 60)
print("🔑 Get your FREE NVIDIA API key →")
print("   https://build.nvidia.com/explore/discover")
print("=" * 60)
api_key = getpass.getpass("Paste your NVIDIA NIM API Key: ")

# NVIDIA NIM is OpenAI-compatible — just change the base_url
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

# ── Available NVIDIA NIM Models (pick one) ────────────────────
# Free tier models (as of 2025):
#   "meta/llama-3.1-70b-instruct"          ← Fast, great quality
#   "meta/llama-3.3-70b-instruct"          ← Latest Llama 3.3
#   "nvidia/llama-3.1-nemotron-70b-instruct" ← NVIDIA tuned
#   "mistralai/mistral-7b-instruct-v0.3"   ← Lightweight
#   "google/gemma-2-27b-it"                ← Google Gemma

MODEL = "meta/llama-3.1-70b-instruct"   # ✅ Free tier, fast & smart
print(f"✅ Connected to NVIDIA NIM! Using model: {MODEL}\n")


# ── Core AI Function with Retry Logic ──────────────────────────
def ask_padhai(question: str, student_class: str, language: str) -> str:
    grade_map = {
        "Class 6":  "Middle school, age ~11",
        "Class 7":  "Middle school, age ~12",
        "Class 8":  "Middle school, age ~13",
        "Class 9":  "High school, age ~14",
        "Class 10": "Board exam year, age ~15",
        "Class 11": "Pre-university, age ~16",
        "Class 12": "Board + entrance exam year, age ~17",
        "engineering": "Engineering students, age ~ 23",
        "medical": "Medical students , age ~ 25",
    }
    grade_level = grade_map.get(student_class, "High school student")

    system_prompt = f"""You are PadhAI 🎓, a warm and encouraging AI study buddy for Indian students.
You are helping a {student_class} student ({grade_level}).
Always respond in {language}.

Your expertise:
- 📚 Scholarships: NSP, INSPIRE, Pragati, Saksham, state scholarships
- 🏆 Olympiads: IMO, NSO, NCO, IEO, NTSE, KVPY, RMO
- 🎯 Entrance Exams: JEE, NEET, GATE, CLAT, CAT, NDA
- 💼 Internships, hackathons, and career guidance
- 📅 Important deadlines and application procedures

Tone: friendly, concise, encouraging. Use bullet points for lists.
Always mention eligibility, deadlines, and official websites when relevant."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    # Retry up to 3 times with exponential backoff
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=700,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except RateLimitError:
            wait = 2 ** attempt * 10  # 10s, 20s, 40s
            print(f"⚠️ Rate limit hit. Retrying in {wait}s (attempt {attempt+1}/3)...")
            time.sleep(wait)
            continue

        except APIError as e:
            return f"⚠️ NVIDIA API Error: {str(e)}"

        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"

    return "⚠️ The service is currently busy. Please wait a minute and try again."


def student_qa_agent(language, student_class, question, history):
    if not question.strip():
        return history, ""
    answer = ask_padhai(question, student_class, language)
    history.append((question, answer))
    return history, ""


# ── Custom CSS ────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;800&family=Nunito:wght@400;600;700&display=swap');

body, .gradio-container {
    font-family: 'Nunito', sans-serif !important;
    background: linear-gradient(135deg, #0a0a1a, #1a1040, #0d1a2e) !important;
    min-height: 100vh;
}
.app-header {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    background: linear-gradient(90deg, #76b900, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Baloo 2', cursive !important;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
}
.app-subtitle {
    text-align: center;
    color: #88ccff;
    font-size: 1rem;
    margin-bottom: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.stat-badge {
    display: inline-block;
    background: rgba(118,185,0,0.12);
    border: 1px solid rgba(118,185,0,0.4);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    color: #a8e063;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 0.2rem;
}
.free-badge {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.5);
    border-radius: 8px;
    padding: 0.3rem 1rem;
    color: #00d4aa;
    font-size: 0.85rem;
    font-weight: 700;
    margin: 0.5rem auto;
}
label span, .block label span {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
textarea, input[type=text] {
    background: rgba(255,255,255,0.06) !important;
    border: 2px solid rgba(118,185,0,0.35) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 3px rgba(118,185,0,0.2) !important;
}
#submit-btn {
    background: linear-gradient(135deg, #76b900, #00d4aa) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #0a0a1a !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 20px rgba(118,185,0,0.35) !important;
    font-family: 'Baloo 2', cursive !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(118,185,0,0.55) !important;
}
#clear-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 2px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
}
#clear-btn:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
}
.chatbot {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(118,185,0,0.15) !important;
    border-radius: 20px !important;
}
.quick-btn button {
    background: rgba(0,212,170,0.1) !important;
    border: 1px solid rgba(0,212,170,0.35) !important;
    border-radius: 25px !important;
    color: #7fffd4 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s;
    font-family: 'Nunito', sans-serif !important;
}
.quick-btn button:hover {
    background: rgba(0,212,170,0.28) !important;
    color: white !important;
    transform: translateY(-1px);
}
.footer-text {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.75rem;
    padding: 1rem;
    margin-top: 1rem;
}
"""

QUICK_QUESTIONS = {
    "🏆 Olympiads": [
        "What Olympiads can I join in Class 8?",
        "How do I prepare for IMO?",
        "What is NTSE and how to apply?",
        "Tell me about KVPY scholarship",
    ],
    "💰 Scholarships": [
        "What scholarships are available for Class 10?",
        "How to apply for NSP scholarship?",
        "Tell me about INSPIRE scholarship",
        "Scholarships for girls in India",
    ],
    "🎯 Career Guidance": [
        "What should I do after Class 10?",
        "How to prepare for JEE from Class 11?",
        "Career options after Class 12 science",
        "GATE CSE syllabus overview",
    ],
}


# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(
    css=custom_css,
    title="PadhAI — Your AI Study Buddy (NVIDIA NIM)",
    theme=gr.themes.Base(
        primary_hue="green",
        secondary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Nunito"),
    )
) as demo:

    gr.HTML("""
    <div class="app-header">🎓 PadhAI</div>
    <div class="app-subtitle">✨ Your Personal AI Study Buddy ✨</div>
    <div style="text-align:center; margin-bottom:0.5rem;">
        <span class="free-badge">🟢 Powered by NVIDIA</span>
    </div>
    <div style="text-align:center; margin-bottom:1.5rem;">
        <span class="stat-badge">📚 Scholarships</span>
        <span class="stat-badge">🏆 Olympiads</span>
        <span class="stat-badge">🎯 Career Guidance</span>
        <span class="stat-badge">📅 Exam Deadlines</span>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<p style='color:#00d4aa;font-weight:700;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 1rem;'>⚙️ Settings</p>")

            language = gr.Radio(
                choices=["English", "Hindi"],
                value="English",
                label="🌐 Preferred Language"
            )
            student_class = gr.Dropdown(
                choices=["Class 6","Class 7","Class 8","Class 9",
                         "Class 10","Class 11","Class 12"],
                value="Class 9",
                label="🎓 Your Class"
            )

            gr.HTML("<hr style='border-color:rgba(255,255,255,0.08);margin:1.2rem 0;'>")
            gr.HTML("<p style='color:#00d4aa;font-weight:700;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.8rem;'>⚡ Quick Questions</p>")

            quick_btns = []
            for category, questions in QUICK_QUESTIONS.items():
                gr.HTML(f"<p style='color:#4a6060;font-size:0.75rem;font-weight:700;margin:0.5rem 0 0.3rem;'>{category}</p>")
                for q in questions:
                    b = gr.Button(q, size="sm", elem_classes="quick-btn")
                    quick_btns.append((b, q))

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="🤖 PadhAI Says",
                height=420,
                bubble_full_width=False,
                elem_classes="chatbot",
                placeholder="<div style='text-align:center;color:#4a6060;padding:2rem;'>👋 Hi! I'm PadhAI. Ask me anything about scholarships, Olympiads, or career guidance!</div>"
            )
            with gr.Row():
                question = gr.Textbox(
                    placeholder="e.g., GATE CSE syllabus, JEE preparation tips... 💬",
                    label="💬 Your Question",
                    lines=2,
                    scale=4
                )
            with gr.Row():
                clear_btn  = gr.Button("🗑️ Clear Chat", elem_id="clear-btn", scale=1)
                submit_btn = gr.Button("🚀 Ask PadhAI", elem_id="submit-btn",
                                       scale=2, variant="primary")

    gr.HTML("""
    <div class="footer-text">
        🎓 PadhAI v4.0 · Built with ❤️ for Indian Students · NVIDIA NIM (LLaMA 3.1 70B)<br>
        Get your free API key → <a href="https://build.nvidia.com/explore/discover"
        style="color:#00d4aa;" target="_blank">build.nvidia.com</a>
    </div>
    """)

    history_state = gr.State([])

    submit_btn.click(
        fn=student_qa_agent,
        inputs=[language, student_class, question, history_state],
        outputs=[chatbot, question]
    ).then(fn=lambda h: h, inputs=[chatbot], outputs=[history_state])

    question.submit(
        fn=student_qa_agent,
        inputs=[language, student_class, question, history_state],
        outputs=[chatbot, question]
    ).then(fn=lambda h: h, inputs=[chatbot], outputs=[history_state])

    clear_btn.click(
        fn=lambda: ([], [], ""),
        outputs=[chatbot, history_state, question]
    )

    for btn, q_text in quick_btns:
        btn.click(fn=lambda q=q_text: q, outputs=[question])


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(debug=True, share=True, show_error=True)
