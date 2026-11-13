"""Zyro Dynamics HR Help Desk - Streamlit chat app.

Run locally:  streamlit run app.py
"""

import os

import streamlit as st

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💬", layout="centered")

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zyro-dynamics-hr-corpus")

SAMPLE_QUESTIONS = [
    "How many days of casual leave do I get?",
    "Who is eligible for hybrid work from home?",
    "What is the notice period for an L5 employee?",
    "What happens if I lose my company laptop?",
]


def load_secrets() -> None:
    """Copy keys from Streamlit secrets into the environment (env vars win)."""
    for key in ("GROQ_API_KEY", "LANGCHAIN_API_KEY", "GROQ_MODEL"):
        if os.getenv(key):
            continue
        try:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
        except Exception:      # no secrets file at all
            pass


@st.cache_resource(show_spinner="Reading the HR policy documents and building the index…")
def get_bot(groq_key: str):
    # groq_key is only here so the cache is rebuilt when the key changes.
    from rag_pipeline import HRBot

    return HRBot(PDF_DIR)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['document']}** · {s['section'].title()} · page {s['page']}")
            st.caption(s["snippet"][:400] + ("…" if len(s["snippet"]) > 400 else ""))


load_secrets()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Zyro HR Help Desk")
    st.write(
        "Answers come only from Zyro Dynamics' internal HR policy documents, "
        "with the source section shown under each answer."
    )

    if not os.getenv("GROQ_API_KEY"):
        key = st.text_input(
            "Groq API key", type="password",
            help="Free key from console.groq.com. Kept for this session only.",
        )
        if key:
            os.environ["GROQ_API_KEY"] = key.strip()
            st.rerun()

    st.subheader("Topics covered")
    st.markdown(
        "- Leave and holidays\n- Work from home\n- Compensation and benefits\n"
        "- Performance reviews and PIP\n- Code of conduct\n- IT and data security\n"
        "- POSH\n- Onboarding and separation\n- Travel and expenses"
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("For anything not covered here, write to hr.helpdesk@zyrodynamics.com.")

# ---------------------------------------------------------------- main
st.title("HR Help Desk")
st.caption("Ask a question about Zyro Dynamics HR policies.")

if not os.getenv("GROQ_API_KEY"):
    st.info("Add a Groq API key in the sidebar (or in Streamlit secrets) to start chatting.")
    st.stop()

try:
    bot = get_bot(os.environ["GROQ_API_KEY"])
except Exception as exc:
    st.error(f"The help desk could not start: {exc}")
    st.stop()

st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", "\\$"))
        render_sources(msg.get("sources", []))

pending = None
if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            pending = q

question = st.chat_input("Type your HR question…") or pending

if question:
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking the policies…"):
                result = bot.ask(question, chat_history=history)
            answer, sources = result["answer"], result["sources"]
        except Exception as exc:
            answer, sources = f"Sorry, something went wrong while answering: {exc}", []
        st.markdown(answer.replace("$", "\\$"))
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    if pending:
        st.rerun()      # removes the sample-question buttons
