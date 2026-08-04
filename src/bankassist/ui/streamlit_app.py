"""Streamlit UI for the RAG pipeline — basic (FR-L2-10) and enterprise (FR-L3-13).

A chat-style transcript: mode selector -> chat input -> scrolling question/answer
bubbles. Each question is still answered independently by the API -- there is no
conversation memory or history-aware rewriting (that is Lab 4 scope). Enterprise
mode's classification and rewritten query sit behind a per-turn "Details"
expander so they don't clutter the primary answer, per FR-L3-13.2/§11.

It calls ``POST /rag/query`` over HTTP -- it never imports ``bankassist.rag`` --
so it exercises the same contract a real client would.

    streamlit run src/bankassist/ui/streamlit_app.py
"""

from __future__ import annotations

import httpx
import streamlit as st

from bankassist.config import get_settings

st.set_page_config(page_title="BankAssist AI — Policy Q&A", page_icon="🏦")

settings = get_settings()
query_url = f"{settings.api_base_url.rstrip('/')}/rag/query"

st.title("BankAssist AI — Policy Q&A")
st.caption(
    "Basic (Lab 2) vs Enterprise (Lab 3) retrieval over the banking policy corpus. "
    "Answers are grounded in retrieved policy text only. Each question is answered "
    "independently — there is no conversation memory yet."
)

mode = st.radio("Mode", ["Basic", "Enterprise"], horizontal=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            if message.get("classification") is not None:
                with st.expander("Details"):
                    classification = message["classification"]
                    st.write(
                        f"**Classification:** {classification.get('label', '—')} "
                        f"(confidence: {classification.get('confidence', 0):.2f})"
                    )
                    st.write(f"**Rewritten query:** {message.get('rewritten_question', '—')}")

            sources = message.get("sources") or []
            if sources:
                st.caption("Sources: " + ", ".join(sources))
            else:
                st.caption("Sources: (none)")

question = st.chat_input("Ask a question about banking policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"), st.spinner("Retrieving and answering..."):
        try:
            response = httpx.post(
                query_url,
                json={"question": question, "mode": mode.lower()},
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("error", {}).get("message", exc.response.text)
            error_text = f"Request failed ({exc.response.status_code}): {detail}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
        except httpx.HTTPError as exc:
            error_text = f"Could not reach the API at {query_url}: {exc}"
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
        else:
            st.write(payload["answer"])

            classification = payload.get("classification")
            if classification is not None:
                with st.expander("Details"):
                    st.write(
                        f"**Classification:** {classification.get('label', '—')} "
                        f"(confidence: {classification.get('confidence', 0):.2f})"
                    )
                    st.write(
                        f"**Rewritten query:** {payload.get('rewritten_question', '—')}"
                    )

            sources = payload["sources"]
            st.caption("Sources: " + (", ".join(sources) if sources else "(none)"))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": payload["answer"],
                    "sources": sources,
                    "classification": classification,
                    "rewritten_question": payload.get("rewritten_question"),
                }
            )
