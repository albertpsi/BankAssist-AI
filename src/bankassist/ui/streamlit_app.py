"""Minimal Streamlit UI for the basic RAG pipeline (FR-L2-10).

Question textbox -> Ask button -> answer -> sources. Nothing else: no chat
history, no tabs, no settings panel. It calls ``POST /rag/query`` over HTTP —
it never imports ``bankassist.rag`` — so it exercises the same contract a real
client would, and the two processes can run on different machines.

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
    "Lab 2: basic RAG over the banking policy corpus. Answers are grounded in "
    "retrieved policy text only."
)

question = st.text_input("Ask a question about banking policy:")

if st.button("Ask", disabled=not question.strip()):
    with st.spinner("Retrieving and answering..."):
        try:
            response = httpx.post(query_url, json={"question": question}, timeout=60.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("error", {}).get("message", exc.response.text)
            st.error(f"Request failed ({exc.response.status_code}): {detail}")
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API at {query_url}: {exc}")
        else:
            st.subheader("Answer")
            st.write(payload["answer"])

            st.subheader("Sources")
            sources = payload["sources"]
            if sources:
                for source in sources:
                    st.markdown(f"- {source}")
            else:
                st.caption("(none)")
