import os
import json
from datetime import datetime
import sys
from pathlib import Path
import streamlit as st
sys.path.append(str(Path(__file__).resolve().parents[1]))
from memory.memory_store import MemoryStore
from retriever.hybrid_retriever import query_hybrid_text
from pipelines.image_ingest import query_image
from pipelines.sql_pipeline import query_sql

# -- Setup ---
st.set_page_config(page_title="Capstone RAG System", layout="wide")
st.title("📚 Capstone RAG System")
memory = MemoryStore(max_len=5)

LOG_FILE = "CHAT-LOGS.json"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f, indent=2)

# ----------------- Helper Functions -----------------
def log_interaction(query, answer, confidence, mode):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "query": query,
        "answer": answer,
        "confidence": confidence
    }
    with open(LOG_FILE, "r+") as f:
        logs = json.load(f)
        logs.append(log_entry)
        f.seek(0)
        json.dump(logs, f, indent=2)

def refine_answer(answer, confidence, history):
    if confidence < 0.5:
        context = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        answer += f"\n\n(Note: confidence was low, context included.)\n{context}"
        confidence = min(confidence + 0.2, 1.0)
    return answer, confidence

# ----------------- Sidebar -----------------
st.sidebar.header("Query Mode")
mode = st.sidebar.radio("Choose query type", ["Text", "Image", "SQL"])

# ----------------- Main UI -----------------
if mode == "Text":
    user_query = st.text_area("Ask a question about the documents:", "")
    if st.button("Submit"):
        if user_query.strip():
            memory.add_message("user", user_query)
            res = query_hybrid_text(user_query)
            answer, confidence = refine_answer(res["answer"], res["confidence"], memory.get_history())
            memory.add_message("assistant", answer)
            st.markdown(f"**Answer (Confidence: {confidence:.2f})**:\n{answer}")
            log_interaction(user_query, answer, confidence, mode)

elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an image or PDF:", type=["png","jpg","jpeg"])
    if st.button("Submit"):
        if uploaded_file:
            # Save temp file
            tmp_path = f"temp_{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            memory.add_message("user", uploaded_file.name)
            res = query_image(tmp_path)
            answer, confidence = refine_answer(res["answer"], res["confidence"], memory.get_history())
            memory.add_message("assistant", answer)
            st.markdown(f"**Answer (Confidence: {confidence:.2f})**:\n{answer}")
            log_interaction(uploaded_file.name, answer, confidence, mode)

            os.remove(tmp_path)

elif mode == "SQL":
    user_query = st.text_area("Ask a question about the database:", "")
    if st.button("Submit"):
        if user_query.strip():
            memory.add_message("user", user_query)
            res = query_sql(user_query)
            answer, confidence = refine_answer(res["answer"], res["confidence"], memory.get_history())
            memory.add_message("assistant", answer)
            st.markdown(f"**Answer (Confidence: {confidence:.2f})**:\n{answer}")
            log_interaction(user_query, answer, confidence, mode)