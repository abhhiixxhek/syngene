import streamlit as st
import os
import json
import tempfile
import sys

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ingestion.indexer import ReferenceIndexer
from src.verification.verifier import SOPVerifier

st.set_page_config(page_title="SOP Compliance Engine (Refined)", layout="wide")

st.title("🛡️ SOP Compliance Engine (Strict Mode)")
st.markdown("### Refined Logic: 0.90 Auto-Pass | Gemini Jury for All Else")

# Sidebar
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Local Testing", "AWS S3 Production"])
bucket_name = "local" if mode == "Local Testing" else st.sidebar.text_input("Bucket", "my-compliance-bucket")

tab1, tab2 = st.tabs(["1. Ingest References (Truth)", "2. Verify SOP"])

# --- TAB 1: INGESTION ---
with tab1:
    st.header("Step 1: Build Reference Index")
    st.info("Phase 1: Preparation (Atomic Normalization)")
    uploaded_refs = st.file_uploader("Upload References", type=['pdf', 'docx'], accept_multiple_files=True)
    
    if st.button("Ingest & Normalize"):
        if not uploaded_refs:
            st.error("Please upload reference docs.")
        else:
            with st.spinner("Decomposing text into Atomic Requirements (using Gemini)..."):
                temp_paths = []
                temp_dir = tempfile.mkdtemp()
                for f in uploaded_refs:
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as w:
                        w.write(f.getbuffer())
                    temp_paths.append(path)
                
                try:
                    indexer = ReferenceIndexer(bucket_name=bucket_name)
                    indexer.process_and_index(temp_paths)
                    st.success(f"Processing Complete! Reference Index updated.")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 2: VERIFICATION ---
with tab2:
    st.header("Step 2: Verify SOP")
    st.info("Phase 3: Reverse Verification Loop (Strict)")
    uploaded_sop = st.file_uploader("Upload SOP", type=['pdf', 'docx'])
    
    if st.button("Run Strict Verification"):
        if not uploaded_sop:
            st.error("Please upload an SOP.")
        else:
            status_container = st.empty()
            with st.spinner("Phase 3 Execution: Scanning Requirements..."):
                temp_sop_path = os.path.join(tempfile.gettempdir(), uploaded_sop.name)
                with open(temp_sop_path, "wb") as f:
                    f.write(uploaded_sop.getbuffer())
                
                try:
                    verifier = SOPVerifier(bucket_name=bucket_name)
                    verifier.load_reference_index()
                    verifier.ingest_sop(temp_sop_path)
                    
                    status_container.info("Applying Traffic Light Logic (>0.90 Auto-Pass, <0.90 Gemini Jury)...")
                    gaps = verifier.verify_documents()
                    
                    if not gaps:
                        st.balloons()
                        st.success("✅ fully COMPLIANT! No missing requirements found.")
                    else:
                        st.warning(f"⚠️ Found {len(gaps)} Non-Compliant Items")
                        
                        for gap in gaps:
                            with st.expander(f"[{gap['status']}] {gap['atomic_requirement'][:80]}...", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Reference Requirement (Truth)**")
                                    st.markdown(f"> {gap['full_reference_text']}")
                                    st.caption(f"Source: {gap['reference_source']} | Severity: {gap['severity']}")
                                with col2:
                                    st.markdown("**SOP Evidence Analysis**")
                                    if gap['status'] == "MISSING":
                                        st.error("❌ EVIDENCE MISSING")
                                        st.markdown(f"**LLM Verdict:** {gap['justification']}")
                                    else:
                                        st.warning("⚠️ PARTIAL MATCH")
                                        st.markdown(f"**Evidence:** _{gap['sop_evidence']}_")
                                        st.markdown(f"**LLM Verdict:** {gap['justification']}")
                                
                except Exception as e:
                    st.error(f"Verification Failed: {e}")
