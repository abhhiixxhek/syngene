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

st.set_page_config(page_title="SOP Compliance Engine", layout="wide")

st.title("🛡️ SOP Compliance Verification Engine")
st.markdown("Use this tool to audit SOPs against Reference Documents using a Reverse-Check Architecture.")

# Sidebar Configuration
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Local Testing", "AWS S3 Production"])

if mode == "Local Testing":
    bucket_name = "local"
    st.sidebar.info("Running in **Local Mode**. Index will be saved/loaded from disk.")
else:
    bucket_name = st.sidebar.text_input("S3 Bucket Name", "my-compliance-bucket")

# Tabs
tab1, tab2 = st.tabs(["1. Ingest References (Admin)", "2. Verify SOP (User)"])

# --- TAB 1: INGESTION ---
with tab1:
    st.header("Step 1: Build Reference Index")
    st.markdown("Upload Reference Documents (PDF/DOCX) to build the 'Truth' index.")
    
    uploaded_refs = st.file_uploader("Upload References", type=['pdf', 'docx'], accept_multiple_files=True)
    
    if st.button("Ingest References"):
        if not uploaded_refs:
            st.error("Please upload at least one file.")
        else:
            with st.spinner("Processing documents... (Parsing, Normalizing, Embedding)"):
                # Save uploaded files temporarily
                temp_paths = []
                temp_dir = tempfile.mkdtemp()
                
                for uploaded_file in uploaded_refs:
                    path = os.path.join(temp_dir, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(path)
                
                # Run Indexer
                try:
                    indexer = ReferenceIndexer(bucket_name=bucket_name)
                    indexer.process_and_index(temp_paths)
                    st.success(f"Successfully processed {len(temp_paths)} documents! Index saved.")
                except Exception as e:
                    st.error(f"Ingestion Failed: {e}")

# --- TAB 2: VERIFICATION ---
with tab2:
    st.header("Step 2: Verify SOP")
    st.markdown("Upload an SOP to check for compliance gaps.")
    
    uploaded_sop = st.file_uploader("Upload SOP", type=['pdf', 'docx'])
    
    if st.button("Verify Compliance"):
        if not uploaded_sop:
            st.error("Please upload an SOP file.")
        else:
            with st.spinner("Analyzing Compliance... (This involves LLM Reasoning)"):
                # Save temp
                temp_sop_path = os.path.join(tempfile.gettempdir(), uploaded_sop.name)
                with open(temp_sop_path, "wb") as f:
                    f.write(uploaded_sop.getbuffer())
                
                # Run Verifier
                try:
                    verifier = SOPVerifier(bucket_name=bucket_name)
                    
                    # 1. Load Index
                    verifier.load_reference_index()
                    
                    # 2. Ingest SOP
                    verifier.ingest_sop(temp_sop_path)
                    
                    # 3. Verify
                    gaps = verifier.verify_documents()
                    
                    if not gaps:
                        st.balloons()
                        st.success("No compliance gaps found! The SOP is fully compliant.")
                    else:
                        st.warning(f"Found {len(gaps)} Potential Compliance Gaps")
                        
                        for gap in gaps:
                            with st.expander(f"{gap['status']} : {gap['full_reference_text'][:80]}..."):
                                st.markdown(f"**Status:** `{gap['status']}`")
                                st.markdown(f"**Requirement:** {gap['full_reference_text']}")
                                st.markdown(f"**Reference Source:** {gap['reference_source']}")
                                st.markdown(f"**LLM Reason:** _{gap['justification']}_")
                                st.markdown("**SOP Evidence Found:**")
                                st.code(gap['sop_evidence'], language="text")

                except Exception as e:
                    st.error(f"Verification Failed: {e}")

