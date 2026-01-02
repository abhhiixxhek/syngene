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

st.set_page_config(page_title="Regulator-Grade Verification", layout="wide")

st.title("⚖️ Regulator-Grade Compliance Verification")
st.markdown("""
**Core Principle**: Reverse Check (Reference → SOP).  
**Logic**: presence of evidence, not similarity.
""")

# Sidebar
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Local Testing", "AWS S3 Production"])
bucket_name = "local" if mode == "Local Testing" else st.sidebar.text_input("Bucket", "my-compliance-bucket")

tab1, tab2 = st.tabs(["Phase 1: Ingest Truth (Ref)", "Phase 2 & 3: Verify (SOP)"])

# --- TAB 1: INGESTION ---
with tab1:
    st.header("Phase 1: Reference Preparation")
    st.info("Step 3.2: Atomic Normalization (breaking paragraphs into verifiable rules).")
    uploaded_refs = st.file_uploader("Upload Reference Document (Source of Truth)", type=['pdf', 'docx'], accept_multiple_files=True)
    
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
    st.header("Phase 2 & 3: SOP Verification")
    st.markdown("""
    **Verification Loop (Per Atomic Requirement):**
    1. **Search**: Find candidate SOP chunks.
    2. **Threshold**: ≥ 0.90 Auto-Pass. < 0.90 Send to LLM.
    3. **LLM Jury**: Decides PRESENT / PARTIAL / MISSING.
    """)
    uploaded_sop = st.file_uploader("Upload SOP (Document under evaluation)", type=['pdf', 'docx'])
    
    if st.button("Run Compliance Check"):
        if not uploaded_sop:
            st.error("Please upload an SOP.")
        else:
            status_container = st.empty()
            with st.spinner("Executing Phase 3: Reverse Verification Loop..."):
                temp_sop_path = os.path.join(tempfile.gettempdir(), uploaded_sop.name)
                with open(temp_sop_path, "wb") as f:
                    f.write(uploaded_sop.getbuffer())
                
                try:
                    verifier = SOPVerifier(bucket_name=bucket_name)
                    
                    # Phase 2: SOP Preparation
                    status_container.info("Phase 2: Ingesting & Vectorizing SOP...")
                    verifier.load_reference_index()
                    
                    # HARD RULE: No Atomic Normalization -> No Execution
                    if not verifier.reference_index:
                         st.error("❌ Critical Error: Reference Index is empty.")
                         st.stop()
                    
                    # Check for Failed Normalizations
                    failed_reqs = [r for r in verifier.reference_index if "FAIL" in r.get('requirement_id', '')]
                    if failed_reqs:
                         st.error(f"❌ Critical Error: Found {len(failed_reqs)} Failed Normalizations in Reference Index.")
                         st.markdown("Cannot proceed with Verification until Reference is cleanly normalized.")
                         with st.expander("View Failed Items"):
                              st.write(failed_reqs)
                         st.stop()
                         
                    verifier.ingest_sop(temp_sop_path)
                    
                    # Phase 3: Verification
                    status_container.info("Phase 3: Running Atomic Compliance Checks...")
                    gaps = verifier.verify_documents()
                    
                    # Phase 4: Reporting
                    status_container.success("Verification Complete.")
                    
                    if not gaps:
                        st.balloons()
                        st.success("✅ fully COMPLIANT! No missing requirements found.")
                    else:
                        st.error(f"❌ Found {len(gaps)} Compliance Gaps")
                        
                        for gap in gaps:
                            with st.expander(f"[{gap['status']}] Rule: {gap['reference_requirement'][:60]}...", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("### 📜 Reference (Truth)")
                                    st.markdown(f"**Requirement:** {gap['reference_requirement']}")
                                    st.caption(f"Source: {gap['reference_source']} | Severity: {gap['severity']}")
                                    if gap.get('reference_context'):
                                        with st.expander("Show Original Context Paragraph"):
                                            st.text(gap['reference_context'])
                                
                                with col2:
                                    st.markdown("### 🔍 SOP Evidence")
                                    st.markdown(f"**Status:** `{gap['status']}`")
                                    if gap['status'] == "MISSING":
                                        st.error("No Evidence Found")
                                        st.markdown(f"**Reason:** {gap['justification']}")
                                    else:
                                        st.warning("Partial Match")
                                        st.markdown(f"**Evidence:**\n{gap['sop_evidence']}")
                                        st.markdown(f"**Analysis:** {gap['justification']}")
                                
                except Exception as e:
                    st.error(f"Verification Failed: {e}")
