import streamlit as st
import os
import json
import tempfile
import sys
import pandas as pd
import time

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ingestion.indexer import ReferenceIndexer
from src.verification.verifier import SOPVerifier

# Page Config
st.set_page_config(
    page_title="Syngene Compliance Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #4B5563;}
    .metric-card {background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A;}
    .success-box {padding: 1rem; border-radius: 0.5rem; background-color: #D1FAE5; color: #065F46;}
    .error-box {padding: 1rem; border-radius: 0.5rem; background-color: #FEE2E2; color: #991B1B;}
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Syngene_International_Logo.svg/1200px-Syngene_International_Logo.svg.png", width=200) # Placeholder or actual logo URL if public
    st.title("Settings")
    
    st.markdown("### ☁️ Environment")
    mode = st.radio("Deployment Mode", ["Local Testing", "AWS S3 Production"])
    bucket_name = "local" if mode == "Local Testing" else st.text_input("S3 Bucket Name", "syngene-compliance-bucket")
    
    st.markdown("### 🤖 Model Configuration")
    st.info(f"**Parser:** AWS Bedrock (Mistral)\n**Normalizer:** AWS Bedrock (Mistral)\n**Verifier:** AWS Bedrock (Mistral)")
    
    st.markdown("---")
    st.caption(f"Project Root: `{project_root}`")

# Main Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="main-header">Compliance Verification Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Regulator-Grade SOP Analysis</div>', unsafe_allow_html=True)
with col2:
    st.metric(label="System Status", value="Online", delta="Ready")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📂 Phase 1: Ingest Truth (Ref)", "🔍 Phase 2 & 3: Verify (SOP)", "📊 Dashboard"])

# --- TAB 1: INGESTION ---
with tab1:
    st.markdown("### 1️⃣ Reference Preparation")
    st.markdown("The system builds a **Reference Index** by breaking down source documents (PDF/DOCX) into atomic, verifiable requirements.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info("**Step 1:** Upload Source of Truth (e.g. FDA Guidelines, Corporate Policy)")
        uploaded_refs = st.file_uploader("Upload Reference Document", type=['pdf', 'docx'], accept_multiple_files=True, key="ref_uploader")
        
        if st.button("🚀 Ingest & Normalize Reference", use_container_width=True):
            if not uploaded_refs:
                st.error("Please upload at least one reference document.")
            else:
                status_container = st.status("Processing Reference Documents...", expanded=True)
                try:
                    status_container.write("📂 Saving temp files...")
                    temp_paths = []
                    temp_dir = tempfile.mkdtemp()
                    for f in uploaded_refs:
                        path = os.path.join(temp_dir, f.name)
                        with open(path, "wb") as w:
                            w.write(f.getbuffer())
                        temp_paths.append(path)
                    
                    status_container.write("🧠 Instantiating Bedrock Indexer...")
                    indexer = ReferenceIndexer(bucket_name=bucket_name)
                    
                    status_container.write("🔨 Parsing & Normalizing (this calls Mistral LLM)...")
                    indexer.process_and_index(temp_paths)
                    
                    status_container.update(label="Ingestion Complete!", state="complete", expanded=False)
                    st.success(f"✅ Successfully created Reference Index with valid requirements.")
                    time.sleep(1)
                    st.rerun() # Refresh to show new index stats
                    
                except Exception as e:
                    status_container.update(label="Ingestion Failed", state="error")
                    st.error(f"Error: {e}")

    with col_b:
        st.markdown("**Current Reference Index**")
        try:
            # Quick check of index
            idx_path = "reference_index.json"
            if os.path.exists(idx_path):
                with open(idx_path, 'r') as f:
                    data = json.load(f)
                count = len(data)
                st.metric("Indexed Requirements", count)
                
                # Show sample
                if count > 0:
                    df = pd.DataFrame(data)
                    st.dataframe(df[['requirement_id', 'full_requirement_text', 'mandatory_level']], height=300)
            else:
                st.warning("No Reference Index found. Please ingest documents.")
        except Exception as e:
            st.error("Could not read index.")

# --- TAB 2: VERIFICATION ---
with tab2:
    st.markdown("### 2️⃣ SOP Verification")
    st.markdown("Upload a Standard Operating Procedure (SOP) to verify it against the ingested Reference Index.")
    
    uploaded_sop = st.file_uploader("Upload SOP for Audit", type=['pdf', 'docx'], key="sop_uploader")
    
    if st.button("🕵️‍♂️ Run Compliance Check", type="primary", use_container_width=True):
        if not uploaded_sop:
            st.error("Please upload an SOP file.")
        else:
            final_gaps = []
            
            # Progress Container
            with st.status("Executing Regulator-Grade Verification...", expanded=True) as status:
                temp_sop_path = os.path.join(tempfile.gettempdir(), uploaded_sop.name)
                with open(temp_sop_path, "wb") as f:
                    f.write(uploaded_sop.getbuffer())
                
                try:
                    status.write("⚙️ Initializing Verifier...")
                    verifier = SOPVerifier(bucket_name=bucket_name)
                    
                    status.write("📚 Loading Reference Index...")
                    verifier.load_reference_index()
                    
                    # HARD RULE CHECKS
                    if not verifier.reference_index:
                         st.error("❌ Critical Error: Reference Index is empty.")
                         st.stop()
                    
                    failed_reqs = [r for r in verifier.reference_index if "FAIL" in r.get('requirement_id', '')]
                    if failed_reqs:
                         st.error(f"❌ Critical Error: Found {len(failed_reqs)} Failed Normalizations.")
                         st.stop()
                    
                    status.write("📖 Ingesting SOP & Generating Embeddings...")
                    verifier.ingest_sop(temp_sop_path)
                    
                    status.write("🔍 Performing Reverse-Check Verification (Bedrock Mistral)...")
                    # Progress bar for verification
                    progress_bar = status.empty()
                    
                    # We can't easily hook into the loop inside verifier without changing it slightly
                    # or just letting it run. For UI responsiveness, we let it run.
                    gaps = verifier.verify_documents()
                    final_gaps = gaps
                    
                    status.update(label="Verification Complete!", state="complete", expanded=False)
                    
                    # --- RESULTS DISPLAY ---
                    st.divider()
                    st.subheader("📋 Audit Results")
                    
                    # Summary Metrics
                    total_reqs = len(verifier.reference_index)
                    failed_count = len(gaps)
                    passed_count = total_reqs - failed_count
                    score = (passed_count / total_reqs) * 100
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Compliance Score", f"{score:.1f}%", delta=f"{passed_count} Passed")
                    m2.metric("Total Requirements", total_reqs)
                    m3.metric("Passed", passed_count)
                    m4.metric("Gaps Found", failed_count, delta_color="inverse")
                    
                    if not gaps:
                        st.balloons()
                        st.markdown('<div class="success-box">✅ FULLY COMPLIANT! No gaps found against the reference standard.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-box">❌ Found {len(gaps)} Compliance Gaps. See details below.</div>', unsafe_allow_html=True)
                        
                        # Detailed Gaps
                        for gap in gaps:
                            color = "red" if gap['status'] == "MISSING" else "orange"
                            icon = "🛑" if gap['status'] == "MISSING" else "⚠️"
                            
                            with st.expander(f"{icon} [{gap['status']}] {gap['requirement_id']}: {gap['reference_requirement'][:80]}...", expanded=True):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("#### 📜 Reference Requirement")
                                    st.info(gap['reference_requirement'])
                                    st.caption(f"Source: {gap['reference_source']} | Severity: {gap['severity']}")
                                    if gap.get('reference_context'):
                                         st.text_area("Original Context", gap['reference_context'], height=100, disabled=True, key=f"context_{gap.get('requirement_id')}")
                                
                                with c2:
                                    st.markdown("#### 🔍 SOP Evidence")
                                    if gap['status'] == "MISSING":
                                        st.error(f"Status: {gap['status']}")
                                        st.markdown(f"**Reason:** {gap['justification']}")
                                    else:
                                        st.warning(f"Status: {gap['status']}")
                                        st.markdown(f"**Evidence:**\n> {gap['sop_evidence']}")
                                        st.markdown(f"**Analysis:** {gap['justification']}")

                        # Download Report
                        report_json = json.dumps(gaps, indent=2)
                        st.download_button("📥 Download Gap Report (JSON)", data=report_json, file_name="compliance_report.json", mime="application/json")
                
                except Exception as e:
                    status.update(label="Verification Failed", state="error")
                    st.error(f"Runtime Error: {e}")

# --- TAB 3: DASHBOARD (Future) ---
with tab3:
    st.info("Historical Compliance Trends and Multi-SOP Analytics will appear here.")
