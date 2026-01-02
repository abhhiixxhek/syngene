from reportlab.pdfgen import canvas
import os

def create_pdf(path, text_content):
    c = canvas.Canvas(path)
    y = 800
    for line in text_content.split('\n'):
        c.drawString(100, y, line)
        y -= 20
    c.save()

def generate_dummies():
    os.makedirs("data/references", exist_ok=True)
    os.makedirs("data/sops", exist_ok=True)
    
    # 1. Reference: Good Clinical Practice (Dummy)
    ref_text = """
    Good Clinical Practice Guidelines (Dummy)
    
    Section 1: Responsibilities
    1.1 The Investigator should be qualified by education, training, and experience.
    1.2 The Investigator should maintain a list of appropriately qualified persons to whom the investigator has delegated significant trial-related duties.
    
    Section 2: Safety Reporting
    2.1 All serious adverse events (SAEs) should be reported immediately to the sponsor.
    2.2 The immediate reports should be followed promptly by detailed, written reports.
    2.3 Adverse events and/or laboratory abnormalities identified in the protocol as critical to safety evaluations should be reported.
    
    Section 3: Protocol Compliance
    3.1 The investigator should conduct the trial in compliance with the protocol agreed to by the sponsor and, if required, by the regulatory authority(ies).
    3.2 The investigator should not implement any deviation from, or changes of the protocol without agreement by the sponsor and prior review and documented approval/favorable opinion from the IRB.
    """
    create_pdf("data/references/GCP_Dummy_Ref.pdf", ref_text)
    
    # 2. SOP: Deployment SOP (Has gaps!)
    sop_text = """
    Standard Operating Procedure: Clinical Trial Management (SOP-CTM-001)
    
    1. Purpose
    To define the responsibilities of the trial team.
    
    2. Responsibilities
    The Investigator must be qualified by education. (Note: Missing 'training and experience')
    The Investigator shall maintain a delegation log.
    
    3. Safety Reporting
    Any SAE must be reported within 24 hours.
    (Note: Missing the requirement for 'detailed written follow-up reports')
    
    4. Compliance
    The trial must be conducted according to the protocol.
    (Note: Missing the prohibition on deviations without approval)
    """
    create_pdf("data/sops/SOP_with_Gaps.pdf", sop_text)
    
    print("Dummy PDFs created in data/references and data/sops")

if __name__ == "__main__":
    generate_dummies()
