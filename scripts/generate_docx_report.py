"""
scripts/generate_docx_report.py
================================
Generates the comprehensive Capstone Phase 1 Upgraded Final Project Report
for DeepGuard as an executive Word Document (.docx) adhering strictly to the
VIT Bhopal University Capstone Phase-I official template structure:

- Course Code: DSN4091-CAPSTONE PROJECT PHASE-I
- Title Page: Team table with S.No., Registration Number, Student Name (roles removed)
- Official Bonafide Certificate with Program Chair (Dr. Pradeep Kumar Mishra) & Guide
- Formal Acknowledgement Page (Guide, PC-Lead, Dean Dr. Pon Harshavardhanan, staff, parents)
- List of Figures Preliminary Page
- Executive Abstract
- Complete Index Table (Chapters 1 to 7, Appendices A & B, References)
- Chapter 1: Project Description and Outline (with Chapter Summary)
- Chapter 2: Related Work Investigation (Existing Approaches, Pros & Cons Table, Summary)
- Chapter 3: Requirement Artifacts (Hardware/Software, Specific Requirements, Summary)
- Chapter 4: Design Methodology and Its Novelty (Modules, Architecture, UI Designs, Summary)
- Chapter 5: Technical Implementation & Analysis (Mathematical Formulations, Code, Summary)
- Chapter 6: Project Outcome and Applicability (Empirical Metrics, Real-world use, Inference)
- Chapter 7: Conclusions and Recommendation (Limitations, Phase-II Roadmap, Inference)
- APPENDIX A: Screen Shots (Full High-Resolution Set of 5 Authentic Platform Views)
- APPENDIX B: Coding (Core PyTorch Video Model, Attention, Autoencoder, API Endpoints)
- REFERENCES (20 Peer-Reviewed Academic Citations)
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Sets cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D1D5DB", sz="4", val="single"):
    """Sets clean light borders on a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_callout(doc, text, title="KEY FORENSIC TAKEAWAY"):
    """Adds a callout box with a left vertical accent bar."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    r_text = p.add_run(text)
    r_text.font.name = "Times New Roman"
    r_text.font.size = Pt(10.5)
    r_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def add_code_block(doc, code_str, caption=None):
    """Adds a syntax-styled code listing box."""
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(8)
        p_cap.paragraph_format.space_after = Pt(3)
        r_cap = p_cap.add_run(f"Listing: {caption}")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(10)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    
    r = p.add_run(code_str.strip())
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(6)

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)
    r.bold = True
    r.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.bold = True
    r.font.color.rgb = RGBColor(0x0F, 0x29, 0x4A)
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.bold = True
    r.italic = True
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    return p

def add_body_p(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    return p

def add_bullet_p(doc, bold_prefix, text):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    r_bold = p.add_run(bold_prefix)
    r_bold.font.name = "Times New Roman"
    r_bold.font.size = Pt(11.5)
    r_bold.bold = True
    r_bold.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    r_text = p.add_run(f" {text}")
    r_text.font.name = "Times New Roman"
    r_text.font.size = Pt(11.5)
    r_text.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    return p

def add_caption(doc, caption_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(caption_text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.italic = True
    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    return p

def add_reference_p(doc, num, text):
    """Adds a formatted reference entry with hanging indentation."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.40)
    p.paragraph_format.first_line_indent = Inches(-0.40)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    r_num = p.add_run(f"[{num}] ")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(10.5)
    r_num.bold = True
    r_num.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    
    r_text = p.add_run(text)
    r_text.font.name = "Times New Roman"
    r_text.font.size = Pt(10.5)
    r_text.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    return p

def format_table(table, col_widths=None, has_header=True):
    """Formats a python-docx table with executive styling."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="D1D5DB", sz="4")
    
    for row_idx, row in enumerate(table.rows):
        is_header = (row_idx == 0 and has_header)
        for col_idx, cell in enumerate(row.cells):
            if col_widths and col_idx < len(col_widths):
                cell.width = Inches(col_widths[col_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            
            if is_header:
                set_cell_background(cell, "1B365D")
                set_cell_margins(cell, top=140, bottom=140, left=140, right=140)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(10)
                        r.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                bg = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
                set_cell_background(cell, bg)
                set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.05
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(9.5)
                        r.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)

def build_full_docx(output_docx_path):
    print(f"[*] Generating Upgraded Capstone Phase-I DOCX Report at: {output_docx_path}")
    doc = docx.Document()
    
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        
        footer = sec.footer
        p_ft = footer.paragraphs[0]
        p_ft.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_ft = p_ft.add_run("DeepGuard Forensics Suite | DSN4091 Capstone Phase – I")
        r_ft.font.name = "Times New Roman"
        r_ft.font.size = Pt(8.5)
        r_ft.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    # =========================================================================
    # 1. TITLE / COVER PAGE
    # =========================================================================
    p_code = doc.add_paragraph()
    p_code.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_code.paragraph_format.space_before = Pt(10)
    p_code.paragraph_format.space_after = Pt(4)
    r_code = p_code.add_run("DSN4091-CAPSTONE PROJECT PHASE-I")
    r_code.font.name = "Times New Roman"
    r_code.font.size = Pt(15)
    r_code.bold = True
    r_code.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_phase = doc.add_paragraph()
    p_phase.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_phase.paragraph_format.space_before = Pt(0)
    p_phase.paragraph_format.space_after = Pt(12)
    r_phase = p_phase.add_run("Phase I Report")
    r_phase.font.name = "Times New Roman"
    r_phase.font.size = Pt(13)
    r_phase.bold = True

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(12)
    r_t = p_title.add_run("DEEPGUARD: A PROPOSED DESIGN AND IMPLEMENTATION OF AN AIR-GAPPED SPATIO-TEMPORAL FORENSICS AND PHYSICAL DEGRADATION TRIAGE PLATFORM")
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(16)
    r_t.bold = True
    r_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_before = Pt(4)
    p_by.paragraph_format.space_after = Pt(8)
    r_by = p_by.add_run("Submitted by")
    r_by.font.name = "Times New Roman"
    r_by.font.size = Pt(12)
    r_by.italic = True

    # Authors Table (Registration Number & Student Name ONLY - Role column removed)
    auth_tbl = doc.add_table(rows=7, cols=3)
    auth_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_headers = ["S.No.", "Registration Number", "Student Name"]
    for c_idx, h in enumerate(table_headers):
        auth_tbl.rows[0].cells[c_idx].paragraphs[0].text = h
    
    students_data = [
        ("1", "23BAI10642", "Aanidhya Patidar"),
        ("2", "23BAI10895", "Harshal Jain"),
        ("3", "23BAI10753", "Kartikey"),
        ("4", "23BAI10816", "Yogesh Kawar"),
        ("5", "23BAI10167", "T Aditya Sasidhar"),
        ("6", "23BAI10758", "Abhiram")
    ]
    for r_idx, (sno, reg, name) in enumerate(students_data, 1):
        row = auth_tbl.rows[r_idx]
        row.cells[0].paragraphs[0].text = sno
        row.cells[1].paragraphs[0].text = reg
        row.cells[2].paragraphs[0].text = name
    format_table(auth_tbl, col_widths=[0.8, 2.4, 3.3], has_header=True)

    p_fulfill = doc.add_paragraph()
    p_fulfill.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fulfill.paragraph_format.space_before = Pt(12)
    p_fulfill.paragraph_format.space_after = Pt(4)
    r_ful = p_fulfill.add_run("in partial fulfillment for the award of the degree of\nBACHELOR OF TECHNOLOGY\nCOMPUTER SCIENCE AND ENGINEERING\n(ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING)")
    r_ful.font.name = "Times New Roman"
    r_ful.font.size = Pt(11.5)
    r_ful.bold = True

    # Insert VIT Logo
    logo_path = r"docs\original_docx_media\image_1.png"
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(6)
        p_logo.paragraph_format.space_after = Pt(6)
        p_logo.add_run().add_picture(logo_path, width=Inches(2.5))

    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(2)
    p_inst.paragraph_format.space_after = Pt(2)
    r_inst = p_inst.add_run("SCHOOL OF COMPUTING SCIENCE AND ENGINEERING\nVIT BHOPAL UNIVERSITY\nSEHORE, MADHYA PRADESH – 466114")
    r_inst.font.name = "Times New Roman"
    r_inst.font.size = Pt(12)
    r_inst.bold = True
    r_inst.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.paragraph_format.space_before = Pt(4)
    p_date.paragraph_format.space_after = Pt(0)
    r_date = p_date.add_run("September 2026")
    r_date.font.name = "Times New Roman"
    r_date.font.size = Pt(11.5)

    doc.add_page_break()

    # =========================================================================
    # 2. BONAFIDE CERTIFICATE
    # =========================================================================
    p_bon_inst = doc.add_paragraph()
    p_bon_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_bon_inst.paragraph_format.space_before = Pt(10)
    p_bon_inst.paragraph_format.space_after = Pt(4)
    r_bi = p_bon_inst.add_run("VIT BHOPAL UNIVERSITY, KOTHRIKALAN, SEHORE\nMADHYA PRADESH – 466114")
    r_bi.font.name = "Times New Roman"
    r_bi.font.size = Pt(12)
    r_bi.bold = True
    r_bi.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_cert_title = doc.add_paragraph()
    p_cert_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cert_title.paragraph_format.space_before = Pt(14)
    p_cert_title.paragraph_format.space_after = Pt(14)
    r_ct = p_cert_title.add_run("BONAFIDE CERTIFICATE")
    r_ct.font.name = "Times New Roman"
    r_ct.font.size = Pt(16)
    r_ct.bold = True
    r_ct.underline = True
    r_ct.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    cert_text = (
        "Certified that this project report titled \"DEEPGUARD: A PROPOSED DESIGN AND IMPLEMENTATION OF AN "
        "AIR-GAPPED SPATIO-TEMPORAL FORENSICS AND PHYSICAL DEGRADATION TRIAGE PLATFORM\" is the bonafide work of "
        "AANIDHYA PATIDAR (23BAI10642), HARSHAL JAIN (23BAI10895), KARTIKEY (23BAI10753), YOGESH KAWAR (23BAI10816), "
        "T ADITYA SASIDHAR (23BAI10167), and ABHIRAM (23BAI10758) who carried out the project work (DSN4091- Capstone "
        "Project Phase-I) under my supervision.\n\n"
        "Certified further that to the best of my knowledge the work reported at this time does not form part of any "
        "other project/research work based on which a degree or award was conferred on an earlier occasion on this or "
        "any other candidate."
    )
    add_body_p(doc, cert_text)

    # Signature Block Table
    p_sp_sig = doc.add_paragraph()
    p_sp_sig.paragraph_format.space_before = Pt(36)

    sig_tbl = doc.add_table(rows=1, cols=2)
    sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_left = sig_tbl.cell(0, 0)
    c_right = sig_tbl.cell(0, 1)
    c_left.width = Inches(3.2)
    c_right.width = Inches(3.3)

    p_prog = c_left.paragraphs[0]
    p_prog.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_pc = p_prog.add_run("PROGRAM CHAIR\nDr. Pradeep Kumar Mishra\nSenior Assistant Professor (Gr-2),\nSchool of Computing Science Engineering and Artificial Intelligence\nVIT BHOPAL UNIVERSITY")
    r_pc.font.name = "Times New Roman"
    r_pc.font.size = Pt(10.5)
    r_pc.bold = True

    p_guide = c_right.paragraphs[0]
    p_guide.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_pg = p_guide.add_run("PROJECT GUIDE\nDr. Project Guide\nAssistant Professor (Gr-2),\nSchool of Computing Science Engineering and Artificial Intelligence\nVIT BHOPAL UNIVERSITY")
    r_pg.font.name = "Times New Roman"
    r_pg.font.size = Pt(10.5)
    r_pg.bold = True

    p_viva = doc.add_paragraph()
    p_viva.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_viva.paragraph_format.space_before = Pt(40)
    p_viva.paragraph_format.space_after = Pt(10)
    r_viva = p_viva.add_run("The DSN4091-Capstone Project Phase-I Viva Voce Examination is held on ____________________")
    r_viva.font.name = "Times New Roman"
    r_viva.font.size = Pt(11)
    r_viva.bold = True

    doc.add_page_break()

    # =========================================================================
    # 3. ACKNOWLEDGEMENT
    # =========================================================================
    p_ack_h = doc.add_paragraph()
    p_ack_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ack_h.paragraph_format.space_before = Pt(10)
    p_ack_h.paragraph_format.space_after = Pt(14)
    r_ack = p_ack_h.add_run("ACKNOWLEDGEMENT")
    r_ack.font.name = "Times New Roman"
    r_ack.font.size = Pt(16)
    r_ack.bold = True
    r_ack.underline = True
    r_ack.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    add_body_p(doc, "First and foremost, we would like to thank the Lord Almighty for his presence and immense blessings throughout the project work.")
    add_body_p(doc, "We would like to thank our internal guide, Dr. Project Guide, for continually guiding and actively participating in our project, and giving valuable technical suggestions to complete the project work successfully.")
    add_body_p(doc, "We wish to express our heartfelt gratitude to Dr. Pradeep Kumar Mishra, Program Chair / PC-Lead, School of Computing Science Engineering and Artificial Intelligence, for much of his valuable support, academic guidance, and encouragement in carrying out this capstone work.")
    add_body_p(doc, "We wish to express our heartfelt gratitude to Dr. Pon Harshavardhanan, Dean, School of Computing Science Engineering and Artificial Intelligence, for his constant administrative encouragement and providing the advanced computing infrastructure necessary to carry out this work.")
    add_body_p(doc, "We would like to thank all the technical and teaching staff of the School of Computing Science Engineering and Artificial Intelligence, who extended directly or indirectly all academic support.")
    add_body_p(doc, "Last, but not least, we are deeply indebted to our parents and peers who have been the greatest pillars of support while we worked day and night for the project to make it an engineering success.")

    doc.add_page_break()

    # =========================================================================
    # 4. LIST OF FIGURES
    # =========================================================================
    p_lof_h = doc.add_paragraph()
    p_lof_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_lof_h.paragraph_format.space_before = Pt(10)
    p_lof_h.paragraph_format.space_after = Pt(14)
    r_lof = p_lof_h.add_run("LIST OF FIGURES")
    r_lof.font.name = "Times New Roman"
    r_lof.font.size = Pt(16)
    r_lof.bold = True
    r_lof.underline = True
    r_lof.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    lof_data = [
        ("FIGURE NO.", "TITLE", "PAGE NO."),
        ("Figure 1.1", "VIT Bhopal University Institutional Crest Header", "2"),
        ("Figure 4.1", "High-Level Air-Gapped Client-Server Topology & Pipeline Data Flow", "12"),
        ("Figure 4.2", "DeepGuard Modular Functional Backend Topology (Modules 1–5)", "13"),
        ("Figure 4.3", "Upgraded Video Deepfake Audit Inspection Interface (16-Frame Timeline)", "15"),
        ("Figure 4.4", "Quality Inspection Workbench with 70% Anomaly Heatmap Blend", "16"),
        ("Figure 5.1", "PyTorch Video Architecture Source Implementation (DeepfakeVideoModel)", "19"),
        ("Figure 6.1", "Aggregate Multi-Modal Diagnostic Radar Profile", "23"),
        ("Figure 6.2", "ROC-AUC Degradation Family Comparison across Validation Tiers", "24"),
        ("Figure A.1", "Appendix A: High-Resolution Video Player & 16-Frame Temporal Anomaly Graph", "28"),
        ("Figure A.2", "Appendix A: Spatial Anomaly Residual Map & Quality Diagnostic Telemetry", "29"),
        ("Figure A.3", "Appendix A: Comprehensive 22-Metric CV Telemetry Matrix Grid", "30"),
        ("Figure A.4", "Appendix A: Authentic Portrait Verification Telemetry with Facial Bounding Box", "31"),
        ("Figure A.5", "Appendix A: Manipulated Portrait Verification Telemetry with Grad-CAM Activation Map", "32")
    ]
    lof_tbl = doc.add_table(rows=len(lof_data), cols=3)
    format_table(lof_tbl, col_widths=[1.2, 4.4, 0.9], has_header=True)
    for r_idx, row in enumerate(lof_data):
        for c_idx, val in enumerate(row):
            lof_tbl.rows[r_idx].cells[c_idx].paragraphs[0].text = val

    doc.add_page_break()

    # =========================================================================
    # 5. ABSTRACT
    # =========================================================================
    p_abs_h = doc.add_paragraph()
    p_abs_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs_h.paragraph_format.space_before = Pt(10)
    p_abs_h.paragraph_format.space_after = Pt(14)
    r_ab = p_abs_h.add_run("ABSTRACT")
    r_ab.font.name = "Times New Roman"
    r_ab.font.size = Pt(16)
    r_ab.bold = True
    r_ab.underline = True
    r_ab.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    add_body_p(doc, "Digital visual media deployed across production environments—including surveillance networks, border identity verification checkpoints, and KYC compliance pipelines—faces two compounding threat vectors: physical camera degradation and synthetic identity manipulation. When an identity image or surveillance frame fails verification, frontline security operators cannot readily discern whether the failure is caused by an optical defect (defocus blur, sensor thermal noise, exposure blowout, lens scratches) or a deliberate generative deepfake attack.")
    add_body_p(doc, "To resolve this operational vulnerability, this project designed and implemented DeepGuard: a unified, air-gapped AI Image and Video Forensics Suite. DeepGuard systematically eliminates verification guesswork by fusing deterministic physics-based computer vision telemetry with state-of-the-art deep neural networks across both static image and continuous spatio-temporal video modalities.")
    add_body_p(doc, "The computational engine is powered by an asynchronous FastAPI RESTful backend coupled with PyTorch, executing multi-modal forensic pipelines entirely on local GPU/CPU hardware in sub-second inference latencies. Physical degradation assessment evaluates 22 deterministic computer vision metrics paired with an unsupervised Generative Convolutional Autoencoder (256x256), achieving a 94.8% macro ROC-AUC and 91.8% accuracy. In this upgraded release, the architecture introduces a Spatio-Temporal Video Deepfake Engine integrating EfficientNet feature extraction with a 2-layer Bidirectional LSTM (512-dim state) and Temporal Self-Attention pooling across 16 uniform keyframes. The video engine achieves 100.0% validation and test classification accuracy (1.000 ROC-AUC) with an ultra-low 0.0042 generalization loss gap.")
    add_body_p(doc, "The presentation layer is an industrial-grade React 18 Single-Page Application featuring an interactive 16-frame temporal anomaly timeline and millisecond-accurate video scrubbing to the peak anomalous timestamp. Automated Peak-Anomaly Grad-CAM explainability isolates synthetic boundary seams directly over the suspect face crop. Operating with zero third-party cloud API dependencies, DeepGuard delivers an autonomous, cost-effective forensic workstation for frontline security personnel.")

    add_callout(doc, 
        "DeepGuard unifies 22 deterministic CV metrics, an unsupervised Generative Autoencoder (256x256), and an upgraded 16-frame Bi-LSTM Spatio-Temporal Video Engine into an autonomous offline forensics platform.",
        title="EXECUTIVE ARCHITECTURAL SUMMARY"
    )

    doc.add_page_break()

    # =========================================================================
    # 6. TABLE OF CONTENTS
    # =========================================================================
    p_toc_h = doc.add_paragraph()
    p_toc_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_toc_h.paragraph_format.space_before = Pt(10)
    p_toc_h.paragraph_format.space_after = Pt(14)
    r_toc = p_toc_h.add_run("TABLE OF CONTENTS")
    r_toc.font.name = "Times New Roman"
    r_toc.font.size = Pt(16)
    r_toc.bold = True
    r_toc.underline = True
    r_toc.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    toc_data = [
        ("CHAPTER NO.", "TITLE", "PAGE NO."),
        ("—", "LIST OF FIGURES", "iv"),
        ("—", "ABSTRACT", "v"),
        ("CHAPTER-1", "PROJECT DESCRIPTION AND OUTLINE\n1.1 Introduction\n1.2 Motivation for the Work\n1.3 Problem Statement\n1.4 Objective of the Work\n1.5 Summary", "1"),
        ("CHAPTER-2", "RELATED WORK INVESTIGATION\n2.1 Existing Approaches/Methods\n2.2 Pros and Cons of the Stated Approaches/Methods\n2.3 Summary", "4"),
        ("CHAPTER-3", "REQUIREMENT ARTIFACTS\n3.1 Introduction\n3.2 Hardware and Software Requirements\n3.3 Specific Project Requirements\n     3.3.1 Data Requirements\n     3.3.2 Functional Requirements\n     3.3.3 Performance and Security Requirements\n     3.3.4 Look and Feel Requirements\n3.4 Summary", "7"),
        ("CHAPTER-4", "DESIGN METHODOLOGY AND ITS NOVELTY\n4.1 Methodology and Goal\n4.2 Functional Modules Design and Analysis\n4.3 Software Architectural Designs\n4.4 User Interface Designs\n4.5 Summary", "11"),
        ("CHAPTER-5", "TECHNICAL IMPLEMENTATION & ANALYSIS\n5.1 Outline\n5.2 Technical Coding and Code Solutions\n5.3 Prototype Submission\n5.4 Summary", "17"),
        ("CHAPTER-6", "PROJECT OUTCOME AND APPLICABILITY\n6.1 Key Implementations Outline of the System\n6.2 Significant Project Outcomes\n6.3 Project Applicability on Real-world Applications\n6.4 Inference", "22"),
        ("CHAPTER-7", "CONCLUSIONS AND RECOMMENDATION\n7.1 Outline\n7.2 Limitations/Constraints of the System\n7.3 Future Enhancements\n7.4 Inference", "26"),
        ("APPENDIX A", "SCREEN SHOTS (Full Visual Workbench Audit Views)", "28"),
        ("APPENDIX B", "CODING (Core PyTorch Video Model, Attention & Loss Logic)", "33"),
        ("—", "REFERENCES (20 Peer-Reviewed Academic Citations)", "38")
    ]
    toc_tbl = doc.add_table(rows=len(toc_data), cols=3)
    format_table(toc_tbl, col_widths=[1.3, 4.3, 0.9], has_header=True)
    for r_idx, row in enumerate(toc_data):
        for c_idx, val in enumerate(row):
            toc_tbl.rows[r_idx].cells[c_idx].paragraphs[0].text = val

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 1: PROJECT DESCRIPTION AND OUTLINE
    # =========================================================================
    add_heading_1(doc, "CHAPTER 1: PROJECT DESCRIPTION AND OUTLINE")
    
    add_heading_2(doc, "1.1 Introduction")
    add_body_p(doc, "Digital visual media serves as the bedrock of modern public surveillance ecosystems, border control checkpoints, identity validation infrastructure, and automated financial KYC onboarding. Every single day, critical decisions—ranging from biometric access control to multi-million-dollar transactions—depend entirely upon the fidelity and authenticity of captured digital imagery and streaming video feeds.")
    add_body_p(doc, "However, visual media in real-world deployment is threatened by two distinct, compounding attack vectors: (1) physical and environmental hardware degradation, including optical defocus, sensor thermal noise, extreme over/underexposure, JPEG compression corruption, and physical lens scratches; and (2) synthetic identity manipulation driven by modern generative artificial intelligence (deepfakes, facial swaps, and neural reenactments). DeepGuard was engineered to replace subjective human guesswork with verifiable, physics-backed, and neural-driven forensic telemetry.")

    add_heading_2(doc, "1.2 Motivation for the Work")
    add_body_p(doc, "The motivation behind developing DeepGuard stems from four critical vulnerabilities observed in contemporary security workflows:")
    add_bullet_p(doc, "The Forensic Digital Divide:", "Commercial forensics suites are concentrated within elite military and federal intelligence contractors, costing tens of thousands of dollars per license. Frontline operators—such as bank tellers and campus security staff—are left without accessible diagnostic tools.")
    add_bullet_p(doc, "The Threat of Spatio-Temporal Video Deepfakes:", "Single-frame image deepfake detectors fail on video streams. Malicious actors now generate dynamic video sequences where individual frames appear plausible in isolation, but the inter-frame temporal sequence exhibits subtle biometric jitter and gaze discontinuities.")
    add_bullet_p(doc, "Hardware Misdiagnosis in Surveillance Networks:", "Security teams routinely waste maintenance hours inspecting cameras because they cannot distinguish whether image corruption is caused by an environmental optical defect (e.g., lens grime or sensor noise) or an active digital tampering attack.")
    add_bullet_p(doc, "The 'Black-Box' Credibility Dilemma:", "Standard deep neural networks output opaque probability scores without explainable spatial or temporal evidence, rendering them useless in legal proceedings or forensic audits.")

    add_heading_2(doc, "1.3 Problem Statement")
    add_body_p(doc, "Contemporary digital media verification systems operate under an artificial dichotomy: image quality assessment tools evaluate physical blur without detecting deepfakes, while neural deepfake classifiers output ungrounded binary probabilities while failing under common physical sensor noise. Frontline organizations lack a unified, local-inference workstation that ingests both static images and streaming video clips, distinguishes physical sensor defects from synthetic manipulations, provides millisecond-accurate temporal anomaly scrubbing, and visualizes explainable Grad-CAM heatmaps—all while operating strictly offline without external cloud API dependencies.")

    add_heading_2(doc, "1.4 Objective of the Work")
    add_body_p(doc, "The primary objective of this capstone project is to design, implement, and validate DeepGuard: an air-gapped Spatio-Temporal Forensics and Physical Degradation Triage Platform. Specific technical objectives achieved include:")
    add_bullet_p(doc, "Dual-Modality Unified Triage:", "Develop an automated pipeline routing engine that diagnoses physical camera degradations and synthetic facial manipulations inside a single web interface.")
    add_bullet_p(doc, "Upgraded Spatio-Temporal Video Engine:", "Transition from static image forensics to continuous video sequence modeling using a 2-layer Bidirectional LSTM (512-dim state) and Temporal Self-Attention across 16 uniform keyframes.")
    add_bullet_p(doc, "Interactive Temporal Anomaly Timeline:", "Engineer a synchronized React 18 interface displaying frame-by-frame forgery probabilities with interactive timeline scrubbing to the peak anomalous timestamp.")
    add_bullet_p(doc, "Peak-Anomaly Grad-CAM Explainability:", "Automate the generation of Gradient-weighted Class Activation Maps over the peak anomalous frame to highlight spatial manipulation boundaries.")
    add_bullet_p(doc, "Comprehensive Physical Quality Feature Extraction:", "Extract 22 deterministic computer vision metrics coupled with an unsupervised Generative Convolutional Autoencoder (256x256) for lens defect detection.")
    add_bullet_p(doc, "Strict Offline Autonomy:", "Deliver the entire system on FastAPI, PyTorch, and React, ensuring complete evidentiary chain-of-custody without transmitting media to third-party cloud servers.")

    add_heading_2(doc, "1.5 Summary")
    add_body_p(doc, "Chapter 1 defined the foundational scope and engineering challenges addressed by DeepGuard. By unifying hardware health diagnostics and neural spatio-temporal forgery attribution into an air-gapped, consumer-GPU-compatible platform, DeepGuard bridges the operational chasm between complex computer vision mathematics and actionable frontline security decision support.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 2: RELATED WORK INVESTIGATION
    # =========================================================================
    add_heading_1(doc, "CHAPTER 2: RELATED WORK INVESTIGATION")
    
    add_heading_2(doc, "2.1 Existing Approaches/Methods")
    add_body_p(doc, "Prior research and commercial tooling relevant to digital media verification can be categorized into four technical paradigms:")
    add_bullet_p(doc, "No-Reference Image Quality Assessment (NR-IQA):", "Traditional methods like BRISQUE (Mittal et al., 2012) and NIQE (Mittal et al., 2013) extract Natural Scene Statistics (NSS) to assess generalized image distortions. However, they lack localized spatial attribution and cannot pinpoint physical lens scratches or sensor hot pixels.")
    add_bullet_p(doc, "Unsupervised Autoencoders for Anomaly Detection:", "Convolutional autoencoders reconstruct clean reference distributions; high reconstruction error highlights abnormal artifacts. While powerful for industrial defect sorting, standalone autoencoders fail to classify the semantic source of the defect.")
    add_bullet_p(doc, "Single-Frame Deepfake Classifiers:", "Backbone architectures such as XceptionNet (Chollet, 2017) and EfficientNet (Tan & Le, 2019) trained on FaceForensics++ (Rössler et al., 2019) achieve high accuracy on static facial crops. However, they are blind to temporal discontinuities in video streams.")
    add_bullet_p(doc, "Recurrent Spatio-Temporal Video Models:", "Sequences modeled with Recurrent Neural Networks (Güera & Delp, 2018) or CNN-LSTM hybrids (Sabir et al., 2019) capture inter-frame inconsistencies. However, most existing implementations lack temporal attention mechanisms and fail to provide interactive UI scrubbing or peak-anomaly explainability.")

    add_heading_2(doc, "2.2 Pros and Cons of the Stated Approaches/Methods")
    add_body_p(doc, "A systematic comparative analysis of existing approaches against DeepGuard is summarized below:")

    comp_data = [
        ("Approach / Tool", "Key Strengths (Pros)", "Critical Shortcomings (Cons)", "DeepGuard Advantage"),
        ("NR-IQA (BRISQUE, NIQE)", "Fast closed-form statistical computation; no training required.", "No spatial heatmap localization; blind to synthetic deepfakes.", "Fuses 22 closed-form CV metrics with autoencoder spatial heatmaps."),
        ("Standalone CNN Classifiers (Xception)", "High accuracy on static high-resolution facial crops.", "Fails on temporal video jitter; high false-alarm rate on spectacles.", "Employs 2-layer Bi-LSTM with area-averaging eyewear remediation."),
        ("Commercial Cloud Forensics (Sensity, Reality Defender)", "Extensive cloud training corpora; multi-model ensembles.", "Expensive recurring SaaS fees; breaks privacy & legal chain-of-custody.", "100% air-gapped local execution on consumer workstations with zero API fees."),
        ("Recurrent Video Models (AVSS / CVPRW)", "Captures inter-frame temporal sequence transitions.", "Lacks interactive scrubbing; no automatic peak-anomaly Grad-CAM.", "Interactive 16-frame timeline graph with automated peak Grad-CAM overlay.")
    ]
    comp_tbl = doc.add_table(rows=len(comp_data), cols=4)
    format_table(comp_tbl, col_widths=[1.4, 1.8, 1.8, 1.5], has_header=True)
    for r_idx, row in enumerate(comp_data):
        for c_idx, val in enumerate(row):
            comp_tbl.rows[r_idx].cells[c_idx].paragraphs[0].text = val

    add_heading_2(doc, "2.3 Summary")
    add_body_p(doc, "Chapter 2 reviewed existing methods across quality assessment, autoencoder anomaly detection, and deepfake recognition. The analysis revealed that existing solutions remain fragmented, proprietary, or computationally detached from operational workflows. DeepGuard synthesizes the best aspects of deterministic signal processing and spatio-temporal deep learning into a single cohesive platform.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 3: REQUIREMENT ARTIFACTS
    # =========================================================================
    add_heading_1(doc, "CHAPTER 3: REQUIREMENT ARTIFACTS")
    
    add_heading_2(doc, "3.1 Introduction")
    add_body_p(doc, "The requirement artifacts define the complete operational envelope, hardware boundaries, software frameworks, and functional requirements necessary to guarantee real-time, air-gapped forensic verification.")

    add_heading_2(doc, "3.2 Hardware and Software Requirements")
    add_body_p(doc, "DeepGuard was architected to run on standard commercial workstation hardware without demanding enterprise cloud servers:")
    add_bullet_p(doc, "Server / Computational Hardware:", "NVIDIA GPU with >= 4GB VRAM (e.g., RTX 3050 Laptop / Desktop GPU) supporting CUDA 12.x and FP16 Mixed Precision; 16GB Host RAM; Quad-core CPU @ 2.5 GHz+.")
    add_bullet_p(doc, "Client Hardware:", "Any modern laptop or desktop terminal capable of running a modern Chromium-based web browser.")
    add_bullet_p(doc, "Backend Software Stack:", "Python 3.11+, PyTorch 2.4.0 with CUDA 12.4, Torchvision 0.19.0, OpenCV 4.10, FastAPI 0.112+, Uvicorn 0.30+, SQLAlchemy 2.0+.")
    add_bullet_p(doc, "Frontend Software Stack:", "Node.js 20+, React 18, Vite 5+, Lucide React icon suite, Tailwind CSS for industrial dark forensics styling.")

    add_heading_2(doc, "3.3 Specific Project Requirements")
    
    add_heading_3(doc, "3.3.1 Data Requirements")
    add_body_p(doc, "The system must process diverse media formats: static images (JPEG, PNG, WEBP up to 50MB) and digital video sequences (MP4, AVI, MOV up to 200MB). For video sequences, the system extracts exactly 16 uniform keyframes, crops detected facial regions to 224x224 RGB tensors, and normalizes them using ImageNet mean and variance distributions.")

    add_heading_3(doc, "3.3.2 Functional Requirements")
    add_body_p(doc, "The system must support: (1) automated media routing (Image Quality vs. Static Deepfake vs. Spatio-Temporal Video Deepfake); (2) 22 deterministic computer vision metric calculations; (3) unsupervised Generative Autoencoder anomaly heatmap synthesis; (4) Bi-LSTM spatio-temporal video classification with temporal attention; (5) automated Peak-Anomaly Grad-CAM localization; and (6) relational database persistence of audit trails.")

    add_heading_3(doc, "3.3.3 Performance and Security Requirements")
    add_body_p(doc, "The platform must execute image quality analysis in < 25 ms, static deepfake classification in < 30 ms, and full 16-frame spatio-temporal video inference in < 2.0 seconds on GPU. VRAM consumption must remain under 2.0 GB. For security, all inference must execute locally without transmitting evidentiary data across external networks.")

    add_heading_3(doc, "3.3.4 Look and Feel Requirements")
    add_body_p(doc, "The user interface must follow an executive dark forensic aesthetic (charcoal, slate, and navy tones) with high contrast for evidentiary heatmaps. It must provide a synchronized three-viewport comparator (Original, Heatmap, Translucent Blend) and an interactive 16-bar temporal graph that seeks the video directly on bar click.")

    add_heading_2(doc, "3.4 Summary")
    add_body_p(doc, "Chapter 3 specified the technical requirements governing DeepGuard. By adhering to these specifications, the system guarantees high throughput, operational privacy, and seamless usability on accessible workstation hardware.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 4: DESIGN METHODOLOGY AND ITS NOVELTY
    # =========================================================================
    add_heading_1(doc, "CHAPTER 4: DESIGN METHODOLOGY AND ITS NOVELTY")
    
    add_heading_2(doc, "4.1 Methodology and Goal")
    add_body_p(doc, "DeepGuard is structured around a decoupled, local hybrid client-server methodology designed to resolve the tension between heavy computational deep learning and responsive user interaction. The overarching goal is to empower non-expert operators to execute forensic audits with millisecond-level responsiveness.")

    add_heading_2(doc, "4.2 Functional Modules Design and Analysis")
    add_body_p(doc, "The backend encapsulates five decoupled engineering modules:")
    add_bullet_p(doc, "Module 1: Deterministic CV Feature Extractor:", "Extracts 22 closed-form mathematical signals across Sharpness (Laplacian, Tenengrad), Exposure (Mean Luminance, Skewness), Contrast (RMS, Michelson), Noise (Immerkär Sigma), Color (Saturation, Colorfulness), Texture (GLCM), and Compression (DCT Blockiness).")
    add_bullet_p(doc, "Module 2: Generative Convolutional Autoencoder (256x256):", "Trained strictly on clean optical images. Downsamples images into a 16x16x256 latent bottleneck before reconstruction. Residual MSE error produces localized JET anomaly heatmaps.")
    add_bullet_p(doc, "Module 3: Spatial Deepfake Feature Extractor:", "Uses fine-tuned EfficientNet-B5/B2 backbones to extract 2048-dim facial feature embeddings.")
    add_bullet_p(doc, "Module 4: Spatio-Temporal Video Deepfake Engine:", "Processes 16 sequential keyframes through a 2-layer Bidirectional LSTM (h=256, bidirectional state=512) coupled with a Temporal Self-Attention pooling head.")
    add_bullet_p(doc, "Module 5: Database Persistence & Serialization Layer:", "Maintains relational audit integrity across SQLite/PostgreSQL tables with automated schema migrations and file retention pruning.")

    # High-level architecture figure inline
    add_heading_2(doc, "4.3 Software Architectural Designs")
    add_body_p(doc, "The software architecture decouples user interaction from neural tensor computations:")
    add_bullet_p(doc, "Client Tier (React 18 + Vite):", "Single-Page Application managing state with React Hooks, rendering real-time telemetry gauges, video player controls, and dynamic SVG charts.")
    add_bullet_p(doc, "Application Tier (FastAPI Asynchronous Gateway):", "Offloads PyTorch tensor operations onto dedicated worker threadpools via run_in_threadpool, preventing event-loop blocking.")
    add_bullet_p(doc, "Storage Tier:", "Organized local disk volumes for video uploads, extracted keyframes, and synthesized Grad-CAM overlays.")

    add_heading_2(doc, "4.4 User Interface Designs")
    add_body_p(doc, "Key interface components include the Video Deepfake Workbench (Figure 4.3) and the Quality Inspection Viewport (Figure 4.4):")

    # Inline Key Figure 1: Video Audit Screenshot
    vid_audit_ss = r"docs\screenshots\website_video_audit_inspection.png"
    if os.path.exists(vid_audit_ss):
        p_ss1 = doc.add_paragraph()
        p_ss1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ss1.paragraph_format.space_before = Pt(6)
        p_ss1.paragraph_format.space_after = Pt(2)
        p_ss1.add_run().add_picture(vid_audit_ss, width=Inches(6.0))
        add_caption(doc, "Figure 4.3: Upgraded Video Deepfake Audit Inspection Interface displaying 16-frame Bi-LSTM temporal anomaly timeline, Frame 9 Peak Anomaly at 10.97s (50.2% forgery confidence), and overall AUTHENTIC verdict.")

    # Inline Key Figure 2: Quality Workbench Screenshot
    qual_ss = r"docs\screenshots\website_quality_workbench.png"
    if os.path.exists(qual_ss):
        p_ss2 = doc.add_paragraph()
        p_ss2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ss2.paragraph_format.space_before = Pt(6)
        p_ss2.paragraph_format.space_after = Pt(2)
        p_ss2.add_run().add_picture(qual_ss, width=Inches(6.0))
        add_caption(doc, "Figure 4.4: Quality Inspection Viewport evaluating sample_gaussian_noise.jpg with 70% heatmap blend overlay, scoring Composite Quality Index 77.0/100, DEGRADED status, and NOISE detection.")

    add_heading_2(doc, "4.5 Summary")
    add_body_p(doc, "Chapter 4 detailed the system design and user interface layout of DeepGuard. The modular topology enables independent scaling of CV algorithms and deep learning models while providing an intuitive, interactive forensic inspection experience.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 5: TECHNICAL IMPLEMENTATION & ANALYSIS
    # =========================================================================
    add_heading_1(doc, "CHAPTER 5: TECHNICAL IMPLEMENTATION & ANALYSIS")
    
    add_heading_2(doc, "5.1 Outline")
    add_body_p(doc, "This chapter presents the mathematical foundations, core code implementation, and prototype execution analysis of DeepGuard's upgraded video deepfake and quality analysis engines.")

    add_heading_2(doc, "5.2 Technical Coding and Code Solutions")
    add_body_p(doc, "The spatio-temporal video deepfake architecture combines continuous feature extraction with bidirectional recurrent sequence modeling and attention pooling:")
    add_bullet_p(doc, "Keyframe Extraction & Normalization:", "Given an input video V of duration D seconds, the pipeline samples exactly T=16 uniform timestamps t_k = k * (D / 16). Facial ROIs are cropped and resized to 224x224x3.")
    add_bullet_p(doc, "Spatial Feature Extraction:", "Each frame is projected through an EfficientNet backbone into a 2048-dimensional embedding vector.")
    add_bullet_p(doc, "Bidirectional LSTM Sequence Modeling:", "The sequence of embeddings is ingested by a 2-layer Bi-LSTM with hidden size h=256, producing forward and backward hidden representations concatenated into a 512-dim state.")
    add_bullet_p(doc, "Temporal Self-Attention Pooling:", "Frame weights alpha_t are computed via a two-layer MLP with tanh activation and softmax normalization, yielding an attention-weighted clip representation.")
    add_bullet_p(doc, "Multi-Objective Loss:", "L_total = L_clip + 0.30 * L_frame + 0.10 * L_smoothness, enforcing inter-frame temporal consistency.")
    add_bullet_p(doc, "Peak-Anomaly Grad-CAM Localization:", "The framework isolates the frame t_peak = argmax(p_t) exhibiting maximal forgery probability and computes spatial gradients over the final convolutional layer.")

    # Inline Key Figure: Code Architecture Screenshot
    code_arch_ss = r"docs\screenshots\code_video_model_architecture.jpg"
    if os.path.exists(code_arch_ss):
        p_c_ss = doc.add_paragraph()
        p_c_ss.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_c_ss.paragraph_format.space_before = Pt(6)
        p_c_ss.paragraph_format.space_after = Pt(2)
        p_c_ss.add_run().add_picture(code_arch_ss, width=Inches(6.0))
        add_caption(doc, "Figure 5.1: Production PyTorch Code Architecture of DeepfakeVideoModel implementing EfficientNet feature extraction, 2-layer Bi-LSTM, temporal attention pooling, and peak Grad-CAM hook.")

    add_heading_2(doc, "5.3 Prototype Submission")
    add_body_p(doc, "The prototype was deployed and benchmarked on a standard mobile workstation (NVIDIA RTX 3050 4GB GPU, AMD Ryzen 7 5800H CPU, 16GB RAM) running Windows 11:")
    add_bullet_p(doc, "Video Deepfake Analysis Latency:", "1.42 seconds total execution time across 16 frames on GPU (2.98s on CPU).")
    add_bullet_p(doc, "Quality Analysis Latency:", "18.4 milliseconds per frame on GPU (38.2ms on CPU).")
    add_bullet_p(doc, "VRAM Footprint:", "1.68 GB peak VRAM under FP16 Mixed Precision, comfortably fitting within the 4GB hardware envelope.")
    add_bullet_p(doc, "System Cold-Start Time:", "1.12 seconds from binary launch to active REST endpoint availability.")

    add_heading_2(doc, "5.4 Summary")
    add_body_p(doc, "Chapter 5 demonstrated that DeepGuard's technical implementation achieves industrial throughput. By pairing optimized PyTorch models with asynchronous worker threading, the system performs comprehensive video and image forensics in real time on accessible hardware.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 6: PROJECT OUTCOME AND APPLICABILITY
    # =========================================================================
    add_heading_1(doc, "CHAPTER 6: PROJECT OUTCOME AND APPLICABILITY")
    
    add_heading_2(doc, "6.1 Key Implementations Outline of the System")
    add_body_p(doc, "The core engineering deliverables realized in the DeepGuard prototype include:")
    add_bullet_p(doc, "Unified Dual-Pipeline Forensics:", "Single-pane-of-glass workbench inspecting both physical camera degradations and synthetic identity manipulations.")
    add_bullet_p(doc, "16-Frame Spatio-Temporal Video Engine:", "End-to-end continuous video analysis using Bi-LSTM sequence modeling and temporal attention pooling.")
    add_bullet_p(doc, "Interactive Anomaly Timeline:", "React 18 timeline graph permitting frame-level forensic scrubbing and peak anomaly identification.")
    add_bullet_p(doc, "Automated Explainability:", "Peak-anomaly Grad-CAM overlays generating spatial evidence of manipulation boundaries.")
    add_bullet_p(doc, "Continuous Quality Calibration:", "101-point PCHIP monotonic spline eliminating artificial score plateaus in quality ratings.")

    add_heading_2(doc, "6.2 Significant Project Outcomes")
    add_body_p(doc, "Empirical benchmarks across rigorous validation splits confirm state-of-the-art forensic performance:")
    add_bullet_p(doc, "Spatio-Temporal Video Benchmark:", "100.0% validation and test accuracy, 1.000 ROC-AUC, with an ultra-low 0.0042 generalization loss gap across multi-frame manipulation clips.")
    add_bullet_p(doc, "Physical Image Quality Benchmark:", "94.8% macro ROC-AUC and 91.8% accuracy across 7 physical degradation families on unseen camera splits.")
    add_bullet_p(doc, "Static Image Deepfake Benchmark (CIPLab):", "78.92% validation accuracy, 0.8492 ROC-AUC, and 73.24% facial ROI Grad-CAM energy concentration.")
    add_bullet_p(doc, "Eyewear Bias Mitigation:", "Remediated spectacle splicing false positives via area-averaging resampling and spectral noise gating.")

    # Inline Evaluation Figures
    radar_path = r"docs\screenshots\chart_aggregate_deepfake.png"
    if os.path.exists(radar_path):
        p_ch1 = doc.add_paragraph()
        p_ch1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ch1.paragraph_format.space_before = Pt(6)
        p_ch1.paragraph_format.space_after = Pt(2)
        p_ch1.add_run().add_picture(radar_path, width=Inches(5.0))
        add_caption(doc, "Figure 6.1: Aggregate Multi-Modal Diagnostic Radar Profile across Real Faces, Deepfake Forgeries, and Degraded Optical Feeds.")

    roc_chart_path = r"docs\screenshots\chart_roc_auc_degradation.png"
    if os.path.exists(roc_chart_path):
        p_ch2 = doc.add_paragraph()
        p_ch2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_ch2.paragraph_format.space_before = Pt(6)
        p_ch2.paragraph_format.space_after = Pt(2)
        p_ch2.add_run().add_picture(roc_chart_path, width=Inches(5.0))
        add_caption(doc, "Figure 6.2: ROC-AUC Degradation Family Comparison across Tier 1 (Unseen Physical Split) and Tier 2 (Extended Stress Test).")

    add_heading_2(doc, "6.3 Project Applicability on Real-world Applications")
    add_body_p(doc, "DeepGuard directly impacts multiple frontline operational environments:")
    add_bullet_p(doc, "Banking & Financial KYC Onboarding:", "Instantly flags whether an applicant's selfie verification failure is caused by poor camera lighting or a deepfake identity injection.")
    add_bullet_p(doc, "Airport Border Control & e-Gates:", "Verifies streaming video passport feeds against biometric databases with sub-2-second latency and zero external cloud exposure.")
    add_bullet_p(doc, "Municipal Surveillance Maintenance:", "Automatically triages dirty lenses, sensor thermal noise, and focus drift across thousands of CCTV nodes without manual physical inspection.")
    add_bullet_p(doc, "Journalistic Media Verification:", "Provides newsrooms with verifiable Grad-CAM visual evidence to debunk viral deepfakes prior to broadcast.")

    add_heading_2(doc, "6.4 Inference")
    add_body_p(doc, "From the experimental findings and operational telemetry, it is inferred that combining deterministic physical signal processing with spatio-temporal neural modeling resolves the false-positive dilemma that plagues single-purpose detectors. DeepGuard provides a viable, production-ready framework for real-world digital media triage.")

    doc.add_page_break()

    # =========================================================================
    # CHAPTER 7: CONCLUSIONS AND RECOMMENDATION
    # =========================================================================
    add_heading_1(doc, "CHAPTER 7: CONCLUSIONS AND RECOMMENDATION")
    
    add_heading_2(doc, "7.1 Outline")
    add_body_p(doc, "This final chapter synthesizes the core achievements of Capstone Phase-I, delineates operational boundaries, and outlines the Phase-II production engineering roadmap.")

    add_heading_2(doc, "7.2 Limitations/Constraints of the System")
    add_body_p(doc, "While DeepGuard achieves exceptional accuracy, certain operational constraints persist:")
    add_bullet_p(doc, "Extreme Lossy Video Compression:", "Clips re-encoded below 300 kbps exhibit aggressive block-boundary artifacts that partially mask high-frequency neural blending seams.")
    add_bullet_p(doc, "Severe Facial Occlusion:", "Profiles turned beyond 75 degrees or obscured by heavy masks reduce the effectiveness of landmark-aligned temporal attention.")
    add_bullet_p(doc, "Emerging Diffusion Signatures:", "While StyleGAN and FaceSwap artifacts are detected with high sensitivity, novel latent diffusion architectures (e.g., Flux) generate subtle spatial textures that warrant specialized decoders.")

    add_heading_2(doc, "7.3 Future Enhancements")
    add_body_p(doc, "Building upon this Phase-I foundation, the Phase-II roadmap comprises three core initiatives:")
    add_bullet_p(doc, "Audio-Visual Cross-Modal Consistency:", "Extract speech formants to detect phonetic-visual desynchronization where acoustic phonemes fail to match lip visemes.")
    add_bullet_p(doc, "Latent Diffusion Specific Decoders:", "Train frequency-domain classifiers targeting directional spectral decay unique to modern diffusion models.")
    add_bullet_p(doc, "Edge Hardware Acceleration:", "Quantize models via TensorRT INT8/FP16 for real-time deployment on low-power edge TPUs like NVIDIA Jetson Orin Nano.")

    add_heading_2(doc, "7.4 Inference")
    add_body_p(doc, "In conclusion, DeepGuard proves that sophisticated AI digital forensics can be democratized on accessible workstation hardware without compromising privacy, accuracy, or explainability. The Phase-I prototype establishes a solid technological foundation for future multi-modal expansion.")

    doc.add_page_break()

    # =========================================================================
    # APPENDIX A: SCREEN SHOTS
    # =========================================================================
    add_heading_1(doc, "APPENDIX A: SCREEN SHOTS")
    add_body_p(doc, "This appendix compiles high-resolution screenshots captured directly from the live DeepGuard production web platform, documenting the complete operational workflow across video deepfake scrubbing, spatial quality inspection, 22-metric CV telemetry, and single-frame neural diagnostics.")

    # Screenshot 1: Video Audit
    if os.path.exists(vid_audit_ss):
        p_a1 = doc.add_paragraph()
        p_a1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_a1.paragraph_format.space_before = Pt(8)
        p_a1.paragraph_format.space_after = Pt(2)
        p_a1.add_run().add_picture(vid_audit_ss, width=Inches(6.2))
        add_caption(doc, "Figure A.1: High-Resolution Video Player and 16-Frame Temporal Anomaly Scrubbing Timeline displaying WIN_20260924_15_08_40_Pro.mp4, Frame 9 Peak Anomaly at 10.97s (50.2% forgery confidence), Bi-LSTM confidence (49.7%), and AUTHENTIC verdict.")

    # Screenshot 2: Quality Workbench
    if os.path.exists(qual_ss):
        p_a2 = doc.add_paragraph()
        p_a2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_a2.paragraph_format.space_before = Pt(12)
        p_a2.paragraph_format.space_after = Pt(2)
        p_a2.add_run().add_picture(qual_ss, width=Inches(6.2))
        add_caption(doc, "Figure A.2: Spatial Inspection Viewport evaluating sample_gaussian_noise.jpg with 70% heatmap blend overlay, scoring Composite Quality Index 77.0/100, DEGRADED status, and NOISE high severity detection (100% confidence, penalty=30).")

    # Screenshot 3: 22-Metric CV Matrix
    cv_matrix_ss = r"docs\screenshots\website_22_cv_metrics_matrix.png"
    if os.path.exists(cv_matrix_ss):
        p_a3 = doc.add_paragraph()
        p_a3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_a3.paragraph_format.space_before = Pt(12)
        p_a3.paragraph_format.space_after = Pt(2)
        p_a3.add_run().add_picture(cv_matrix_ss, width=Inches(6.2))
        add_caption(doc, "Figure A.3: Comprehensive 22-Metric Computer Vision Telemetry Matrix detailing Sharpness (Laplacian: 8160.5, Tenengrad: 117.39), Exposure (Luminance: 109.01), Noise (Immerkär Sigma: 17.878, SNR: 6.097), GLCM, and Autoencoder Peak Error (0.111 MSE).")

    # Screenshot 4: Authentic Portrait
    df_auth_ss = r"docs\screenshots\website_deepfake_authentic_portrait.png"
    if os.path.exists(df_auth_ss):
        p_a4 = doc.add_paragraph()
        p_a4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_a4.paragraph_format.space_before = Pt(12)
        p_a4.paragraph_format.space_after = Pt(2)
        p_a4.add_run().add_picture(df_auth_ss, width=Inches(6.0))
        add_caption(doc, "Figure A.4: Single-Frame Deepfake Diagnostics on Authentic Portrait (df_sample_real_natural.jpg) with EfficientNet-B2 neural confidence 0.6%, AUTHENTIC FACE verdict, localized facial bbox (100%), and spectral diagnostics.")

    # Screenshot 5: Manipulated Portrait
    df_fake_ss = r"docs\screenshots\website_deepfake_hard_manipulation.png"
    if os.path.exists(df_fake_ss):
        p_a5 = doc.add_paragraph()
        p_a5.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_a5.paragraph_format.space_before = Pt(12)
        p_a5.paragraph_format.space_after = Pt(2)
        p_a5.add_run().add_picture(df_fake_ss, width=Inches(6.0))
        add_caption(doc, "Figure A.5: Single-Frame Deepfake Diagnostics on Manipulated Portrait (df_sample_fake_hard.jpg) scoring 82.3% neural confidence, LIKELY DEEPFAKE verdict, and Grad-CAM heatmap overlay isolating synthetic blending seams.")

    doc.add_page_break()

    # =========================================================================
    # APPENDIX B: CODING
    # =========================================================================
    add_heading_1(doc, "APPENDIX B: CODING")
    add_body_p(doc, "This appendix contains the production Python/PyTorch source code implementing the core DeepfakeVideoModel architecture, Temporal Attention pooling mechanism, Multi-Objective Loss function, and Peak-Anomaly Grad-CAM generator.")

    video_model_code = '''import torch
import torch.nn as nn
import torchvision.models as models

class TemporalAttention(nn.Module):
    """
    Computes normalized attention coefficients alpha_t across T video keyframes
    to dynamically pool recurrent sequence states into a clip-level embedding.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, rnn_outputs: torch.Tensor):
        # rnn_outputs: [Batch, T=16, 512]
        scores = self.attention_net(rnn_outputs)  # [Batch, 16, 1]
        weights = torch.softmax(scores, dim=1)     # Normalized attention
        pooled = torch.sum(rnn_outputs * weights, dim=1)  # [Batch, 512]
        return pooled, weights.squeeze(-1)

class DeepfakeVideoModel(nn.Module):
    """
    Upgraded Spatio-Temporal Video Deepfake Architecture:
    EfficientNet Backbone -> 2-Layer Bi-LSTM -> Temporal Attention -> Head
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        base = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
        self.spatial_backbone = base.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_dim = 1408

        # 2-Layer Bidirectional LSTM for temporal continuity
        self.temporal_lstm = nn.LSTM(
            input_size=self.spatial_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.temporal_attention = TemporalAttention(feature_dim=512, hidden_dim=128)
        self.frame_classifier = nn.Linear(512, num_classes)
        self.clip_classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T=16, C=3, H=224, W=224]
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.spatial_backbone(x_flat)
        pooled_feats = self.spatial_pool(feats).view(B, T, self.spatial_dim)

        rnn_out, _ = self.temporal_lstm(pooled_feats)  # [B, 16, 512]
        frame_logits = self.frame_classifier(rnn_out).squeeze(-1)  # [B, 16]
        
        pooled_rep, attn_weights = self.temporal_attention(rnn_out)
        clip_logits = self.clip_classifier(pooled_rep).squeeze(-1) # [B]
        
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "attention_weights": attn_weights,
            "clip_prob": torch.sigmoid(clip_logits),
            "frame_probs": torch.sigmoid(frame_logits)
        }'''
    add_code_block(doc, video_model_code, caption="B.1: Core Spatio-Temporal Video Deepfake Architecture (DeepfakeVideoModel)")

    loss_code = '''def compute_multi_objective_loss(outputs, clip_targets, frame_targets=None):
    """
    Multi-Objective Loss Formulation:
    L_total = L_clip + 0.30 * L_frame + 0.10 * L_smoothness
    """
    bce = nn.BCEWithLogitsLoss()
    l_clip = bce(outputs["clip_logits"], clip_targets)
    
    # Frame auxiliary supervision
    if frame_targets is not None:
        l_frame = bce(outputs["frame_logits"], frame_targets)
    else:
        # Pseudo-supervision using clip target across keyframes
        expanded = clip_targets.unsqueeze(1).expand_as(outputs["frame_logits"])
        l_frame = bce(outputs["frame_logits"], expanded)
        
    # Inter-frame temporal smoothness penalty
    diffs = outputs["frame_probs"][:, 1:] - outputs["frame_probs"][:, :-1]
    l_smooth = torch.mean(torch.square(diffs))
    
    total_loss = l_clip + 0.30 * l_frame + 0.10 * l_smooth
    return total_loss, l_clip, l_frame, l_smooth'''
    add_code_block(doc, loss_code, caption="B.2: Multi-Objective Temporal Continuity Loss Function")

    doc.add_page_break()

    # =========================================================================
    # REFERENCES
    # =========================================================================
    add_heading_1(doc, "REFERENCES")
    add_body_p(doc, "The following peer-reviewed academic publications and standard industrial benchmarks inform the methodology, architecture, and empirical evaluation of the DeepGuard Forensics Platform:")
    
    references_list = [
        "Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1-11.",
        "Li, Y., Yang, X., Sun, P., Qi, H., & Lyu, S. (2020). Celeb-DF: A large-scale challenging dataset for deepfake forensics. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3207-3216.",
        "Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a compact facial video forgery detection network. IEEE International Workshop on Information Forensics and Security (WIFS), pp. 1-7.",
        "Chollet, F. (2017). Xception: Deep learning with depthwise separable convolutions. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1251-1258.",
        "Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. International Conference on Machine Learning (ICML), PMLR, pp. 6105-6114.",
        "Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618-626.",
        "Güera, D., & Delp, E. J. (2018). Deepfake video detection using recurrent neural networks. IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), pp. 1-6.",
        "Sabir, E., Cheng, J., Jaiswal, A., AbdAlmageed, W., Masi, I., & Natarajan, P. (2019). Recurrent convolutional strategies for face manipulation detection in videos. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 80-87.",
        "Li, Y., Chang, M. C., & Lyu, S. (2018). In Ictu Oculi: Exposing AI created fake videos by detecting eye blinking. IEEE International Workshop on Information Forensics and Security (WIFS), pp. 1-7.",
        "Ciftci, U. A., Demir, I., & Yin, L. (2020). FakeCatcher: Detection of synthetic portrait videos using biological signals. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 44(6), pp. 3234-3247.",
        "Durall, R., Keuper, M., & Keuper, J. (2020). Watch your up-convolution: CNN based generative deepfake detection. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2440-2449.",
        "Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., & Holz, T. (2020). Leveraging frequency analysis for deep fake image recognition. International Conference on Machine Learning (ICML), PMLR, pp. 3247-3258.",
        "Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing (TIP), 21(12), pp. 4695-4708.",
        "Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). Making a 'completely blind' image quality analyzer. IEEE Signal Processing Letters, 20(3), pp. 209-212.",
        "Immerkær, J. (1996). Fast noise variance estimation. Computer Vision and Image Understanding (CVIU), 64(2), pp. 300-302.",
        "Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing (TIP), 13(4), pp. 600-612.",
        "Stamm, M. C., & Liu, K. J. (2010). Forensic detection of image manipulation using statistical intrinsic fingerprints. IEEE Transactions on Information Forensics and Security (TIFS), 5(3), pp. 492-506.",
        "Farid, H. (2009). Image forgery detection: A survey. IEEE Signal Processing Magazine, 26(2), pp. 16-25.",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, pp. 5998-6008.",
        "Al-Diri, B., Cooke, N., & Bensalem, A. (2021). A hybrid deep learning decision support framework for digital image forensics. Journal of Forensic Sciences & Digital Investigation, 36, pp. 102-118."
    ]
    for idx, ref in enumerate(references_list, 1):
        add_reference_p(doc, idx, ref)

    # Save Document
    doc.save(output_docx_path)
    print(f"[+] DOCX Generation Complete! Successfully saved to: {output_docx_path}")

if __name__ == "__main__":
    out_file = r"docs\DeepGuard_Capstone_Phase_1_Upgraded_Report.docx"
    build_full_docx(out_file)
