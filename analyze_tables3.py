"""Check body tables (actual data tables, not ToC) more deeply including paragraph style inheritance"""
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

doc = Document(r'c:\Users\arbab\Desktop\myfrontend\FYP_Report_111.docx')

# Check tables 4-10 (actual content tables)
for t_idx in range(4, 12):
    table = doc.tables[t_idx]
    print(f'\n=== TABLE {t_idx+1} | {len(table.rows)}r x {len(table.columns)}c ===')

    # Print first 2 rows content
    for r_idx, row in enumerate(table.rows[:4]):
        cells_text = [c.text[:30].strip().encode('ascii','replace').decode() for c in row.cells[:4]]
        print(f'  row{r_idx}: {cells_text}')

        # Check first cell borders and shading for each row
        if r_idx <= 1:
            for c_idx, cell in enumerate(row.cells[:2]):
                tc_pr = cell._tc.find(qn('w:tcPr'))
                if tc_pr is not None:
                    # vertical alignment
                    va = tc_pr.find(qn('w:vAlign'))
                    if va is not None:
                        print(f'    cell[{c_idx}] vAlign: {va.get(qn("w:val"))}')
                    # shading
                    shd = tc_pr.find(qn('w:shd'))
                    if shd is not None:
                        fill  = shd.get(qn('w:fill'), '')
                        theme = shd.get(qn('w:themeFill'), '')
                        tint  = shd.get(qn('w:themeFillTint'), '')
                        shade_val = shd.get(qn('w:val'), '')
                        print(f'    cell[{c_idx}] shd: fill=#{fill} theme={theme} tint={tint} val={shade_val}')
                    # borders
                    borders = tc_pr.find(qn('w:tcBorders'))
                    if borders is not None:
                        for b in borders:
                            tag   = b.tag.split('}')[-1]
                            val   = b.get(qn('w:val'), '')
                            color = b.get(qn('w:color'), '')
                            sz    = b.get(qn('w:sz'), '')
                            if color not in ('auto', 'FFFFFF', ''):
                                print(f'    cell[{c_idx}] border[{tag}]: val={val} color=#{color} sz={sz}')

# Now get the Normal paragraph style formatting (body text)
print('\n=== BODY PARAGRAPH FORMATTING (from Normal style) ===')
normal_style = doc.styles['Normal']
ns_font = normal_style.font
pf = normal_style.paragraph_format
print(f'  Font name : {ns_font.name}')
print(f'  Font size : {ns_font.size}')
print(f'  Space before: {pf.space_before}')
print(f'  Space after : {pf.space_after}')
print(f'  Line spacing: {pf.line_spacing}')

# Check the "Table Paragraph" style if it exists
for style_name in ['Table Paragraph', 'Table Grid', 'Normal Table']:
    try:
        s = doc.styles[style_name]
        print(f'\n=== STYLE: {style_name} ===')
        print(f'  font.name={s.font.name}, font.size={s.font.size}')
        pf2 = s.paragraph_format
        print(f'  space_before={pf2.space_before}, space_after={pf2.space_after}')
    except Exception as e:
        print(f'  Style "{style_name}" not found: {e}')
