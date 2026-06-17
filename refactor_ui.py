with open('App.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i == 274: # line 275: '        with gr.Row():'
        new_lines.append('        with gr.Tabs():\n')
        new_lines.append('            with gr.Tab("Interactive Analysis"):\n')
        new_lines.append('                with gr.Row():\n')
    elif 275 <= i <= 413:
        # indent by 8 spaces
        new_lines.append('        ' + line)
    elif i == 415: # line 416: '        # ── ADDITIONAL TABS FOR DEV/TRAINING ──'
        new_lines.append('            # ── ADDITIONAL TABS FOR DEV/TRAINING ──\n')
    elif i == 416: # line 417: '        with gr.Accordion("⚙️ Developer & Training Tools", open=False):'
        new_lines.append('            with gr.Tab("Dataset & Pipeline Training"):\n')
    else:
        new_lines.append(line)

with open('App.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Refactoring complete')
