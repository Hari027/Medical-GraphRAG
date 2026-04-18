import re

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = r'clear_btn = st\.button\("🗑️ Clear Graph".*?st\.rerun\(\)\n'

match = re.search(target, text, flags=re.DOTALL)
if match:
    old = match.group(0)
    new_text = old + '''
    clear_rel_btn = st.button("🗑️ Clear Relationships Only", help="Wipes all relationship edges but keeps the nodes intact")
    if clear_rel_btn:
        st.session_state.med_rag.clear_all_relationships()
        st.success("✅ All relationships successfully cleared")
        st.rerun()
'''
    text = text.replace(old, new_text)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success modifying app.py")
else:
    print("Not found")
