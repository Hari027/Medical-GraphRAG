import re

with open('med_graph_rag.py', 'r', encoding='utf-8') as f:
    text = f.read()

match = re.search(r'def clear_all\(self\).*?clear_all_db\(\)', text, flags=re.DOTALL)
if match:
    old = match.group(0)
    new_text = old + '''

    def clear_all_relationships(self):
        """
        Clears all relationships (edges) from memory and Neo4j, but retains nodes.
        """
        self.nx_graph.clear_edges()
        if self.neo4j:
            self.neo4j.clear_all_relationships()'''
    text = text.replace(old, new_text)
    with open('med_graph_rag.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success")
else:
    print("Not found")
