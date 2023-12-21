from treelib import Node, Tree
from collections import defaultdict


class EN_Ontology_tree:
    def __init__(self) -> None:
        ontology_tree = Tree()
        ontology_tree.create_node("", "root")  # root node

        ontology_tree.create_node("disease", "disease", parent="root")
        ontology_tree.create_node("disease name", parent="disease")
        ontology_tree.create_node("cause", parent="disease")
        ontology_tree.create_node("clinical manifestations", parent="disease")
        ontology_tree.create_node("diagnostic criteria", parent="disease")

        ontology_tree.create_node("sympton", "sympton", parent="root")
        ontology_tree.create_node("symptom signs", parent="sympton")
        ontology_tree.create_node("symptom property", parent="sympton")
        ontology_tree.create_node("symptom timing", parent="sympton")

        ontology_tree.create_node(
            "medical equipment", "medical equipment", parent="root"
        )
        ontology_tree.create_node("equipment name", parent="medical equipment")
        ontology_tree.create_node(
            "symptoms suitable for use", parent="medical equipment"
        )
        ontology_tree.create_node(
            "operational requirements", parent="medical equipment"
        )

        ontology_tree.create_node("body", "anatomical location", parent="root")
        ontology_tree.create_node("part name", parent="anatomical location")
        ontology_tree.create_node("anatomy", parent="anatomical location")
        ontology_tree.create_node("anatomical property", parent="anatomical location")

        ontology_tree.create_node(
            "hospital department", "hospital department", parent="root"
        )
        ontology_tree.create_node("department name", parent="hospital department")
        ontology_tree.create_node(
            "departmental functions", parent="hospital department"
        )
        ontology_tree.create_node("service object", parent="hospital department")
        ontology_tree.create_node("range of action", parent="hospital department")

        ontology_tree.create_node("microbe", "microbe", parent="root")
        ontology_tree.create_node("microbial name", parent="microbe")
        ontology_tree.create_node("microbiology functions", parent="microbe")
        ontology_tree.create_node("biological characteristics", parent="microbe")
        ontology_tree.create_node("pathogenicity", parent="microbe")

        ontology_tree.create_node("drug", "drug", parent="root")
        ontology_tree.create_node("drug name", parent="drug")
        ontology_tree.create_node("drug component", parent="drug")
        ontology_tree.create_node("indications", parent="drug")
        ontology_tree.create_node("contraindication", parent="drug")
        ontology_tree.create_node("dosage", parent="drug")

        self.tree = ontology_tree

    def get_dfs_encoded_str(self):
        # pre_order dfs
        encoded_str = []
        for n in self.tree.expand_tree(nid="root", mode=Tree.DEPTH): # If @key is None sorting is performed on node tag.
            if self.tree[n].tag != "root":
                encoded_str.append(self.tree[n].tag)
        return " ".join(encoded_str)[1:]  # [1:] to drop the first space
    
    def get_post_order_encoded_str(self):
        tree = self.tree
        root = tree.get_node("root")
        # WIP
        # https://zhuanlan.zhihu.com/p/566673074
        ans = []
        stack = []
        nextIndex = defaultdict(int)
        node = root
        while stack or node:
            while node:
                stack.append(node)
                if not tree.children(node):
                    break
                nextIndex[node] = 1
                node = node.children[0]
            node = stack[-1]
            i = nextIndex[node]
            if i < len(node.children):
                nextIndex[node] = i + 1
                node = node.children[i]
            else:
                ans.append(node.val)
                stack.pop()
                del nextIndex[node]
                node = None
        return ans

    def get_pre_post_order_encoded_str(self):
        return self.get_dfs_encoded_str + " " + self.get_post_order_encoded_str
