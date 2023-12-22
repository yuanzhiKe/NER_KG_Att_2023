from treelib import Node, Tree
from collections import defaultdict

class Ontology:
    def __init__(self):
        self.tree = Tree()

    def get_dfs_encoded_str(self):
        # pre_order dfs
        encoded_str = []
        for n in self.tree.expand_tree(nid="root", mode=Tree.DEPTH): # If @key is None sorting is performed on node tag.
            if self.tree[n].tag != "root":
                encoded_str.append(self.tree[n].tag)
        return " ".join(encoded_str).strip()
    
    def get_post_order_encoded_str(self):
        tree = self.tree
        root = "root"
        # WIP
        # https://zhuanlan.zhihu.com/p/566673074
        encoded_str = []
        stack = []
        nextIndex = defaultdict(int)
        node = root
        while stack or node:
            while node:
                stack.append(node)
                if not tree.children(node):
                    break
                nextIndex[node] = 1
                node = tree.children(node)[0].identifier
            node = stack[-1]
            i = nextIndex[node]
            if i < len(tree.children(node)):
                nextIndex[node] = i + 1
                node = tree.children(node)[i].identifier
            else:
                encoded_str.append(tree.get_node(node).tag)
                stack.pop()
                del nextIndex[node]
                node = None
        return " ".join(encoded_str).strip()

    def get_pre_post_order_encoded_str(self):
        return self.get_dfs_encoded_str() + " " + self.get_post_order_encoded_str()
    


class EN_Ontology(Ontology):
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

class CBLUE_Ontology(Ontology):
    def __init__(self) -> None:
        ontology_tree = Tree()
        ontology_tree.create_node("", "root")  # root node

        ontology_tree.create_node("疾病", "disease", parent="root")
        ontology_tree.create_node("疾病名称", parent="disease")
        ontology_tree.create_node("疾病病因", parent="disease")
        ontology_tree.create_node("临床症状", parent="disease")
        ontology_tree.create_node("诊断标准", parent="disease")

        ontology_tree.create_node("症状", "sympton", parent="root")
        ontology_tree.create_node("体征", parent="sympton")
        ontology_tree.create_node("表现性质", parent="sympton")
        ontology_tree.create_node("表现时序", parent="sympton")

        ontology_tree.create_node(
            "医疗设备", "medical equipment", parent="root"
        )
        ontology_tree.create_node("设备名称", parent="medical equipment")
        ontology_tree.create_node(
            "适用症状", parent="medical equipment"
        )
        ontology_tree.create_node(
            "操作要求", parent="medical equipment"
        )

        ontology_tree.create_node("身体", "anatomical location", parent="root")
        ontology_tree.create_node("部位名称", parent="anatomical location")
        ontology_tree.create_node("部位结构", parent="anatomical location")
        ontology_tree.create_node("部位性质", parent="anatomical location")

        ontology_tree.create_node(
            "部门", "hospital department", parent="root"
        )
        ontology_tree.create_node("部门名称", parent="hospital department")
        ontology_tree.create_node(
            "部门职能", parent="hospital department"
        )
        ontology_tree.create_node("服务对象", parent="hospital department")
        ontology_tree.create_node("作用范围", parent="hospital department")

        ontology_tree.create_node("微生物", "microbe", parent="root")
        ontology_tree.create_node("微生物名称", parent="microbe")
        ontology_tree.create_node("微生物工作职能", parent="microbe")
        ontology_tree.create_node("生物学特征", parent="microbe")
        ontology_tree.create_node("致病机理", parent="microbe")

        ontology_tree.create_node("药物", "drug", parent="root")
        ontology_tree.create_node("药物名称", parent="drug")
        ontology_tree.create_node("药物成分", parent="drug")
        ontology_tree.create_node("适应症", parent="drug")
        ontology_tree.create_node("禁忌症", parent="drug")
        ontology_tree.create_node("剂量", parent="drug")

        self.tree = ontology_tree

if __name__=="__main__":
    tree = EN_Ontology()
    print(tree.tree)
    print(tree.get_dfs_encoded_str())
    # print(tree.get_pre_post_order_encoded_str())