class Node:
    def __init__(self, vector, level, file_name=None, root=False):
        self.vector = vector
        self.child = list()
        self.parent = None
        self.is_root = root
        self.file_name = file_name
        self.is_buggy = None
        self.level = level
        self.all_node_list = None

    def add_child(self, child_node):
        for item in self.child:
            if item is self:
                pass
        child_node.level = self.level + 1
        child_node.parent = self
        self.child.append(child_node)

    def set_parent(self, parent_node):
        self.parent = parent_node
        parent_node.add_child(self)