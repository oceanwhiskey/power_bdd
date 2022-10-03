class Node:
    id = 1
    def __init__(self, label):
        self.t = None
        self.e = None
        self.marked = False
        self.count = None
        self.paths = 0     
        self.label = label    
        self.id = Node.id
        Node.id += 1

    def __eq__(self, other):
        return self.t is other.t and self.e is other.e

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f'({self.label},{self.id})'        

class Node_AVL(Node):
    def __init__(self, label, lb, ub):
        super(Node_AVL, self).__init__(label)
        self.lb = lb
        self.ub = ub
        self.id = Node.id
        Node.id += 1

    def __eq__(self, other):
        return self.ub == other.ub

    def __lt__(self, other):
        return self.ub < other.ub

    def __repr__(self):
        return f'({self.lb}, {self.ub}]'

    def __hash__(self):
        return self.id
       

