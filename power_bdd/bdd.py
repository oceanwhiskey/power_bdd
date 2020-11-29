from sortedcontainers import SortedList
from power_bdd.node import Node, Node_AVL

class WeightedGame:
    def __init__(self, q, w):
        self.q = q          # quota, can be float
        self.n = len(w)     # number of players
        self.wN = sum(w)    # w(N) is the sum of all weights
        self.w = w          # list of voting weights, should be integers or fractions; no floats due to rounding errors.
        self.w.insert(0, 0) # w[i] is voting weight of player i, w[0] has no meaning

class BDD:
    """Creates the ROBDD of a weighted game and calculates power indices according to Banzhaf/Penrose and Shapley/Shubik.

    This method allows to easily connect bdds with AND or OR and is also suited for voting systems with multiple layers.
    The method was published by S. Bolus:
    https://www.sciencedirect.com/science/article/abs/pii/S0377221710006181
    https://macau.uni-kiel.de/receive/diss_mods_00009114
    If you are interested in calculating power indices you should also check out the website:
    https://www.informatik.uni-kiel.de/~progsys/simple_games/lab/lab.html
    basic usage:
    >>> q = 4
    >>> w = [1, 2, 3]
    >>> game1 = WeightedGame(q, w)
    >>> bdd = BDD(game1)
    >>> bdd.calc_banzhaf()
    [0.25, 0.25, 0.75]"""    

    def __init__(self, game=None):
        self.game = game
        if game is not None:
            self.n = game.n
            self.node_I = Node(label=self.n + 1)
            self.node_O = Node(label=self.n + 1)

    def create_BDD(self):
        lb, ub = float('-inf'), 0
        node = Node_AVL(label=self.n+1, lb=lb, ub=ub)
        self.node_I = node

        lb, ub = 0, float('inf')
        node = Node_AVL(label=self.n+1, lb=lb, ub=ub)
        self.node_O = node

        self.count = 0
        self.AVL = []
        self.uniqueTable = list()
        for _ in range(self.n + 1):
            self.AVL.append(SortedList([]))
            self.uniqueTable.append(dict())
        self.root = self.create(i=1, q=self.game.q, w_left=self.game.wN)
        self.AVL = None    
        return self

    def create(self, i, q, w_left):         
        if i == self.n + 1:
            if q > w_left:
                return self.node_O
            else:
                return self.node_I                
        else:       
            node = Node_AVL(label=i, lb=q, ub=q) if q > 0 else Node_AVL(label=i, lb=0, ub=0)
            ind = self.AVL[i].bisect_left(node)
            if ind < len(self.AVL[i]):
                node_found = self.AVL[i].__getitem__(ind)
                if node_found.lb < max(0, q):
                    return node_found      

            w_i = self.game.w[i]
            t = self.create(i=i+1, q=q-w_i, w_left=w_left-w_i)
            e = self.create(i=i+1, q=q, w_left=w_left-w_i)

            if q <= 0:
                lb, ub = float('-inf'), 0
            elif q > w_left:
                lb, ub = w_left, float('inf')
            else:
                lb, ub = max(t.lb + w_i, e.lb), min(t.ub + w_i, e.ub)
            node = Node_AVL(label=i, lb=lb, ub=ub)
            node.t, node.e = t, e
            self.addnode(node)
            self.uniqueTable[i][(t, e)] = node # just for being able to manage nodes later using uniqueTable 
            return node

    @staticmethod
    def create_BDD_if_not_already_done(bdd):
        if not hasattr(bdd, 'uniqueTable'):
            bdd.create_BDD()

    def ite(self, i, t, e):
        key = (t, e)
        if key in self.uniqueTable[i]:
            return self.uniqueTable[i].get(key)
        else:
            node = Node(label=i)
            node.t, node.e = t, e
            self.uniqueTable[i][key] = node
            return node

    def addnode(self, node):
        i = node.label
        self.AVL[i].add(node)
        if i == 1:
            self.root = node
        self.count += 1
        if self.count % 100000 == 0:
            print(self.count)

    @staticmethod
    def traverse(bdd, node, f_node, f_sink=None, unmarked=False):
        if node.marked == unmarked:
            node.marked = not unmarked  
            if node.label == bdd.n + 1:
                if f_sink is not None:
                    f_sink(node) 
            else:
                BDD.traverse(bdd=bdd, node=node.t, f_sink=f_sink, f_node=f_node, unmarked=unmarked)
                BDD.traverse(bdd=bdd, node=node.e, f_sink=f_sink, f_node=f_node, unmarked=unmarked)
                f_node(node)

    @staticmethod
    def connect_BDD(bdd1, bdd2, connector):
        bdd = BDD() 
        bdd.n = bdd1.n

        bdd.node_I = Node(label=bdd.n + 1)
        bdd.node_O = Node(label=bdd.n + 1)

        bdd.count = 0   

        bdd.uniqueTable = list()
        bdd.computedTable = list()
        for _ in range(bdd.n + 1):
            bdd.uniqueTable.append(dict())
            bdd.computedTable.append(dict())

        bdd.root = bdd.connect(bdd1, bdd2, bdd1.root, bdd2.root, connector)
        # print(f'{connector.__name__}: BDD created with {sum(map(lambda x: len(x), bdd.uniqueTable))} nodes.')  
        return bdd      

    def connect(self, bdd1, bdd2, u, v, connector):
        i = u.label
        if i == self.n + 1:
            if connector(bdd1, bdd2, u, v):
                return self.node_I
            else:
                return self.node_O
        else:
            key = (u, v)
            if key in self.computedTable[i]:
                return self.computedTable[i].get(key)
            else:
                t = self.connect(bdd1, bdd2, u.t, v.t, connector)
                e = self.connect(bdd1, bdd2, u.e, v.e, connector)
                node = self.ite(i=i, t=t, e=e)
                self.computedTable[i][key] = node
                return node    

    def preparecount(self, node, unmarked=False):
        def f_sink(u): 
            if u is self.node_I:
                u.count = 1
            elif u is self.node_O:
                u.count = 0            
        
        def f_node(u):
            u.count = u.t.count + u.e.count  

        BDD.traverse(self, node, f_node=f_node, f_sink=f_sink, unmarked=unmarked)

    def preparepaths(self):
        self.root.paths = 1
        for i in range(1, self.n+1):
            nodes = self.uniqueTable[i].values()
            for node in nodes:
                node.t.paths += node.paths
                node.e.paths += node.paths

    def calc_banzhaf(self):
        BDD.create_BDD_if_not_already_done(self)

        self.beta = [0 for i in range(self.n + 1)]
        self.pow2 = 2**(self.n-1)

        self.preparecount(self.root, unmarked=self.root.marked)
        self.preparepaths()

        for i in range(1, self.n + 1):
            beta_i = 0
            nodes = self.uniqueTable[i].values()

            for node in nodes:
                beta_i += node.paths * (node.t.count - node.e.count)
            self.beta[i] = beta_i / self.pow2
        return self.beta[1:]
       
    @staticmethod
    def calc_binom(n):
        binom = [1, n]
        prev = n
        for i in range(1, n):
            prev = prev * (n - i) // (i + 1)
            binom.append(prev)
        for i in range(1, n + 1):
            binom[i] = 1 / binom[i] / i
        return binom

    @staticmethod
    def diff_dash(t, e):
        n = max(len(t), len(e))
        c = [x for x in t] + [0] * (n - len(t))
        for i in range(0, len(e)):
            c[i] -= e[i]
        return c

    @staticmethod
    def add_dash(t, e):
        n = max(len(t), len(e))
        c = [x for x in t] + [0] * (n - len(t))
        for i in range(0, len(e)):
            c[i] += e[i]
        return c        
   
    @staticmethod
    def circ_shifted(u, v, shift):
        h, l = len(v), len(u)
        d = [0] * (h + l - 1 + shift)
        for i in range(1, h + l - 1 + shift):
            sum = 0
            if i <= h:
                for k in range(1, min(i, l) + 1):
                    sum += u[k - 1] * v[i - k]
            else:
                for k in range(1, min(h, h - i + l) + 1):
                    sum += u[i - h + k - 1] * v[h - k]
            d[i - 1 + shift] = sum
        return d

    def preparecount_dash(self):
        def f_node_init(u): u.count = [0] * (self.n - u.label + 2)

        def f_sink_init(u): 
            if u is self.node_I:
                u.count = [1]
            elif u is self.node_O:
                u.count = [0]   

        BDD.traverse(self, self.root, f_node=f_node_init, f_sink=f_sink_init, unmarked=self.root.marked) 

        def f_node_process(u):
            if u.e.count is None:
                print(u, u.e)
            for i in range(0, self.n - u.label + 1):
                u.count[i] += u.e.count[i]
                u.count[i + 1] += u.t.count[i]       
              
        BDD.traverse(self, self.root, f_node=f_node_process, f_sink=None, unmarked=self.root.marked)                       

    def preparepaths_dash(self):
        # init
        def f(u): u.paths = [0] * u.label
        BDD.traverse(self, self.root, f_node=f, f_sink=f, unmarked=self.root.marked) 
        self.root.paths = [1]

        for i in range(1, self.n + 1):
            nodes = self.uniqueTable[i].values()
            for node in nodes:                
                for j in range(0, node.label):
                    node.t.paths[j + 1] += node.paths[j]
                    node.e.paths[j] += node.paths[j]

    def calc_shapley(self):
        self.binom = BDD.calc_binom(self.n) 
        BDD.create_BDD_if_not_already_done(self)
        self.preparecount_dash()
        self.preparepaths_dash()
        self.phi = [0 for i in range(self.n + 1)]
        for i in range(1, self.n + 1):
            d = [0] * (self.n + 1)

            nodes = self.uniqueTable[i].values()
            for node in nodes:
                count = BDD.diff_dash(node.t.count, node.e.count)
                delta = BDD.circ_shifted(node.paths, count, shift=1)
                d = BDD.add_dash(d, delta)
    
            phi = 0
            for j in range(1, self.n + 1):
                # phi += d[j] / math.comb(self.n, j) / j
                phi += d[j] * self.binom[j]
            self.phi[i] = phi
        return self.phi[1:]
      
   
    def __eq__(self, other):
        self.computedTable = list()
        for _ in range(self.n + 1):
            self.computedTable.append(dict())
        return self.equals(other, self.root, other.root)

    def equals(self, other, u, v):
        i = u.label
        if i == self.n + 1:
            return (u is self.node_I and v is other.node_I) or (u is self.node_O and v is other.node_O)
        else:
            key = (u, v)
            if key in self.computedTable[i]:
                return self.computedTable[i].get(key)
            else:
                t = self.equals(other, u.t, v.t)
                e = self.equals(other, u.e, v.e)
                self.computedTable[i][key] = t and e
                return t and e

    @staticmethod
    def and_connector(bdd, other, u, v):
        return u is bdd.node_I and v is other.node_I

    @staticmethod
    def or_connector(bdd, other, u, v):
        return u is bdd.node_I or v is other.node_I
    
    def __and__(self, other):
        BDD.create_BDD_if_not_already_done(self)
        BDD.create_BDD_if_not_already_done(other)
        return BDD.connect_BDD(self, other, BDD.and_connector)

    def __or__(self, other):
        BDD.create_BDD_if_not_already_done(self)
        BDD.create_BDD_if_not_already_done(other)
        return BDD.connect_BDD(self, other, BDD.or_connector)
         
def main():
    import datetime
    t0 = datetime.datetime.now()

    game1 = WeightedGame(218, [0, 0] + [0] * 100 + [1] * 435)
    game2 = WeightedGame(51,  [0, 0] + [1] * 100 + [0] * 435)
    game3 = WeightedGame(1,   [1, 0] + [0] * 100 + [0] * 435)
    game4 = WeightedGame(50,  [0, 0] + [1] * 100 + [0] * 435)
    game5 = WeightedGame(1,   [0, 1] + [0] * 100 + [0] * 435)
    game6 = WeightedGame(290, [0, 0] + [0] * 100 + [1] * 435)
    game7 = WeightedGame(67,  [0, 0] + [1] * 100 + [0] * 435)

    federal_system = BDD(game1) & BDD(game2) & BDD(game3) | BDD(game1) & BDD(game4) & BDD(game5) & BDD(game3) | BDD(game6) & BDD(game7)
    beta = federal_system.calc_banzhaf() # be patient, takes probably more than 10 seconds, the calc_shapley() takes approx. 800 seconds.
    t1 = datetime.datetime.now()

    print(beta)
    print(f'The ROBDD has {sum(map(lambda x: len(x), federal_system.uniqueTable))} nodes.')
    print(f'finished in {(t1 - t0).total_seconds()} seconds.')

if __name__ == '__main__': 
    main()
