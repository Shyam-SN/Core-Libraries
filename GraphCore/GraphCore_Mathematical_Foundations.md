# GraphCore: Mathematical Foundations & Proofs Handbook

This handbook provides graduate-level first-principles derivations, proofs of correctness, and physical grounding for every core algorithm in `GraphCore`.

---

## 1. Shortest Path Optimality & Heuristics (Dijkstra & A*)

### 📐 Mathematical Derivation & Proof of Correctness

#### Dijkstra's Algorithm
For a graph $G = (V, E)$ with non-negative edge weights $w(u, v) \ge 0$:

**Loop Invariant**: At the beginning of each iteration, for every vertex $u$ in the settled set $S$, $d[u] = \delta(s, u)$ (the true shortest distance from source $s$).

**Proof by Induction**:
- **Base Case**: Initially $S = \{s\}$ and $d[s] = 0 = \delta(s, s)$, which holds.
- **Inductive Step**: Assume invariant holds for $S$. Let $v$ be the unsettled vertex with minimum key $d[v]$ extracted from the priority queue. Suppose there exists a shorter path $P$ from $s$ to $v$.
- Let $(x, y)$ be the first edge in $P$ such that $x \in S$ and $y \notin S$.
- $\delta(s, v) = \delta(s, x) + w(x, y) + \delta(y, v) \ge d[x] + w(x, y) \ge d[y]$.
- Since non-negative weights imply $\delta(y, v) \ge 0$, and $v$ was chosen with minimum key, $d[v] \le d[y] \le \delta(s, v)$.
- Thus $d[v] = \delta(s, v)$, maintaining the invariant. $\blacksquare$

#### A* Search & Consistency (Monotonicity)
A heuristic $h(u)$ is **consistent** (or monotonic) if for every edge $(u, v)$:
$$h(u) \le w(u, v) + h(v)$$
- Consistency implies admissibility ($h(u) \le \delta(u, \text{target})$).
- Under consistency, the $f$-score $f(u) = g(u) + h(u)$ along any path is non-decreasing, ensuring A* never re-opens closed nodes.

### ⚛️ Physics-Grounded Intuition
- **Principle of Least Action & Wave Front Propagation**:
  - Dijkstra's algorithm models **Huygens-Fresnel Wavefront Propagation** in an inhomogeneous medium where edge weights $w(u, v)$ represent optical travel time $dt = \frac{ds}{v(\mathbf{r})}$. The priority queue expands concentric spherical wavefronts minimizing action $S = \int L dt$.

---

## 2. Strongly Connected Components (Tarjan's Algorithm)

### 📐 Mathematical Derivation & Proof of Correctness
In a directed graph $G = (V, E)$, a Strongly Connected Component (SCC) is a maximal sub-graph where every vertex is reachable from every other vertex.

#### Low-Link Value Invariant
For a DFS tree node $u$:
$$\text{low}[u] = \min \begin{cases} \text{disc}[u] \\ \text{disc}[w] & \text{where } (v, w) \text{ is a back-edge from a descendant } v \text{ of } u \text{ in the DFS tree} \end{cases}$$

- A node $u$ is the root of an SCC if and only if $\text{low}[u] = \text{disc}[u]$.
- Nodes are pushed to a recursion stack during DFS. When $\text{low}[u] == \text{disc}[u]$, popping nodes from the stack until $u$ yields the exact SCC. Running time is $\mathcal{O}(V + E)$.

### ⚛️ Physics-Grounded Intuition
- **Closed Thermodynamic Circulation & Vector Potentials**:
  - SCCs represent non-dissipative closed circulation loops $\oint \mathbf{v} \cdot d\mathbf{r} \neq 0$ in vector fields, identifying isolated recirculating fluid eddies or closed electrical current loops.

---

## 3. Minimum Spanning Trees (Kruskal & Prim)

### 📐 Mathematical Derivation & Proof of Correctness

#### Cut Property
For any cut $(S, V \setminus S)$ of graph $G$, if $e = (u, v)$ is the strictly light edge crossing the cut, then $e$ belongs to every MST of $G$.

#### DSU Amortized Complexity
Disjoint Set Union (DSU) with path compression and rank union guarantees operations in $\mathcal{O}(\alpha(N))$ amortized time, where $\alpha(N)$ is the inverse Ackermann function ($\alpha(N) \le 4$ for all physical universes).

### ⚛️ Physics-Grounded Intuition
- **Minimum Energy Resistor Networks**:
  - MSTs represent minimum resistance electrical grids or elastic spring networks minimizing total potential energy $U = \sum \frac{1}{2} k_i (\Delta x_i)^2$.

---

## 4. Maximum Flow & Minimum Cut (Dinic & Edmonds-Karp)

### 📐 Mathematical Derivation & Max-Flow Min-Cut Theorem

#### Max-Flow Min-Cut Theorem
The maximum value of an $s$-$t$ flow equals the minimum capacity of an $s$-$t$ cut:
$$\max |f| = \min_{S, T} c(S, T)$$

#### Dinic's Algorithm
1. Construct level graph $G_L$ via BFS on residual graph $G_f$.
2. Find blocking flow $f'$ in $G_L$ using DFS.
3. Update $G_f \leftarrow G_f - f'$.
- Time complexity on general graphs: $\mathcal{O}(V^2 E)$.

### ⚛️ Physics-Grounded Intuition
- **Fluid Continuity & Bottleneck Capacity**:
  - Max flow models non-compressible fluid flow obeying the continuity equation $\nabla \cdot \mathbf{v} = 0$. Min cut identifies the physical structural bottleneck where cross-sectional hydraulic throughput is saturated.

---

## 5. Bipartite Matching (Hopcroft-Karp)

### 📐 Mathematical Derivation & Berge's Lemma
A matching $M$ in $G$ is maximum if and only if there exists no augmenting path with respect to $M$.
Hopcroft-Karp finds a maximal set of shortest augmenting paths simultaneously in $\mathcal{O}(E \sqrt{V})$ time using BFS + DFS layers.

---

## 6. Centrality & Markov Chains (PageRank)

### 📐 Mathematical Derivation
For transition probability matrix $M_{ij} = \frac{1}{\text{out\_degree}(j)}$ and damping factor $d = 0.85$:
$$\mathbf{p}_{k+1} = \left( d M + \frac{1-d}{N} \mathbf{E} \right) \mathbf{p}_k$$

#### Perron-Frobenius Convergence
Since the Google matrix $A = d M + \frac{1-d}{N} \mathbf{E}$ is primitive and strictly positive, the **Perron-Frobenius Theorem** guarantees a unique stationary eigenvector $\mathbf{p}^*$ with eigenvalue $\lambda_1 = 1$.

### ⚛️ Physics-Grounded Intuition
- **Brownian Diffusion Steady State**:
  - PageRank models Markovian random walk particle diffusion. The stationary vector $\mathbf{p}^*$ represents the steady-state spatial probability density function of Brownian gas particles diffusing across a network.
