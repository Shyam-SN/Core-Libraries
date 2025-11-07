# PointCloudCore: Mathematical & Physics-Grounded Foundations Handbook

This handbook provides graduate-level first-principles derivations and physical grounding for every core module in `PointCloudCore`.

---

## 1. Local Geometry & PCA Surface Normal Estimation

### 📐 Mathematical Derivation
For a point $\mathbf{p}_i \in \mathbb{R}^3$ with $k$-NN neighborhood $\mathcal{N}(i) = \{\mathbf{p}_1, \dots, \mathbf{p}_k\}$:

#### Local Covariance Matrix
$$\bar{\mathbf{p}} = \frac{1}{k} \sum_{j=1}^k \mathbf{p}_j, \quad \mathbf{C} = \frac{1}{k} \sum_{j=1}^k (\mathbf{p}_j - \bar{\mathbf{p}}) (\mathbf{p}_j - \bar{\mathbf{p}})^T \in \mathbb{R}^{3 \times 3}$$

#### Eigen-Decomposition & Surface Normal
Since $\mathbf{C}$ is symmetric positive semi-definite, it decomposes as:
$$\mathbf{C} \mathbf{v}_m = \lambda_m \mathbf{v}_m, \quad \text{with eigenvalues } \lambda_0 \le \lambda_1 \le \lambda_2$$
- The estimated surface normal $\mathbf{n}_i$ corresponds to the eigenvector $\mathbf{v}_0$ associated with the smallest eigenvalue $\lambda_0$.
- Local surface curvature estimate $\sigma(\mathbf{p}_i)$:
  $$\sigma(\mathbf{p}_i) = \frac{\lambda_0}{\lambda_0 + \lambda_1 + \lambda_2}$$

### ⚛️ Physics-Grounded Intuition
- **Moment of Inertia & Minimum Mechanical Resistance**:
  - The spatial covariance matrix $\mathbf{C}$ is directly proportional to the **Mass Moment of Inertia Tensor** $\mathbf{I} = \text{tr}(\mathbf{C}) I_{3 \times 3} - \mathbf{C}$ of a uniform point mass distribution.
  - The eigenvector $\mathbf{v}_0$ associated with $\lambda_0$ represents the axis of minimum variance (and minimum mechanical moment of inertia), which physically aligns with the vector normal to the local tangent plane.

---

## 2. Spatial Downsampling & Farthest Point Sampling (FPS)

### 📐 Mathematical Derivation
Given point set $P = \{\mathbf{p}_1, \dots, \mathbf{p}_N\}$, iteratively construct subset $S_m = \{\mathbf{s}_1, \dots, \mathbf{s}_m\}$:

1. Select initial arbitrary seed $\mathbf{s}_1 \in P$.
2. For iteration $k = 2, \dots, m$, select $\mathbf{s}_k$ maximizing the minimum distance to already sampled points:
   $$\mathbf{s}_k = \arg\max_{\mathbf{x} \in P \setminus S_{k-1}} \left( \min_{\mathbf{y} \in S_{k-1}} \|\mathbf{x} - \mathbf{y}\|_2 \right)$$

#### Coverage Guarantee
FPS provides a 2-approximation to the optimal $k$-center spatial clustering problem, guaranteeing uniform geometric coverage across complex non-Euclidean 3D surface manifolds.

### ⚛️ Physics-Grounded Intuition
- **Coulomb Repulsion & Electrostatic Energy Minimization**:
  - FPS mimics **Electrostatic Repulsion** in a system of like-charged particles constrained to a surface manifold. Particles push away from each other to minimize total Coulomb potential energy:
    $$U = \sum_{i < j} \frac{q_2}{\|\mathbf{r}_i - \mathbf{r}_j\|}$$
  - The steady-state equilibrium configuration yields maximally equidistant point sampling across the surface.

---

## 3. Fast Point Feature Histograms (FPFH)

### 📐 Mathematical Derivation
For point pair $(\mathbf{p}_i, \mathbf{p}_j)$ with surface normals $(\mathbf{n}_i, \mathbf{n}_j)$, define the local **Darboux Frame** at $\mathbf{p}_i$:
$$\mathbf{u} = \mathbf{n}_i, \quad \mathbf{v} = (\mathbf{p}_j - \mathbf{p}_i) \times \mathbf{u}, \quad \mathbf{w} = \mathbf{u} \times \mathbf{v}$$

#### Pairwise Angular Features
$$\alpha = \mathbf{v} \cdot \mathbf{n}_j, \quad \phi = \mathbf{u} \cdot \frac{\mathbf{p}_j - \mathbf{p}_i}{\|\mathbf{p}_j - \mathbf{p}_i\|}, \quad \theta = \arctan(\mathbf{w} \cdot \mathbf{n}_j, \mathbf{u} \cdot \mathbf{n}_j)$$

#### FPFH Aggregation
1. Compute Simplified Point Feature Histogram $\text{SPFH}(\mathbf{p}_i)$ by binning $(\alpha, \phi, \theta)$ over $k$-NN pairs.
2. Weight neighbor SPFH descriptors by inverse spatial distance $w_i = \frac{1}{\|\mathbf{p}_i - \mathbf{p}_j\|}$:
   $$\text{FPFH}(\mathbf{p}_i) = \text{SPFH}(\mathbf{p}_i) + \frac{1}{k} \sum_{j=1}^k \frac{1}{w_j} \text{SPFH}(\mathbf{p}_j)$$

### ⚛️ Physics-Grounded Intuition
- **Differential Micro-Geometry & Invariant Curvature Fields**:
  - The Darboux frame $(\mathbf{u}, \mathbf{v}, \mathbf{w})$ forms a intrinsic coordinate system independent of external $SE(3)$ reference frames.
  - The angular tuple $(\alpha, \phi, \theta)$ measures local principal curvatures (Gaussian curvature $K = \kappa_1 \kappa_2$ and Mean curvature $H = \frac{1}{2}(\kappa_1 + \kappa_2)$), creating a 33D feature signature invariant to rigid spatial motions.

---

## 4. PointNet++ Set Abstraction & Feature Propagation

### 📐 Mathematical Derivation

#### Set Abstraction (SA) Layer
$$\text{SA}(X) = \text{MaxPool}_{j \in \mathcal{N}(i)} \left( \text{MLP}\left( [x_j - x_i, f_j] \right) \right)$$

#### Feature Propagation (FP) Layer
Interpolates features from sparse points $y_j$ to dense points $x_i$ using Inverse Distance Weighted (IDW) $k$-NN average:
$$f(x_i) = \frac{\sum_{j=1}^k w_j(x_i) f(y_j)}{\sum_{j=1}^k w_j(x_i)}, \quad \text{where } w_j(x_i) = \frac{1}{\|x_i - y_j\|^2}$$

### ⚛️ Physics-Grounded Intuition
- **Renormalization Group Coarse-Graining**:
  - Set Abstraction implements **Renormalization Group (RG)** block-spin transformations, aggregating microscopic local atomic details into macroscopic spatial feature representations while guaranteeing **Permutation Invariance** under particle indices swapping.

---

## 5. Point Transformer Vector Self-Attention

### 📐 Mathematical Derivation
For feature vectors $\mathbf{x}_i \in \mathbb{R}^d$ at spatial coordinates $\mathbf{p}_i \in \mathbb{R}^3$:

#### Vector Self-Attention Equation
$$\mathbf{y}_i = \sum_{j \in \mathcal{N}(i)} \text{Softmax}\left( \gamma\left( \psi(\mathbf{x}_i) - \phi(\mathbf{x}_j) + \delta_{ij} \right) \right) \odot \left( \alpha(\mathbf{x}_j) + \delta_{ij} \right)$$

#### Relative Position Encoding
$$\delta_{ij} = \theta(\mathbf{p}_i - \mathbf{p}_j)$$
where $\psi, \phi, \alpha, \gamma, \theta$ are linear/MLP projections, and $\odot$ denotes element-wise Hadamard product.

### ⚛️ Physics-Grounded Intuition
- **Anisotropic Force Fields & Multi-Body Interactions**:
  - Unlike scalar dot-product attention ($\text{Softmax}(Q K^T)$), **Vector Attention** computes channel-wise anisotropic attention weights. This acts as a direction-dependent spatial force field between multi-body particle systems, modulating information transfer along principal physical axes.

---

## 6. PointNeXt Scaled Inverted Residual Bottlenecks

### 📐 Mathematical Derivation
Given input feature $x$:
$$y = \text{Conv}_{1 \times 1}\left( \text{SetAbstraction}\left( \text{Conv}_{1 \times 1}(x) \right) \right) + x$$

### ⚛️ Physics-Grounded Intuition
- **Energy-Preserving Residual Dynamics**:
  - Residual connections ($+ x$) act as continuous-time Euler discretizations of differential equations $\frac{dx}{dt} = \mathcal{F}(x)$, preserving signal norm energy and preventing vanishing gradients during deep multi-scale propagation.

---

## 7. 3D Distance & Evaluation Metrics

### 📐 Mathematical Derivation

#### Symmetric Chamfer Distance (CD)
$$d_{\text{CD}}(P_1, P_2) = \frac{1}{|P_1|} \sum_{x \in P_1} \min_{y \in P_2} \|x - y\|_2^2 + \frac{1}{|P_2|} \sum_{y \in P_2} \min_{x \in P_1} \|x - y\|_2^2$$

### ⚛️ Physics-Grounded Intuition
- **Harmonic Spring Potential Energy**:
  - Chamfer distance equals the average **Elastic Potential Energy** $U = \frac{1}{2} k \Delta x^2$ stored in zero-rest-length virtual linear springs connecting nearest surface points between two physical shapes.
