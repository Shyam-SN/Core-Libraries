# DeepLearningCore: Mathematical & Physics-Grounded Foundations Handbook

This handbook provides graduate-level first-principles derivations and physical grounding for every core component in `DeepLearningCore`.

---

## 1. Tensors & Multidimensional Storage

### 📐 Mathematical Derivation
A tensor $\mathcal{T} \in \mathbb{R}^{d_0 \times d_1 \times \dots \times d_{k-1}}$ of order $k$ is a multilinear map $\mathcal{T}: V_1^* \times V_2^* \times \dots \times V_k^* \to \mathbb{R}$.

In computer memory, elements are mapped to a contiguous 1D array via row-major index mapping function $f: \mathbb{N}^k \to \mathbb{N}$:
$$\text{offset}(i_0, i_1, \dots, i_{k-1}) = \sum_{j=0}^{k-1} i_j \cdot S_j$$
where stride $S_j$ defines the memory displacement when advancing index $i_j$ by $1$:
$$S_j = \prod_{m=j+1}^{k-1} d_m, \quad S_{k-1} = 1$$

#### Tensor Broadcasting
Two shapes $\mathbf{A} = (a_0, \dots, a_{m-1})$ and $\mathbf{B} = (b_0, \dots, b_{n-1})$ are compatible for broadcast if, after right-aligning dimensions, $a_i = b_i$ or $a_i = 1$ or $b_i = 1$. The broadcasted tensor shape $\mathbf{C}$ has dimension:
$$c_i = \max(a_i, b_i)$$

### ⚛️ Physics-Grounded Intuition
- **Physical Fields**: Tensors generalize scalar fields (temperature $T(\mathbf{x})$), vector fields (fluid velocity $\mathbf{v}(\mathbf{x})$), and second-order tensor fields (Cauchy stress tensor $\boldsymbol{\sigma}$).
- **Discrete Lattice Sampling**: A 4D tensor $X \in \mathbb{R}^{N \times C \times H \times W}$ represents a discretized physical field sampled across a spatial lattice $(H \times W)$ over $C$ physical state channels and $N$ observation instances.

---

## 2. Reverse-Mode Automatic Differentiation (Autograd)

### 📐 Mathematical Derivation
Let a computational graph be a Directed Acyclic Graph (DAG) $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where vertices $v_i \in \mathcal{V}$ represent intermediate scalar/tensor quantities and directed edges $(v_i, v_j) \in \mathcal{E}$ denote functional dependencies $v_j = f_j(\text{parents}(v_j))$.

For a scalar loss $L = v_N$, the sensitivity $\bar{v}_i \equiv \frac{\partial L}{\partial v_i}$ is given by the multivariate chain rule:
$$\bar{v}_i = \sum_{v_j \in \text{children}(v_i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}$$

#### Vector-Jacobian Product (VJP)
For a vector-valued function $\mathbf{y} = f(\mathbf{x})$ with Jacobian $J_{ij} = \frac{\partial y_i}{\partial x_j}$, the backward pass computes the vector-Jacobian product given incoming gradient $\mathbf{v} = \bar{\mathbf{y}}$:
$$\bar{\mathbf{x}} = J^T \mathbf{v} \quad \implies \quad \bar{x}_j = \sum_i v_i \frac{\partial y_i}{\partial x_j}$$

#### Topological Execution Algorithm
1. Construct DAG during forward evaluation.
2. Perform Depth-First Search (DFS) starting at scalar output $L$.
3. Append vertices to post-order list upon completion of child visits.
4. Reverse post-order list to obtain topological order $(v_N, v_{N-1}, \dots, v_0)$.
5. Execute reverse derivative evaluation along topological order.

### ⚛️ Physics-Grounded Intuition
- **Adjoint State Method & Conservation Laws**: Reverse-mode AD is mathematically equivalent to the **Adjoint State Method** in continuum mechanics and optimal control.
- **Costate Trajectory in Hamiltonian Systems**: In classical mechanics, forward evaluation computes the trajectory $q(t)$ under Hamilton's equations $\dot{q} = \frac{\partial H}{\partial p}$. Backward autograd integrates the adjoint costate variable $\lambda(t)$ backward in time to measure the sensitivity of action functional $S = \int L(q, \dot{q}) dt$ to initial perturbations.

---

## 3. Fully Connected (Linear) Layer Calculus

### 📐 Mathematical Derivation
For input $X \in \mathbb{R}^{N \times D_{in}}$, weight matrix $W \in \mathbb{R}^{D_{in} \times D_{out}}$, and bias vector $b \in \mathbb{R}^{1 \times D_{out}}$:
$$Y = X W + b$$

Given downstream gradient $\bar{Y} = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times D_{out}}$:

$$\frac{\partial L}{\partial W} = X^T \bar{Y} \in \mathbb{R}^{D_{in} \times D_{out}}$$
$$\frac{\partial L}{\partial X} = \bar{Y} W^T \in \mathbb{R}^{N \times D_{in}}$$
$$\frac{\partial L}{\partial b} = \sum_{n=1}^N \bar{Y}_{n, :} \in \mathbb{R}^{1 \times D_{out}}$$

#### Weight Initialization (He / Kaiming Normal)
To preserve signal variance $\text{Var}(Y) = \text{Var}(X)$ across $D_{in}$ inputs with activation non-linearities:
$$\text{Var}(W) = \frac{2}{D_{in}} \implies W_{ij} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{D_{in}}}\right)$$

### ⚛️ Physics-Grounded Intuition
- **Resistive Networks & Linear Potentials**: A linear transformation acts as an electrical potential mapping $V_{\text{out}} = R \cdot I_{\text{in}}$, where weights $W$ represent conductances between node networks.
- **Momentum-Preserving Linear Maps**: Matrix multiplication $X W$ performs an affine coordinate frame transformation in multi-dimensional phase space.

---

## 4. Activation Functions

### 📐 Mathematical Derivations & Physics Intuition

#### 1. Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
- **Physics View**: Identical to the **Fermi-Dirac Distribution** $f(E) = \frac{1}{e^{(E - E_F)/k_B T} + 1}$ in quantum statistical mechanics, describing the thermal occupancy probability of fermions in energy states.

#### 2. Tanh Function
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$
- **Physics View**: Represents the mean magnetization $M = \tanh\left(\frac{J M + h}{k_B T}\right)$ in the mean-field solution of the **Ising Model** for ferromagnetism.

#### 3. GELU (Gaussian Error Linear Unit)
$$\text{GELU}(x) = x \Phi(x) = x P(X \le x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right)$$
- **Physics View**: Models **Quantum Tunneling / Stochastic Gating** where input values pass through a threshold perturbed by Gaussian zero-point quantum fluctuations.

#### 4. Softmax Function
$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i - \max_k z_k}}{\sum_{j} e^{z_j - \max_k z_k}}$$
$$\frac{\partial \text{Softmax}(\mathbf{z})_i}{\partial z_j} = \text{Softmax}_i (\delta_{ij} - \text{Softmax}_j)$$
- **Physics View**: Directly derived from the **Boltzmann-Gibbs Distribution** $P(s) = \frac{e^{-E(s)/k_B T}}{Z}$ in statistical thermodynamics, mapping microstate energy levels $z_i = -E(s_i)$ to canonical ensemble probabilities.

---

## 5. 2D Spatial Convolution & Pooling

### 📐 Mathematical Derivation
For 4D input $X \in \mathbb{R}^{N \times C_{in} \times H \times W}$ and kernel $K \in \mathbb{R}^{C_{out} \times C_{in} \times K_H \times K_W}$:
$$Y_{n, oc, ho, wo} = b_{oc} + \sum_{ic=0}^{C_{in}-1} \sum_{kh=0}^{K_H-1} \sum_{kw=0}^{K_W-1} X_{n, ic, ho \cdot S_H + kh, wo \cdot S_W + kw} \cdot K_{oc, ic, kh, kw}$$

Output spatial resolution:
$$H_{out} = \left\lfloor \frac{H + 2 P_H - K_H}{S_H} \right\rfloor + 1, \quad W_{out} = \left\lfloor \frac{W + 2 P_W - K_W}{S_W} \right\rfloor + 1$$

#### Gradient w.r.t Input & Kernel
$$\frac{\partial L}{\partial K_{oc, ic, kh, kw}} = \sum_{n, ho, wo} \bar{Y}_{n, oc, ho, wo} \cdot X_{n, ic, ho \cdot S_H + kh, wo \cdot S_W + kw}$$
$$\frac{\partial L}{\partial X_{n, ic, hi, wi}} = \sum_{oc, kh, kw} \bar{Y}_{n, oc, \frac{hi-kh}{S_H}, \frac{wi-kw}{S_W}} \cdot K_{oc, ic, kh, kw}$$

### ⚛️ Physics-Grounded Intuition
- **Heat Diffusion & Green's Functions**: Spatial convolution $u(x, t) = (f * G)(x)$ is the exact analytical solution to the **Heat Diffusion Equation** $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$, where a Gaussian kernel acts as the fundamental Green's function for thermal dissipation over time $t = \frac{\sigma^2}{2\alpha}$.
- **Renormalization Group & Coarse-Graining**: Max pooling and average pooling implement **Renormalization Group (RG)** block-spin transformations, coarse-graining microscopic lattice degrees of freedom while preserving macroscopic topological features.

---

## 6. Batch Normalization & Layer Normalization

### 📐 Mathematical Derivation
For a mini-batch of features $\mathcal{B} = \{x_1, \dots, x_N\}$:

1. **Batch Mean**: $\mu_{\mathcal{B}} = \frac{1}{N} \sum_{i=1}^N x_i$
2. **Batch Variance**: $\sigma_{\mathcal{B}}^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu_{\mathcal{B}})^2$
3. **Standardized Value**: $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$
4. **Scale & Shift Output**: $y_i = \gamma \hat{x}_i + \beta$

#### Running Statistics (Inference Tracking)
$$\mu_{\text{running}} \leftarrow (1 - \eta) \mu_{\text{running}} + \eta \mu_{\mathcal{B}}$$
$$\sigma^2_{\text{running}} \leftarrow (1 - \eta) \sigma^2_{\text{running}} + \eta \sigma^2_{\mathcal{B}}$$

### ⚛️ Physics-Grounded Intuition
- **Thermodynamic Standardization**: Batch Normalization shifts the center of mass velocity to zero ($\mu = 0$) and scales internal kinetic temperature to unit variance ($\sigma^2 = 1$). This prevents **Internal Covariate Shift**, ensuring energy fluctuations remain bounded during gradient flow.

---

## 7. Optimizers (SGD, Momentum, Adam, RMSProp)

### 📐 Mathematical Derivations & Physics Intuition

#### 1. Stochastic Gradient Descent (SGD) with Momentum
$$v_t = \gamma v_{t-1} + g_t + \lambda \theta_{t-1}, \quad \theta_t = \theta_{t-1} - \eta v_t$$
- **Physics View**: Represents the **Damped Heavy-Ball Particle Dynamics** in classical mechanics:
  $$m \frac{d^2 \theta}{dt^2} + b \frac{d\theta}{dt} = -\nabla U(\theta)$$
  where $\gamma = 1 - \frac{b}{m} \Delta t$ is the velocity memory coefficient and $-\nabla U(\theta) = -g_t$ is the conservative gravitational force.

#### 2. Adam (Adaptive Moment Estimation)
First and second uncentered moment estimates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Unbiased moment correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update rule:
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- **Physics View**: Models a **Particle in a Viscous Medium with Variable Adaptive Friction Coefficient** $\gamma(\theta) \propto \sqrt{\langle g^2 \rangle}$. In steep directions ($\text{large } g^2$), friction increases to prevent numerical overshoot; in flat valleys ($\text{small } g^2$), friction decreases to accelerate convergence.

---

## 8. Loss Functions & Information Theory

### 📐 Mathematical Derivation

#### 1. Mean Squared Error (MSE Loss)
$$L_{\text{MSE}}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

#### 2. Cross-Entropy Loss
$$L_{\text{CE}}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i, c} \log(\text{Softmax}(\hat{y})_{i, c})$$

### ⚛️ Physics-Grounded Intuition
- **Harmonic Oscillator Energy**: MSE loss equals the **Elastic Potential Energy** $U = \frac{1}{2} k (\Delta x)^2$ of a system of coupled harmonic springs.
- **Kullback-Leibler (KL) Divergence & Entropy Production**: Cross-Entropy loss minimizes the information-theoretic **KL Divergence** $D_{KL}(P \parallel Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$, which represents the non-reversible **Entropy Production Rate** $\frac{d S_{\text{universe}}}{dt} \ge 0$ in non-equilibrium thermodynamics.

---

## 9. Transformer Attention & Positional Encoding

### 📐 Mathematical Derivation
Given input sequence $X \in \mathbb{R}^{T \times d_{\text{model}}}$:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}})$$

#### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ preserves unit variance of inner products $\mathbb{E}[q \cdot k] = 0, \text{Var}(q \cdot k) = d_k$, preventing Softmax gradients from vanishing for high dimensions $d_k$.

#### Sinusoidal Positional Encoding
$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

### ⚛️ Physics-Grounded Intuition
- **$N$-Body Field Interactions**: Scaled Dot-Product Attention models **Pairwise Gravitational/Electrostatic Force Interaction Fields**:
  $$F_{ij} \propto \frac{m_i m_j}{r_{ij}^2} \iff A_{ij} = \text{Softmax}\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}\right)$$
  Queries act as test charges, Keys act as source field charges, and Values act as vector fields transported across spacetime.
- **Fourier Wave Function Modes**: Sinusoidal positional encodings represent **Harmonic Standing Wave Functions** $e^{i \omega_k t}$ spanning orthogonal frequency modes $\omega_k$, encoding relative spatial phase shifts continuously.
