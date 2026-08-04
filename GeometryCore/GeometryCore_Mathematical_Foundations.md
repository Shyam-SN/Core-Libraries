# GeometryCore: Mathematical & Physics-Grounded Foundations Handbook

This handbook provides graduate-level first-principles derivations and physical grounding for every core component in `GeometryCore`.

---

## 1. 3D Linear Algebra & Rigid Transformations

### 📐 Mathematical Derivation
For 3D vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^3$:

#### Cross Product & Skew-Symmetric Operator
The cross product $\mathbf{u} \times \mathbf{v}$ is equivalent to matrix multiplication using the skew-symmetric matrix $[\mathbf{u}]_\times \in \mathfrak{so}(3)$:
$$[\mathbf{u}]_\times = \begin{bmatrix} 0 & -u_z & u_y \\ u_z & 0 & -u_x \\ -u_y & u_x & 0 \end{bmatrix} \implies \mathbf{u} \times \mathbf{v} = [\mathbf{u}]_\times \mathbf{v}$$

Properties of the skew-symmetric operator:
$$[\mathbf{u}]_\times^T = -[\mathbf{u}]_\times, \quad [\mathbf{u}]_\times \mathbf{u} = \mathbf{0}, \quad [\mathbf{u}]_\times^3 = -\|\mathbf{u}\|^2 [\mathbf{u}]_\times$$

#### Rotation Matrix Properties
A 3D rotation matrix $R \in SO(3)$ (Special Orthogonal Group) satisfies:
$$R^T R = R R^T = I_{3 \times 3}, \quad \det(R) = +1$$

#### Hestenes-Jacobi Singular Value Decomposition (SVD)
For matrix $A \in \mathbb{R}^{3 \times 3}$, SVD decomposes $A = U \Sigma V^T$, where $U, V \in SO(3)$ and $\Sigma = \text{diag}(\sigma_1, \sigma_2, \sigma_3)$.
Jacobi rotations iteratively apply plane rotation matrices $J(p, q, \theta)$ to annihilate off-diagonal elements $A_{pq}$:
$$\tan(2\theta) = \frac{2 A_{pq}}{A_{qq} - A_{pp}}$$

### ⚛️ Physics-Grounded Intuition
- **Angular Momentum, Torques & Velocity Fields**:
  - The skew-symmetric matrix $[\mathbf{r}]_\times$ maps linear momentum $\mathbf{p}$ to orbital angular momentum $\mathbf{L} = \mathbf{r} \times \mathbf{p} = [\mathbf{r}]_\times \mathbf{p}$, and force $\mathbf{F}$ to torque $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F} = [\mathbf{r}]_\times \mathbf{F}$.
  - For a rigid rotating body with angular velocity $\boldsymbol{\omega}$, the velocity of any point at position $\mathbf{r}$ is $\mathbf{v} = \boldsymbol{\omega} \times \mathbf{r} = [\boldsymbol{\omega}]_\times \mathbf{r}$.
- **Kinetic Energy & Frame Orthogonality**:
  - Rotation matrix $R \in SO(3)$ preserves kinetic energy $T = \frac{1}{2} m \|\mathbf{v}\|^2 = \frac{1}{2} m \|R \mathbf{v}\|^2$ and rigid inter-particle distance $\|\mathbf{r}_i - \mathbf{r}_j\|$, maintaining frame invariance under Galilean reference frame changes.
- **Deformation Gradients & Principal Strain Axes**:
  - Applying SVD $F = U \Sigma V^T$ to continuum deformation gradients $F = \frac{\partial \mathbf{x}}{\partial \mathbf{X}}$ separates spatial deformation into initial principal orientation $V^T$, pure volumetric/shear stretching $\Sigma$ along principal axes, and final rigid body rotation $U$, matching the Polar Decomposition Theorem $F = R U_{\text{stretch}}$ in continuum mechanics.

---

## 2. Quaternions & Spherical Interpolation

### 📐 Mathematical Derivation
A quaternion $q \in \mathbb{H}$ is defined as:
$$q = w + x i + y j + z k = (w, \mathbf{v}), \quad i^2 = j^2 = k^2 = ijk = -1$$

#### Hamilton Product
Given $q_1 = (w_1, \mathbf{v}_1)$ and $q_2 = (w_2, \mathbf{v}_2)$:
$$q_1 \otimes q_2 = (w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2, \; w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2)$$

#### Point Rotation Action
A 3D vector $\mathbf{p} \in \mathbb{R}^3$ embedded as pure quaternion $p = (0, \mathbf{p})$ is rotated by unit quaternion $q = (\cos\frac{\theta}{2}, \mathbf{u} \sin\frac{\theta}{2})$ via:
$$p' = q \otimes p \otimes q^*$$

#### Spherical Linear Interpolation (SLERP)
Interpolating between unit quaternions $q_1$ and $q_2$ along the geodesic on 3-sphere $S^3$:
$$\text{SLERP}(q_1, q_2; t) = \frac{\sin((1-t)\theta)}{\sin\theta} q_1 + \frac{\sin(t\theta)}{\sin\theta} q_2, \quad \theta = \arccos(q_1 \cdot q_2)$$

### ⚛️ Physics-Grounded Intuition
- **Spinors, $SU(2)$ Symmetry & Quantum Mechanics**:
  - Unit quaternions are isomorphic to the special unitary group $SU(2)$ via Pauli spin matrices $\sigma_x, \sigma_y, \sigma_z$:
    $$q = w I - i (x \sigma_x + y \sigma_y + z \sigma_z)$$
  - Rotating a physical system by angle $\theta$ introduces a quantum phase factor $e^{-i \theta / 2}$. A full $360^\circ$ ($2\pi$) rotation multiplies the state by $-1$; a $720^\circ$ ($4\pi$) rotation is required to restore the original quantum state vector, demonstrating the double-covering $SU(2) \to SO(3)$.
- **Torque-Free Precession & Minimum Energy Geodesics**:
  - SLERP traverses the minimum distance geodesic path on the Riemannian manifold $S^3$ with constant angular velocity $\|\boldsymbol{\omega}\| = \text{const}$. This mirrors the natural torque-free precession trajectory of a symmetric top governed by Euler's rotational equations of motion:
    $$\mathbf{I} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega}) = \mathbf{0}$$

---

## 3. Lie Groups $SO(3)$ & $SE(3)$

### 📐 Mathematical Derivation

#### $SO(3)$ Exponential Map (Rodrigues' Formula)
For Lie algebra vector $\boldsymbol{\omega} = \theta \mathbf{u} \in \mathfrak{so}(3)$ where $\|\mathbf{u}\| = 1$:
$$\exp(\boldsymbol{\omega}^\wedge) = I + \frac{\sin\theta}{\theta} [\boldsymbol{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times^2$$

#### $SO(3)$ Logarithmic Map
Given $R \in SO(3)$:
$$\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right), \quad \boldsymbol{\omega}^\wedge = \frac{\theta}{2 \sin\theta} (R - R^T)$$

#### $SE(3)$ Exponential Map
For 6D twist vector $\boldsymbol{\xi} = (\boldsymbol{\rho}, \boldsymbol{\omega})^T \in \mathfrak{se}(3)$:
$$\exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} R & V \boldsymbol{\rho} \\ \mathbf{0}^T & 1 \end{bmatrix}$$
where $V = I + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times + \frac{\theta - \sin\theta}{\theta^3} [\boldsymbol{\omega}]_\times^2$.

### ⚛️ Physics-Grounded Intuition
- **Infinitesimal Generators & Hamiltonian State Trajectories**:
  - The Lie algebra element $\boldsymbol{\omega}^\wedge \in \mathfrak{so}(3)$ is the **infinitesimal generator** of rotation. The matrix exponential $\exp(t \boldsymbol{\omega}^\wedge)$ represents continuous time integration of rigid rotation along one-parameter subgroup trajectories $\dot{\mathbf{r}}(t) = \boldsymbol{\omega} \times \mathbf{r}(t)$, satisfying Hamilton's canonical equations of motion.
- **Chasles' Theorem & Screw Motion Kinematics**:
  - The Lie group $SE(3)$ captures the full Euclidean motion of rigid bodies. According to **Chasles' Theorem** in classical mechanics, any arbitrary rigid body displacement is physically equivalent to a **screw motion**: a simultaneous rotation around an axis and translation along that same axis dictated by the 6D twist $\boldsymbol{\xi} = (\boldsymbol{\rho}, \boldsymbol{\omega})^T \in \mathfrak{se}(3)$.

---

## 4. Pinhole Camera Geometry & Lens Distortion

### 📐 Mathematical Derivation
For 3D point $\mathbf{P}_{\text{cam}} = (X, Y, Z)^T$ in camera frame:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \mathbf{P}_{\text{cam}} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}$$

#### Radial & Tangential Distortion Model
Normalized coordinates $(x, y) = (X/Z, Y/Z)$:
$$r^2 = x^2 + y^2$$
$$x_{\text{dist}} = x (1 + k_1 r^2 + k_2 r^4) + 2 p_1 x y + p_2 (r^2 + 2 x^2)$$
$$y_{\text{dist}} = y (1 + k_1 r^2 + k_2 r^4) + p_1 (r^2 + 2 y^2) + 2 p_2 x y$$

Pixel coordinates:
$$u = f_x x_{\text{dist}} + c_x, \quad v = f_y y_{\text{dist}} + c_y$$

### ⚛️ Physics-Grounded Intuition
- **Fermat's Principle of Least Time & Geometrical Optics**:
  - Light rays propagate along paths that minimize optical travel time $\delta \int n(\mathbf{r}) ds = 0$. Ideal pinhole projection represents central perspective projection where spherical electromagnetic wavefronts emitted from 3D points pass through an infinitely small aperture, forming an inverted real image on the focal plane at distance $z = f$.
- **Optical Wavefront Aberrations**:
  - **Radial Distortion** ($k_1, k_2$) models spherical aberration caused by non-paraxial refraction at the curved glass-air interface of thick lenses, where marginal light rays bend more sharply than axial rays.
  - **Tangential Distortion** ($p_1, p_2$) models physical optical alignment imperfections, where lens elements are slightly tilted or misaligned relative to the image sensor plane.

---

## 5. Multi-View Geometry & Epipolar Geometry

### 📐 Mathematical Derivation
For corresponding normalized image points $\mathbf{x}_1, \mathbf{x}_2$:
$$\mathbf{x}_2^T E \mathbf{x}_1 = 0, \quad \text{where } E = [\mathbf{t}]_\times R \in \mathbb{R}^{3 \times 3}$$

Fundamental Matrix $F$:
$$\mathbf{p}_2^T F \mathbf{p}_1 = 0, \quad F = K_2^{-T} E K_1^{-1}$$

#### Essential Matrix SVD Decomposition
Singular value decomposition $E = U \text{diag}(\sigma, \sigma, 0) V^T$:
$$R_1 = U W V^T, \quad R_2 = U W^T V^T, \quad \mathbf{t} = \pm U_{:, 2}, \quad W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### ⚛️ Physics-Grounded Intuition
- **Planar Ray Intersections & Motion Parallax**:
  - The epipolar constraint $\mathbf{x}_2^T E \mathbf{x}_1 = 0$ enforces that the baseline vector $\mathbf{t}$ connecting camera centers $\mathbf{C}_1, \mathbf{C}_2$ and the two optical rays $\mathbf{x}_1, \mathbf{x}_2$ are co-planar (coplanarity constraint $\mathbf{x}_2 \cdot (\mathbf{t} \times R \mathbf{x}_1) = 0$).
  - Triangulation solves for the 3D intersection of two propagating optical rays, minimizing the potential discrepancy between light cone trajectories in physical space.

---

## 6. Iterative Closest Point (ICP) & Kabsch-Umeyama Algorithm

### 📐 Mathematical Derivation
Given source points $\{p_i\}_{i=1}^N$ and target points $\{q_i\}_{i=1}^N$, find optimal $R \in SO(3)$ and $\mathbf{t} \in \mathbb{R}^3$:
$$\min_{R, \mathbf{t}} \sum_{i=1}^N \| q_i - (R p_i + \mathbf{t}) \|^2$$

1. **Centroids**: $\bar{p} = \frac{1}{N} \sum p_i, \quad \bar{q} = \frac{1}{N} \sum q_i$
2. **Cross-Covariance Matrix**: $H = \sum_{i=1}^N (p_i - \bar{p}) (q_i - \bar{q})^T$
3. **SVD Decomposition**: $H = U \Sigma V^T$
4. **Optimal Rotation**: $R = V U^T$. If $\det(R) < 0$, replace last column $V_{:, 2} \leftarrow -V_{:, 2}$.
5. **Optimal Translation**: $\mathbf{t} = \bar{q} - R \bar{p}$.

### ⚛️ Physics-Grounded Intuition
- **Elastic Potential Energy Minimization**:
  - ICP alignment is physically equivalent to minimizing the total elastic potential energy $U(R, \mathbf{t}) = \frac{1}{2} k \sum_i \|\mathbf{q}_i - (R \mathbf{p}_i + \mathbf{t})\|^2$ of coupled virtual Hookean springs connecting corresponding points.
- **Center-of-Mass Frame Separation & Moment of Inertia**:
  - Subtracting centroids $\bar{p}, \bar{q}$ separates net linear translation from rotational torque dynamics. The cross-covariance matrix $H$ acts as a spatial angular momentum correlation tensor; SVD aligns the principal axes of inertia of both point cloud mass distributions.

---

## 7. Spatial Data Structures & Volumetric Fusion (KD-Tree, Octree, TSDF)

### 📐 Mathematical Derivation

#### Truncated Signed Distance Function (TSDF)
For voxel grid coordinate $\mathbf{x} \in \mathbb{R}^3$:
$$d(\mathbf{x}) = \text{depth}(u, v) - z_{\text{cam}}(\mathbf{x})$$
$$\text{TSDF}(\mathbf{x}) = \max\left(-1, \min\left(1, \frac{d(\mathbf{x})}{\tau}\right)\right)$$

#### Running Weight Integration
$$D_{k}(\mathbf{x}) = \frac{W_{k-1}(\mathbf{x}) D_{k-1}(\mathbf{x}) + w_k \text{TSDF}_k(\mathbf{x})}{W_{k-1}(\mathbf{x}) + w_k}, \quad W_k(\mathbf{x}) = W_{k-1}(\mathbf{x}) + w_k$$

### ⚛️ Physics-Grounded Intuition
- **Astrophysical $N$-Body Multipole Hierarchies**:
  - Octrees and 3D KD-Trees partition space hierarchically, mirroring spatial acceleration structures used in $N$-body gravitational and electrostatic astrophysics (e.g., the Barnes-Hut algorithm). Nearby particles are processed individually while distant clusters are aggregated, reducing spatial search complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$.
- **Scalar Potential Fields & Recursive Bayesian Filtering**:
  - TSDF volumes represent 3D physical space as a continuous scalar potential energy field $V(\mathbf{r})$. The physical solid surface corresponds exactly to the zero-equipotential isosurface $V(\mathbf{r}) = 0$.
  - The weighted running average update is mathematically equivalent to a **3D Spatial Kalman Filter**, filtering out high-frequency Gaussian sensor measurement noise across temporal depth observations.

---

## 8. Marching Cubes Iso-Surface Polygonization

### 📐 Mathematical Derivation
For 8 voxel cube vertices with scalar values $v_0, \dots, v_7$, determine cube index:
$$\text{cube\_idx} = \sum_{i=0}^7 (v_i < \text{iso\_level}) \ll i$$

Linearly interpolate edge vertices along voxel edges where scalar values cross the iso-level:
$$\mathbf{P} = \mathbf{P}_1 + \frac{\text{iso\_level} - v_1}{v_2 - v_1} (\mathbf{P}_2 - \mathbf{P}_1)$$

Vertex normals computed from spatial central differences:
$$\mathbf{n}(\mathbf{x}) = -\frac{\nabla \text{TSDF}(\mathbf{x})}{\|\nabla \text{TSDF}(\mathbf{x})\|}$$

### ⚛️ Physics-Grounded Intuition
- **Thermodynamic Phase Boundary Interfaces**:
  - Marching Cubes extracts physical phase boundaries (such as liquid-solid interfaces in Cahn-Hilliard phase-field equations or electron probability density boundaries $|\psi(\mathbf{r})|^2 = c$ in quantum mechanics) where scalar potential fields transition across critical energy thresholds.
- **Conservative Force Fields & Surface Normals**:
  - The spatial gradient $\nabla \text{TSDF}(\mathbf{x})$ represents the directional derivative of the scalar field. In conservative physical fields, the force vector $\mathbf{F} = -\nabla V(\mathbf{r})$ acts perpendicular to equipotential surfaces. Thus, vertex normals computed from $-\nabla \text{TSDF}$ match the physical surface normal forces of solid boundaries.
