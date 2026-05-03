# VisionCore: Mathematical & Physics-Based Foundations of Computer Vision

This document provides a comprehensive, graduate-level reference for the mathematical and physics-based foundations of all algorithms implemented in **VisionCore**. It bridges the gap between discrete digital code, continuous mathematical formulations, and the underlying physical phenomena (optics, thermodynamics, fluid dynamics, electromagnetism, and classical mechanics) that govern image formation and geometry.

---

## Table of Contents
1. [Image Representation, Sensing & Sensor Physics](#1-image-representation-sensing-&-sensor-physics)
2. [2D Spatial Convolution & Heat Diffusion](#2-2d-spatial-convolution-&-heat-diffusion)
3. [Neighborhood Filtering & Scale-Space Physics](#3-neighborhood-filtering-&-scale-space-physics)
4. [Intensity Histograms & Information Entropy](#4-intensity-histograms-&-information-entropy)
5. [Image Thresholding & Phase Transitions](#5-image-thresholding-&-phase-transitions)
6. [Mathematical Morphology & Huygens' Wave Front Propagation](#6-mathematical-morphology-&-huygens-wave-front-propagation)
7. [Image Gradients, Laplacian & Electrostatics](#7-image-gradients-laplacian-&-electrostatics)
8. [Canny Edge Detection & Percolation Dynamics](#8-canny-edge-detection-&-percolation-dynamics)
9. [Keypoint & Corner Detection (Inertia & Continuum Stress)](#9-keypoint-&-corner-detection-inertia-&-continuum-stress)
10. [ORB Feature Descriptors (Center of Mass & Torque)](#10-orb-feature-descriptors-center-of-mass-&-torque)
11. [Feature Matching (Spin Glasses & Potential Wells)](#11-feature-matching-spin-glasses-&-potential-wells)
12. [Template Matching & Wave Resonance](#12-template-matching-&-wave-resonance)
13. [Lucas-Kanade Sparse Optical Flow & Fluid Kinematics](#13-lucas-kanade-sparse-optical-flow-&-fluid-kinematics)
14. [Planar Homography & Optical Wavefront Projections](#14-planar-homography-&-optical-wavefront-projections)
15. [Camera Calibration (Zhang's Method) & Absolute Conic Refraction](#15-camera-calibration-zhangs-method-&-absolute-conic-refraction)
16. [Epipolar Geometry & Rigid Body Coplanarity](#16-epipolar-geometry-&-rigid-body-coplanarity)
17. [Linear Triangulation & Photon Back-Propagation](#17-linear-triangulation-&-photon-back-propagation)
18. [Stereo Matching & Parallax Physics](#18-stereo-matching-&-parallax-physics)
19. [Perspective-n-Point (PnP) & Rigid Body Statics](#19-perspective-n-point-pnp-&-rigid-body-statics)
20. [RANSAC Robust Model Fitting & Maxwell-Boltzmann Filtering](#20-ransac-robust-model-fitting-&-maxwell-boltzmann-filtering)
21. [Bundle Adjustment & Spring-Mass-Damper Network Dynamics](#21-bundle-adjustment-&-spring-mass-damper-network-dynamics)

---

## 1. Image Representation, Sensing & Sensor Physics

A digital image $I$ is a discrete representation of a continuous spatial distribution of electromagnetic energy (irradiance) incident on a sensor plane.

### Mathematical Formulation
Let $L(\mathbf{X}, \mathbf{\omega}, \lambda)$ be the scene radiance (energy emitted/reflected per unit area, solid angle, and wavelength $\lambda$) at a 3D coordinate $\mathbf{X}$ along direction $\mathbf{\omega}$. A camera lens assembly focuses this light onto a 2D sensor plane $\Omega \subset \mathbb{R}^2$. The resulting spatial power distribution is the **irradiance** $E(\mathbf{x}, \lambda)$ (Watts per square meter):

$$E(\mathbf{x}, \lambda) = \int_{\Omega_{\text{aperture}}} L(\mathbf{X}(\mathbf{x}), \mathbf{\omega}, \lambda) \cos\theta \, d\mathbf{\omega}$$

where $\theta$ is the angle of incidence relative to the sensor normal.

To maximize cache locality, pixel values are stored in a flat buffer:

$$\text{Offset}(x, y, c) = y \cdot S_e + x \cdot C + c$$

Alignment on $A$-byte boundaries for AVX vectorization enforces:

$$S_b = S_e \cdot \text{sizeof}(T) = \left\lceil \frac{W \cdot C \cdot \text{sizeof}(T)}{A} \right\rceil \cdot A$$

### Physics-Based Explanation
Modern active-pixel sensors (CMOS/CCD) exploit the **photoelectric effect** in silicon.
1.  **Photon Absorption**: When an incident photon of wavelength $\lambda$ has energy $E = \frac{hc}{\lambda}$ exceeding the bandgap energy of silicon ($E_g \approx 1.12 \text{ eV}$), it is absorbed, generating a free electron-hole pair.
2.  **Charge Accumulation**: The photodiode is reverse-biased, creating a depletion region where the electric field separates the charge carriers. The accumulated charge $Q$ (Coulombs) over integration time $\Delta t$ is:
    $$Q = \int_{0}^{\Delta t} \int_{\lambda} \eta(\lambda) \cdot A_{\text{pixel}} \cdot \Phi(\mathbf{x}, \lambda) \, d\lambda \, dt$$
    where $\eta(\lambda)$ is the quantum efficiency, $A_{\text{pixel}}$ is the active photodiode area, and $\Phi(\mathbf{x}, \lambda)$ is the incident photon flux density.
3.  **Quantization**: The charge is converted to voltage $V = Q/C_{\text{pixel}}$, amplified, and processed by an Analog-to-Digital Converter (ADC) yielding a quantized digital number (DN) representing the pixel intensity:
    $$I(x, y) = \text{clamp}\left( \text{round}\left( G_{\text{ADC}} \cdot V \right), 0, 2^{B}-1 \right)$$
    where $G_{\text{ADC}}$ is the digitizer gain and $B$ is the bit depth (typically $B=8$ for standard grayscale images).

---

## 2. 2D Spatial Convolution & Heat Diffusion

The discrete 2D spatial convolution of an image $I(x, y, c)$ with a kernel $K \in \mathbb{R}^{K_w \times K_h}$ is:

$$(I * K)(x, y, c) = \sum_{j = -r_h}^{r_h} \sum_{i = -r_w}^{r_w} I(x - i, y - j, c) \cdot K(i + r_w, j + r_h)$$

where $r_w = \lfloor K_w / 2 \rfloor$ and $r_h = \lfloor K_h / 2 \rfloor$.

### Mathematical Formulation (Separability)
If a 2D filter kernel can be factored as $K = \mathbf{k}_v \mathbf{k}_h^T$ (where $\mathbf{k}_h \in \mathbb{R}^{K_w}$ and $\mathbf{k}_v \in \mathbb{R}^{K_h}$), the convolution is separable:

$$(I * K)(x, y, c) = \sum_{j = -r_h}^{r_h} \mathbf{k}_v(j + r_h) \left( \sum_{i = -r_w}^{r_w} I(x - i, y - j, c) \cdot \mathbf{k}_h(i + r_w) \right)$$

This reduces complexity from $\mathcal{O}(K_w \cdot K_h)$ to $\mathcal{O}(K_w + K_h)$ operations per pixel.

### Physics-Based Explanation
Discrete convolution represents a linear, shift-invariant system. In physics, this corresponds to the **superposition principle** of fields (e.g. gravitational, electrostatic, or wave fields). The kernel $K$ acts as the system's Green's function (impulse response), characterizing how a point source (Dirac delta impulse of light) diffuses or disperses through an optical medium or sensor pixel array.

---

## 3. Neighborhood Filtering & Scale-Space Physics

### Mathematical Formulation
The continuous isotropic Gaussian distribution is defined as:

$$G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

Since it is separable, we generate 1D kernels of length $K_w$:

$$\mathbf{k}(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{x^2}{2\sigma^2} \right), \quad x \in [-r_w, r_w]$$

which is normalized such that $\sum_{x} \mathbf{k}(x) = 1$.

### Physics-Based Explanation
Gaussian filtering is the exact physical solution to the **isotropic heat diffusion equation**.
Consider the image intensity $I(x, y)$ as a temperature distribution $u(x, y, t)$ in a homogeneous thermal conductor. The evolution of the temperature field over time $t$ is governed by the heat equation:

$$\frac{\partial u}{\partial t} = D \nabla^2 u = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

where $D$ is the thermal diffusivity coefficient.

Given the initial condition $u(x, y, 0) = I(x, y)$, the temperature distribution at time $t$ is solved by convolving the initial state with the Green's function (fundamental solution) of the heat equation:

$$u(x, y, t) = (I * G_t)(x, y)$$

where $G_t(x, y)$ is the 2D Gaussian kernel:

$$G_t(x, y) = \frac{1}{4\pi D t} \exp\left( -\frac{x^2 + y^2}{4 D t} \right)$$

By mapping the physical time parameter $t$ to the scale parameter $\sigma$ such that:

$$\sigma^2 = 2 D t$$

the Green's function becomes the standard Gaussian filter:

$$G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

Thus, smoothing an image with a Gaussian filter of scale $\sigma$ is physically equivalent to letting the image's "heat" diffuse over a time interval $t = \frac{\sigma^2}{2D}$.

---

## 4. Intensity Histograms & Information Entropy

### Mathematical Formulation
The intensity histogram $H(k)$ of a single-channel image $I$ measures the discrete probability distribution of intensity levels:

$$H(k) = \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} \delta(I(x, y) - k), \quad k \in [0, 255]$$

where $\delta$ is the Kronecker delta.

### Physics-Based Explanation
From the perspective of statistical mechanics and information theory, the information entropy $S$ of an image with intensity probability distribution $P(k) = \frac{H(k)}{W \cdot H}$ is:

$$S = - \sum_{k=0}^{255} P(k) \ln P(k)$$

By the second law of thermodynamics, isolated systems evolve toward maximum entropy (thermal equilibrium). In image processing, maximizing $S$ corresponds to spreading the intensity distribution as uniformly as possible over the available spectrum.
Let $y = T(k)$ be a continuous, strictly increasing transformation of intensity. We want the PDF of the transformed variable, $P_Y(y)$, to be uniform: $P_Y(y) = \frac{1}{255}$. By the conservation of probability mass:

$$P_Y(y) \, dy = P_X(k) \, dk \implies \frac{1}{255} \, dy = P_X(k) \, dk$$

Integrating both sides yields:

$$y = T(k) = 255 \int_{0}^{k} P_X(x) \, dx = 255 \cdot C(k)$$

where $C(k)$ is the Cumulative Distribution Function (CDF). In the discrete domain, we normalize and round to the nearest integer:

$$T(k) = \text{round}\left( 255 \cdot \frac{C(k) - C_{\min}}{1 - C_{\min}} \right)$$

---

## 5. Image Thresholding & Phase Transitions

Thresholding partitions an image into distinct binary states (foreground/background), which is analogous to a **first-order phase transition** in thermodynamic systems.

### Mathematical Formulation
Otsu's method splits the image intensity distribution into two classes, $C_0$ and $C_1$, separated by a threshold $T$. The total variance of the system is conserved:

$$\sigma_{\text{total}}^2 = \sigma_W^2(T) + \sigma_B^2(T)$$

where $\sigma_W^2(T)$ is the within-class variance (representing the internal thermal fluctuations of each phase) and $\sigma_B^2(T)$ is the between-class variance (representing the energy barrier separating the phases):

$$\sigma_W^2(T) = \omega_0(T)\sigma_0^2(T) + \omega_1(T)\sigma_1^2(T)$$

$$\sigma_B^2(T) = \omega_0(T)\omega_1(T)[\mu_0(T) - \mu_1(T)]^2$$

Maximizing the between-class variance $\sigma_B^2(T)$ is mathematically equivalent to minimizing the internal variance $\sigma_W^2(T)$. The thresholding operation acts like a sharp boundary separating the coexisting phases.

### Physics-Based Explanation
In statistical mechanics, a phase transition occurs when a system transitions from a homogeneous state (continuous intensities) into segregated phases (e.g. liquid-gas separation). The threshold $T$ acts as a **critical parameter**. Otsu's search for the optimal threshold is analogous to finding the coexistence curve where the Helmholtz free energy is minimized, resulting in a thermodynamically stable separation of the two phases.

---

## 6. Mathematical Morphology & Huygens' Wave Front Propagation

Morphological operations alter geometric shapes by applying structuring elements. Dilations and erosions can be modeled as **wave front propagation** governed by the **Eikonal equation** in wave physics.

### Mathematical Formulation
Let $A$ represent the image view, and $B$ represent the structuring element (kernel) with a local offset footprint $\mathcal{F}_B$.

*   **Dilation**: Computes the local supremum:
    $$(A \oplus B)(x, y, c) = \max_{(dx, dy) \in \mathcal{F}_B} A(x + dx, y + dy, c)$$
*   **Erosion**: Computes the local infimum:
    $$(A \ominus B)(x, y, c) = \min_{(dx, dy) \in \mathcal{F}_B} A(x + dx, y + dy, c)$$
*   **Opening**: Erosion followed by dilation:
    $$A \circ B = (A \ominus B) \oplus B$$
*   **Closing**: Dilation followed by erosion:
    $$A \bullet B = (A \oplus B) \ominus B$$

### Physics-Based Explanation
According to **Huygens' principle**, every point on a wave front acts as a source of secondary spherical wavelets. The envelope of these wavelets forms the new wave front at a later time.
Let $S_0 \subset \mathbb{R}^2$ be a set representing the initial binary image boundary. If we propagate this boundary outwards at a constant unit speed $v = 1$, the arrival time $T(x, y)$ of the wave front at any point in space satisfies the Eikonal equation:

$$\|\nabla T(x, y)\| = \frac{1}{v} = 1, \quad T(x, y) = 0 \text{ on } \partial S_0$$

The set of points reached by the wave front at time $t = R$ is:

$$S_t = \{ (x, y) \mid T(x, y) \le R \}$$

This set $S_t$ is exactly equivalent to the morphological dilation $S_0 \oplus B_R$, where $B_R$ is a disk structuring element of radius $R$. Dilation dilutes boundary curvature by expansion, while erosion retracts the boundary, mimicking physical wave reflections and diffraction around obstacles.

---

## 7. Image Gradients, Laplacian & Electrostatics

Image gradients estimate spatial variations of intensity, which share identical mathematics with **electrostatic field theory**.

### Mathematical Formulation
The gradient vector $\nabla I(x, y)$ and Laplacian $\nabla^2 I(x, y)$ are defined as:

$$\nabla I(x, y) = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix}, \quad \nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

In the discrete domain, these are implemented via Sobel/Prewitt/Scharr convolution filters.

### Physics-Based Explanation
Let image intensity $I(x, y)$ act as a continuous electrostatic potential $V(x, y)$. The image gradient vector $\nabla I$ is analogous to the physical electric field vector $\mathbf{E}$:

$$\mathbf{E}(x, y) = - \nabla V(x, y) \sim \nabla I(x, y)$$

The gradient direction points along the path of maximum potential increase (maximum intensity change), and its magnitude measures the field strength.
The Laplacian operator $\nabla^2 I$ is the divergence of the gradient. In electrostatics, the potential satisfies **Poisson's equation**:

$$\nabla^2 V = - \frac{\rho}{\epsilon_0}$$

where $\rho(x, y)$ is the local free charge density and $\epsilon_0$ is the vacuum permittivity.
A positive Laplacian value corresponds to a local minimum (a sink, or negative charge accumulation), while a negative Laplacian value corresponds to a local maximum (a source, or positive charge accumulation). Thus, edge detection via Laplacian zero-crossings corresponds to finding regions of zero net electric charge density.

---

## 8. Canny Edge Detection & Percolation Dynamics

Canny edge detection isolates sharp discontinuities in the image potential field.

### Mathematical Formulation
1.  **Gradient Estimation**: Compute gradient magnitude $M(x, y)$ and direction $\theta(x, y)$:
    $$M(x, y) = \sqrt{I_x^2 + I_y^2}, \quad \theta(x, y) = \text{atan2}(I_y, I_x)$$
2.  **Non-Maximum Suppression**: Discretize $\theta(x,y)$ to $0^\circ, 45^\circ, 90^\circ,$ or $135^\circ$ and suppress $M(x,y)$ to 0 if it is not a local maximum along the discretized direction.
3.  **Double Thresholding**: Label pixels as Strong, Weak, or Suppressed using high ($T_H$) and low ($T_L$) thresholds.

### Physics-Based Explanation
Hysteresis edge tracking behaves like a physical **percolation process** or **phase transition in a lattice network**.
Consider the weak edge pixels as a lattice grid. Strong edge pixels represent regions that are already "wetted" or "conductive". A weak edge pixel becomes conductive if it connects to a strong pixel.
This is identical to the physics of fluid percolation through a porous medium:
*   If the threshold $T_L$ is too low, the system is above the critical percolation threshold $p_c$, causing noise to propagate indefinitely (creating a single giant, chaotic cluster).
*   If $T_L$ is too high, the system is sub-critical, and edges fail to form continuous structures.
Hysteresis tracking balances this phase transition, allowing the physical boundaries of object silhouettes to percolate while suppressing isolated thermal noise fluctuations.

---

## 9. Keypoint & Corner Detection (Inertia & Continuum Stress)

### Mathematical Formulation
The local structure tensor $M$ (second moment matrix) is computed as:

$$M(x, y) = \sum_{(dx, dy) \in W} w(dx, dy) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

The corner response score $R$ is:

$$R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

where $\lambda_1, \lambda_2$ are eigenvalues of $M$.

### Physics-Based Explanation
The structure tensor $M$ is mathematically isomorphic to the **Moment of Inertia Tensor** in classical rigid body mechanics, and the **Stress Tensor** in continuum mechanics.
1.  **Rotational Inertia Analogy**: Treat the squared gradients $I_x^2, I_y^2$ as virtual mass distribution density.
    *   In flat regions (zero gradients), the neighborhood has no virtual mass, meaning no resistance to rotation (both eigenvalues are zero).
    *   In edge regions, the virtual mass is distributed along a single straight line. Like an infinitely thin rod, the moment of inertia is extremely high about one principal axis but zero about the other. The body can rotate freely along the edge direction without resistance.
    *   In corner regions, mass is distributed in all directions. The neighborhood behaves like a balanced, heavy solid disk. It has high resistance to rotation (high moment of inertia) about both principal axes ($\lambda_1$ and $\lambda_2$ are both large).
2.  **Continuum Stress Analogy**: The structure tensor represents the distribution of directional forces. A corner represents a point of triaxial/biaxial stress concentration, where the principal stresses are balanced, preventing shear failure under multi-directional deformation.

---

## 10. ORB Feature Descriptors (Center of Mass & Torque)

ORB combines oriented FAST keypoints and rotation-invariant BRIEF descriptors.

### Mathematical Formulation
The intensity moments $m_{pq}$ of a local circular patch are:

$$m_{pq} = \sum_{x, y \in \text{Patch}} x^p y^q I(x, y)$$

The coordinates of the intensity centroid $C = (x_c, y_c)$ are:

$$x_c = \frac{m_{10}}{m_{00}}, \quad y_c = \frac{m_{01}}{m_{00}}$$

The orientation angle $\theta$ is:

$$\theta = \text{atan2}(y_c, x_c) = \text{atan2}(m_{01}, m_{10})$$

### Physics-Based Explanation
We treat the pixel intensities $I(x, y)$ as a physical mass density distribution $\rho(x, y)$ over a thin 2D plate.
*   $m_{00}$ is the total physical mass of the plate.
*   $(x_c, y_c)$ is the Center of Mass (Centroid) of the plate.
If we pin the plate at the origin $(0,0)$ and apply a uniform gravitational field $g$ pointing in some direction, the gravity forces acting on each pixel create a net torque:

$$\mathbf{\tau} = \mathbf{r}_c \times \mathbf{F}_g$$

where $\mathbf{r}_c = (x_c, y_c)^T$ is the center of mass vector.
The orientation angle $\theta$ represents the direction of the principal axis of rotational equilibrium. Rotating the patch coordinate system by $\theta$ aligns the frame to this equilibrium axis, ensuring that the subsequent binary descriptor extraction occurs in a co-rotating coordinate frame independent of the camera roll.

---

## 11. Feature Matching (Spin Glasses & Potential Wells)

Feature matching pairs descriptors across views.

### Mathematical Formulation
*   **Hamming Distance**: For 256-bit binary descriptors $\mathbf{a}, \mathbf{b} \in \{0, 1\}^{256}$:
    $$d_{\text{Hamming}}(\mathbf{a}, \mathbf{b}) = \sum_{k=0}^{D-1} \text{popcount}(a_k \oplus b_k)$$
*   **Lowe's Ratio Test**: Filters out ambiguous matches by checking:
    $$\frac{d(\mathbf{f}_a, \mathbf{f}_{b1})}{d(\mathbf{f}_a, \mathbf{f}_{b2})} < \tau \quad (\text{typically } \tau \approx 0.8)$$

### Physics-Based Explanation
1.  **Spin Glass Analogy**: We can model the binary descriptor bits as a 1D chain of magnetic spin states (Up/Down) in a **spin glass model** from condensed matter physics. The Hamming distance represents the total magnetization/energy difference required to flip the spins of system $A$ to match the configuration of system $B$:
    $$\Delta E \sim d_{\text{Hamming}}(\mathbf{a}, \mathbf{b})$$
    A match represents finding the global minimum energy state of this coupled spin system.
2.  **Potential Well Analogy**: Lowe's ratio test behaves like a particle in a potential energy landscape.
    *   The best match is the primary local potential well of depth $d_1$.
    *   The second-best match is an adjacent potential well of depth $d_2$.
    For a match to be physically stable against perturbations (such as sensor noise or lighting changes), the particle must be trapped in a deep, isolated potential well. If $d_1 / d_2 \approx 1.0$, the barrier separating the two wells is small, and thermal fluctuations can cause the particle to escape (or tunnel) into the wrong state. A ratio test threshold of $\approx 0.8$ ensures the correct match is trapped in a deep, stable potential well.

---

## 12. Template Matching & Wave Resonance

Template matching finds target patterns by sliding a window over an image.

### Mathematical Formulation
The Normalized Correlation Coefficient (CCOEFF_NORMED) is:

$$R_{\text{CCOEFF\_NORMED}}(x, y) = \frac{\sum_{u, v} [I(x+u, y+v) - \bar{I}][T(u, v) - \bar{T}]}{\sqrt{\sum [I(x+u, y+v) - \bar{I}]^2 \sum [T(u, v) - \bar{T}]^2}}$$

### Physics-Based Explanation
Template matching is physically equivalent to **wave interference and resonance**.
Consider the template $T$ and the image sub-window $I$ as 2D spatial waveforms. When the waveforms are out of phase or structurally different, they undergo destructive interference, minimizing the correlation score.
When the spatial frequency and structure of $I$ match $T$, the waveforms interfere constructively. The CCOEFF_NORMED formula is mathematically identical to the cosine of the phase angle between two vectors in a Hilbert space. Achieving a maximum correlation value of $1.0$ corresponds to physical **resonance**, where the driving wave (the template) perfectly matches the natural frequency and spatial phase of the medium (the image).

---

## 13. Lucas-Kanade Sparse Optical Flow & Fluid Kinematics

Optical flow computes the 2D velocity field $(u, v)^T$ between consecutive frames.

### Mathematical Formulation
Under the assumption of brightness constancy, the material derivative of intensity is zero:

$$\frac{DI}{Dt} = \frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} + \frac{\partial I}{\partial t} = 0 \implies I_x u + I_y v + I_t = 0$$

Lucas-Kanade solves this overdetermined system inside a local window $W$:

$$\sum_{\mathbf{p} \in W} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} -I_x I_t \\ -I_y I_t \end{bmatrix} \implies G \mathbf{u} = \mathbf{b}$$

### Physics-Based Explanation
The Brightness Constancy Constraint Equation is derived from the **continuity equation** for a conserved scalar property in fluid dynamics.
Let a scalar field $\phi(\mathbf{x}, t)$ represent the density of a fluid. The conservation of mass dictates:

$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\phi \mathbf{u}) = 0 \implies \frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi + \phi (\nabla \cdot \mathbf{u}) = 0$$

Assuming an incompressible fluid ($\nabla \cdot \mathbf{u} = 0$), this simplifies to:

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0$$

which is the material derivative $\frac{D\phi}{Dt} = 0$.
In computer vision, we treat the image intensity $I(x, y, t)$ as a conserved fluid density. The optical flow equation assumes that light intensity is advected by the physical velocity field $\mathbf{u}$ without any source, sink, or diffusion.

---

## 14. Planar Homography & Optical Wavefront Projections

A homography matrix $H \in \mathbb{R}^{3 \times 3}$ defines a projective mapping between two coplanar views: $\mathbf{x}' \sim H \mathbf{x}$.

### Mathematical Formulation
Using the cross product projection relation $\mathbf{x}'_i \times H \mathbf{x}_i = \mathbf{0}$, we obtain:

$$\begin{bmatrix} \mathbf{0}^T & -w_i' \mathbf{x}_i^T & y_i' \mathbf{x}_i^T \\ w_i' \mathbf{x}_i^T & \mathbf{0}^T & -x_i' \mathbf{x}_i^T \end{bmatrix} \mathbf{h} = \mathbf{0}$$

which is solved via SVD on normalized coordinates.

### Physics-Based Explanation
In wave optics, a homography models the transformation of an electromagnetic wavefront as it propagates between two planes.
Consider light radiating from a planar surface. The rays pass through a lens (modeled as a projection center). A homography describes how the planar phase fronts are mapped, scaled, and sheared on the destination sensor plane. This is equivalent to mapping the boundary conditions of the electromagnetic wave equation from one planar interface to another.

---

## 15. Camera Calibration (Zhang's Method) & Absolute Conic Refraction

Zhang's method estimates the camera intrinsic matrix $K$ using planar homographies.

### Mathematical Formulation
Let $H = [\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_3]$ be a homography. Since the rotation columns are orthonormal:

$$\mathbf{h}_1^T K^{-T} K^{-1} \mathbf{h}_2 = 0, \quad \mathbf{h}_1^T K^{-T} K^{-1} \mathbf{h}_1 = \mathbf{h}_2^T K^{-T} K^{-1} \mathbf{h}_2$$

This defines a system of equations solved for the absolute conic matrix $B = K^{-T} K^{-1}$.

### Physics-Based Explanation
In physical optics, the absolute conic is a circle at infinity consisting of complex points. It represents the wavefront of an isotropic spherical wave expanding from a point source.
An ideal lens focuses this spherical wavefront into a spherical converging wavefront. However, physical glass elements introduce refractive distortions. The intrinsic matrix $K$ represents how the camera sensor deforms this spherical wavefront into an ellipsoidal wavefront due to focal length differences ($f_x \neq f_y$), skew ($s$), and optical center offsets ($u_0, v_0$). Solving for $B = K^{-T} K^{-1}$ corresponds to measuring the anisotropic deformation of this expanding wavefront to calibrate the optical focal plane.

Furthermore, radial and tangential distortions model the physical deviations of light rays governed by **Snell's Law** ($n_1 \sin\theta_1 = n_2 \sin\theta_2$) at the curved glass interfaces of non-ideal lens assemblies:
*   **Radial Distortion** represents barrel or pincushion distortions caused by variations in magnification as rays refract through the outer edges of spherical lens surfaces.
*   **Tangential Distortion** represents asymmetric refraction caused by physical tilt or decentering of the lens elements relative to the sensor plane.

---

## 16. Epipolar Geometry & Rigid Body Coplanarity

Epipolar geometry models the relationship between two views.

### Mathematical Formulation
Corresponding points satisfy the epipolar constraint:

$$\mathbf{x}'^T F \mathbf{x} = 0$$

where $F = K'^{-T} E K^{-1}$ and $E = [\mathbf{t}]_{\times} R$.

### Physics-Based Explanation
The epipolar constraint is a physical **coplanarity condition**.
Let $C_1$ and $C_2$ be the physical optical centers of the two cameras. Let $\mathbf{X}$ be a 3D scene point. The three points $\{C_1, C_2, \mathbf{X}\}$ define a physical plane in 3D space: the **epipolar plane**.

```
             X (3D Point)
            / \
           /   \
          /     \
      [C1]-------[C2]  (Camera Baseline Vector t)
```

The vector from $C_1$ to the projected point $\mathbf{x}$ lies in this plane. The baseline vector $\mathbf{t}$ connecting $C_1$ and $C_2$ also lies in this plane. The cross product $\mathbf{t} \times \mathbf{x}$ produces a vector normal to this epipolar plane.
For the projected ray $\mathbf{x}'$ in camera 2 to lie in the same plane, its direction must be orthogonal to this normal vector. This is expressed by the triple product:

$$\mathbf{x}' \cdot (\mathbf{t} \times R \mathbf{x}) = 0 \implies \mathbf{x}'^T [\mathbf{t}]_{\times} R \mathbf{x} = 0 \implies \mathbf{x}'^T E \mathbf{x} = 0$$

In classical mechanics, if we apply a force along the direction $R\mathbf{x}$ at a distance $\mathbf{t}$ from the origin $C_2$, the resulting torque is $\mathbf{\tau} = \mathbf{t} \times R\mathbf{x}$. The epipolar constraint means that the projection of this torque vector along the direction of the second ray $\mathbf{x}'$ must be zero, ensuring the physical consistency of the ray intersections.

---

## 17. Linear Triangulation & Photon Back-Propagation

Triangulation reconstructs a 3D point $\mathbf{X}$ from its projections $\mathbf{x}_1, \mathbf{x}_2$.

### Mathematical Formulation
Using projection matrices $P_1, P_2$, we solve $A \mathbf{X} = \mathbf{0}$:

$$A = \begin{bmatrix} u_1 \mathbf{p}_{1,3}^T - \mathbf{p}_{1,1}^T \\ v_1 \mathbf{p}_{1,3}^T - \mathbf{p}_{1,2}^T \\ u_2 \mathbf{p}_{2,3}^T - \mathbf{p}_{2,1}^T \\ v_2 \mathbf{p}_{2,3}^T - \mathbf{p}_{2,2}^T \end{bmatrix}$$

using SVD.

### Physics-Based Explanation
Triangulation models the physical **back-propagation of light rays** (photons) into the scene.
A pixel detection at $(u_1, v_1)$ indicates that a photon traveled along a specific straight ray in space before hitting the sensor. Back-projecting this pixel yields a physical line of propagation.
Doing this for both cameras yields two lines in 3D space. Due to measurement noise and spatial quantization, these two lines will not intersect perfectly, forming a pair of skew lines.
The linear SVD solution finds the point of closest approach of these two rays. This represents the point of maximum probability density of photon emission (the physical source location).

---

## 18. Stereo Matching & Parallax Physics

Stereo vision determines depth from two horizontally shifted views.

### Mathematical Formulation
For baseline $B$ and focal length $f$:

$$Z = \frac{f \cdot B}{x_L - x_R} = \frac{f \cdot B}{d}$$

### Physics-Based Explanation
Stereo matching relies on the physics of **parallax**.
Parallax is the difference in the apparent position of an object viewed along two different lines of sight.
As a camera moves horizontally by distance $B$, the apparent shift of a scene point on the sensor (disparity $d$) is inversely proportional to its physical distance $Z$. Nearby objects require a large angular shift of the eyes or camera to keep them in focus (large parallax/disparity), while distant objects (like stars) exhibit negligible parallax ($d \approx 0 \implies Z \to \infty$).

---

## 19. Perspective-n-Point (PnP) & Rigid Body Statics

PnP estimates camera pose $[R \mid \mathbf{t}]$ from $N$ 3D-to-2D correspondences.

### Mathematical Formulation
The projection relation is $\mathbf{x}_i \sim P \mathbf{X}_i$ where $P = K [R \mid \mathbf{t}]$. DLT solves for $P$, and RQ decomposition extracts $R$ and $\mathbf{t}$. Orthonormalization is performed via SVD:

$$R = U V^T \quad \text{where } \tilde{R} = U \Sigma V^T$$

### Physics-Based Explanation
PnP can be physically interpreted as finding the position and orientation of a rigid camera body suspended in space by $N$ tension wires (projection rays) anchored to known 3D points $\mathbf{X}_i$.
Each anchor point $\mathbf{X}_i$ pulls the camera body. The tension force $\mathbf{F}_i$ acts along the projection ray connecting the anchor to the camera center:

$$\mathbf{F}_i = k \cdot (\mathbf{x}_i - \pi(P, \mathbf{X}_i))$$

The camera pose $[R \mid \mathbf{t}]$ is the state of **static equilibrium** where the net forces and net torques exerted by these tension wires sum to zero:

$$\sum_{i=1}^N \mathbf{F}_i = \mathbf{0}, \quad \sum_{i=1}^N \mathbf{r}_i \times \mathbf{F}_i = \mathbf{0}$$

Projecting the estimated rotation matrix to $SO(3)$ via SVD enforces the physical rigidity constraint of the camera body, preventing it from stretching or shearing under tension.

---

## 20. RANSAC Robust Model Fitting & Maxwell-Boltzmann Filtering

RANSAC robustly fits models (like homographies) by iteratively sampling subsets of data.

### Mathematical Formulation
The number of required iterations $N$ is:

$$N = \frac{\ln(1 - p)}{\ln(1 - (1 - e)^s)}$$

where $e$ is the outlier ratio, $s$ is the sample size, and $p$ is the confidence.

### Physics-Based Explanation
RANSAC acts like a thermodynamic filter that separates the system into a low-entropy "ordered" phase (inliers) and a high-entropy "thermal noise" phase (outliers).
Consider a dataset containing matching points. The true matches belong to a single physical model (low entropy), while the outliers represent random, chaotic matching errors (high entropy).
The random sampling step in RANSAC is analogous to **thermal fluctuations** in a crystal lattice (similar to simulated annealing). By randomly sampling minimal subsets, the algorithm explores the state space. The subset that maximizes the inlier count corresponds to finding the global minimum energy state of the system. The outlier threshold acts like a cutoff energy filter, rejecting particles that possess too much thermal noise to fit into the ordered crystal structure of the inliers.

---

## 21. Bundle Adjustment & Spring-Mass-Damper Network Dynamics

Bundle Adjustment jointly refines camera poses and 3D points by minimizing reprojection error.

### Mathematical Formulation
We minimize the global cost function:

$$E(\mathbf{\theta}) = \sum_{i, j} v_{ij} \left\| \mathbf{x}_{ij} - \pi(P_j, \mathbf{X}_i) \right\|^2$$

Using Levenberg-Marquardt, we solve:

$$(J^T J + \lambda I) \mathbf{\delta} = -J^T \mathbf{r}$$

### Physics-Based Explanation
The optimization process can be modeled as a physical **spring-mass-damper network** in a state of dynamic equilibrium.
Let each 3D point $\mathbf{X}_i$ and camera pose $P_j = [R_j \mid \mathbf{t}_j]$ represent physical nodes in a network.
Each 2D observation $\mathbf{x}_{ij}$ acts as a physical spring connecting the 3D node $\mathbf{X}_i$ to the optical projection ray of camera $j$. The potential energy $E_{ij}$ of the spring is proportional to the square of its extension (the reprojection error):

$$E_{ij} = \frac{1}{2} k_{ij} \left\| \mathbf{x}_{ij} - \pi(P_j, \mathbf{X}_i) \right\|^2$$

where $k_{ij}$ is the spring stiffness (which maps to the measurement confidence or inverse covariance).
The goal of Bundle Adjustment is to find the minimum energy state of the entire network:

$$\min_{\mathbf{\theta}} E(\mathbf{\theta}) = \sum_{i, j} E_{ij}$$

The Newton-Raphson method updates the parameters along the direction of the net force vector $\mathbf{F} = -J^T \mathbf{r}$. The Levenberg-Marquardt update is:

$$(J^T J + \lambda I) \mathbf{\delta} = -J^T \mathbf{r}$$

In classical mechanics, the parameter $\lambda I$ acts as a **viscous drag force** (damping matrix) acting on the system's velocity:

$$\mathbf{F}_{\text{drag}} = - \lambda \mathbf{\delta}$$

*   **Large $\lambda$ (High Damping)**: The system is heavily overdamped. The update $\mathbf{\delta} \approx -\frac{1}{\lambda} J^T \mathbf{r}$ is small and follows the direction of the force vector (gradient descent). This prevents kinetic oscillations when the system is far from equilibrium.
*   **Small $\lambda$ (Low Damping)**: The system is underdamped. The update approaches the Gauss-Newton step, allowing the nodes to accelerate towards the minimum energy equilibrium state with quadratic convergence speed.
