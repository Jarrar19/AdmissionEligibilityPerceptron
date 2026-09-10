# Machine Learning Methodology & Algorithmic Theory

**Project:** Perceptron-Based Admission Eligibility Prediction System (`admission-eligibility-perceptron`)  
**Primary Algorithm:** Single-Layer Binary Perceptron (`sklearn.linear_model.Perceptron`)  
**Pipeline Components:** `StandardScaler` $\to$ `Perceptron`  

---

## 1. Algorithmic Foundation: The Rosenblatt Perceptron

Proposed by Frank Rosenblatt in 1958, the Perceptron is the foundational artificial neuron for supervised binary classification. It models a biological neuron by taking a vector of real-valued inputs, computing a weighted linear combination, and passing the result through a step activation function to emit a binary decision.

### Mathematical Formulation
Given an input feature vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$, an associated weight vector $\mathbf{w} = [w_1, w_2, \dots, w_n]^T \in \mathbb{R}^n$, and a scalar bias $b \in \mathbb{R}$:

1. **Pre-Activation Net Input (Decision Score $z$)**:
   $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$$

2. **Step Activation Function (Heaviside Step)**:
   $$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \geq 0 \quad (\text{Eligible}) \\ 0 & \text{if } z < 0 \quad (\text{Not Eligible}) \end{cases}$$

---

## 2. Training Algorithm & Weight Update Rule (§8.3)

The Perceptron is an error-driven, online learning algorithm. Training iterations pass through instances one by one:

1. Initialize weights $\mathbf{w} = \mathbf{0}$ (or small random values) and bias $b = 0$.
2. For each training sample $(\mathbf{x}_i, y_i)$ where $y_i \in \{0, 1\}$:
   - Compute model prediction $\hat{y}_i \in \{0, 1\}$.
   - If $\hat{y}_i = y_i$, the sample is correctly classified; **no update occurs**.
   - If $\hat{y}_i \neq y_i$, misclassification has occurred; adjust weights and bias using the gradient-free update rule:
     $$w_j \leftarrow w_j + \eta \cdot (y_i - \hat{y}_i) \cdot x_{ij}, \quad \forall j \in \{1, \dots, n\}$$
     $$b \leftarrow b + \eta \cdot (y_i - \hat{y}_i)$$
   where $\eta > 0$ represents the learning rate (`eta0 = 1.0`).

### Update Dynamics
- **False Negative ($\hat{y} = 0, y = 1$)**: $(y - \hat{y}) = +1$. Weights update as $w_j \leftarrow w_j + \eta x_{ij}$ and $b \leftarrow b + \eta$, shifting the hyperplane toward the positive sample.
- **False Positive ($\hat{y} = 1, y = 0$)**: $(y - \hat{y}) = -1$. Weights update as $w_j \leftarrow w_j - \eta x_{ij}$ and $b \leftarrow b - \eta$, pushing the hyperplane away from the negative sample.

---

## 3. Convergence Theorem & Linear Separability

### The Perceptron Convergence Theorem (Novikoff, 1962)
If a training dataset is **linearly separable** (meaning there exists a hyperplane $\mathbf{w}^* \cdot \mathbf{x} + b^* = 0$ that separates the two classes with a positive margin $\gamma > 0$), the Perceptron algorithm is mathematically guaranteed to converge to a separating hyperplane in a finite number of updates $k \le (R / \gamma)^2$, where $R = \max \|\mathbf{x}_i\|$.

### Behavior on Non-Linearly Separable Data
In real-world admission scenarios, applicant profiles have borderline overlaps (e.g. students with modest entrance scores but exceptional research portfolios, or high test scores but low attendance). Because the classes overlap, **the data is not perfectly linearly separable**. 

When data is non-linearly separable:
- The standard Perceptron will never converge to 100% accuracy.
- Hyperparameter `max_iter=1000` and `tol=1e-3` provide an early stopping condition once training score stabilizes.
- **Our Model Achieves 73.00% Test Accuracy**. Per §1 Rule 3, this is reported transparently as a genuine reflection of linear classification boundaries.

---

## 4. Why Feature Scaling (`StandardScaler`) is Mathematically Mandatory

In the Perceptron update equation:
$$\Delta w_j = \eta (y - \hat{y}) x_j$$
The magnitude of the weight adjustment $\Delta w_j$ is directly proportional to the raw feature value $x_j$:
- If `Entrance_Score` ranges from 0 to 100 while `CGPA` ranges from 0 to 10, the gradient step for `Entrance_Score` would be **10 to 100 times larger** than for `CGPA` merely due to measurement units.
- Unstandardized features cause the optimization trajectory to oscillate wildly along high-magnitude axes while ignoring low-magnitude features.
- `StandardScaler` standardizes each feature to zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1$):
  $$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- **Data Leakage Safeguard**: Crucially, $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are computed strictly on $X_{\text{train}}$. Test instances are transformed using the stored training statistics.

---

## 5. Model Interpretation: Weights Vector Analysis (§8.4)

Evaluating the learned weights $\mathbf{w}$ reveals the orientation of the separating hyperplane:

| Feature Name | Learned Weight ($w_i$) | Influence Direction | Academic Interpretation |
| :--- | :--- | :--- | :--- |
| **`Entrance_Score`** | **+3.5826** | Strong Positive | Primary filter; high entrance test scores heavily increase eligibility probability. |
| **`Extracurricular_Score`** | **+2.4881** | Strong Positive | Substantial positive influence on candidate holistic admission. |
| **`CGPA`** | **+1.5494** | Positive | Solid predictor reflecting sustained undergraduate academic capability. |
| **`12th_Percentage`** | **+0.7692** | Moderate Positive | Prerequisite academic strength. |
| **`Interview_Score`** | **+0.7587** | Moderate Positive | Interpersonal and domain communication rating. |
| **`Aptitude_Score`** | **-0.2341** | Minor Negative | Minor correlation artifact due to co-linearity with Entrance Score. |
| **`10th_Percentage`** | **-0.6845** | Minor Negative | Distant historical indicator; overshadowed by recent metrics. |
| **`Attendance`** | **-2.3495** | Negative | Strong negative interaction when other scores are borderline. |
| **`Previous_Backlogs`** | **-8.1285** | Severe Negative | Heaviest penalty; uncleared backlogs sharply push $z$ below zero. |
| **Bias ($b$)** | **+2.0000** | Positive Baseline | Baseline positive intercept in normalized feature space. |

> **Critical Caveat (§8.4 & §13)**: Feature weights indicate the mathematical contribution of a normalized variable to the model's decision function $z$. They represent **correlational decision weights**, not real-world causal mechanisms.

---

## 6. Secondary Model Comparison (§8.4)

To evaluate the Perceptron against standard linear baselines, a Logistic Regression model was trained on the identical 800-sample training partition and evaluated on the identical 200-sample test partition:

| Model | Test Accuracy | Precision | Recall | F1-Score | Decision Boundary Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Perceptron (Primary)** | **73.00%** | **72.44%** | **82.88%** | **77.31%** | Hard linear hyperplane (Step function) |
| **Logistic Regression (Secondary)** | **76.00%** | **76.47%** | **81.98%** | **79.13%** | Smooth linear probability (Sigmoid) |

### Why Perceptron Remains the Primary Model
Logistic Regression scores slightly higher (76.00% vs 73.00%) because its loss function (log-loss) penalizes confidence errors smoothly, whereas the Perceptron only updates on discrete misclassifications. However, **Perceptron is retained as the sole primary classifier** per §1 Rule 1 and §16 Philosophy: the primary pedagogical objective of this NNDL PBL project is to understand the foundational neural computing unit—the artificial neuron—and demonstrate its operational dynamics, strengths, and limitations.
