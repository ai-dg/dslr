# DSLR: Datascience × Logistic Regression  
### Harry Potter and a Data Scientist 

---

> Build a Hogwarts Sorting Hat using logistic regression and gradient descent.  
> Your mission: classify students by house from their course scores. Hogwarts is counting on you.

---

## ▌Project Overview

This project is a complete machine learning pipeline built from scratch — with **no external ML library** — to classify students into Hogwarts houses using **logistic regression**.

It's designed as part of **42 Paris' AI curriculum**, following the logic of `ft_linear_regression`, but extended to **multiclass classification** with:
- Data analysis,
- Visualization,
- Feature selection,
- Logistic regression (One-vs-All),
- Custom gradient descent (BGD, SGD, Mini-Batch).

---

## ▌Features

✔️ Full feature engineering and cleaning  
✔️ Statistical description (mean, std, IQR, skewness, kurtosis...)  
✔️ Data visualizations: Histogram, Scatter Plot, Pair Plot  
✔️ Logistic regression classifier (One-vs-All)  
✔️ Implements BGD, SGD and Mini-Batch GD  
✔️ Output predictions in `houses.csv`  
✔️ Manual handling of all statistics (no pandas, no scikit-learn)

---

## ▌How it works

### ■ Method Used

The model uses **logistic regression** to estimate probabilities of each house, based on student features.  
One model per house is trained (One-vs-All), using gradient descent variants.

### ■ Hypothesis function

```text
h_θ(x) = 1 / (1 + e^(-θᵀx))
```
### ■ Cost function (Log-loss)

```text
J(θ) = -(1/m) * Σ [ y * log(h_θ(x)) + (1 - y) * log(1 - h_θ(x)) ]
```

### ■ Gradient formula

```text
θ_j := θ_j - α * (1/m) * Σ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾
```

▌Getting Started
■ Requirements
Python 3.x

No external ML packages allowed

You may use: `matplotlib`, `math`, `csv`, `os`

> ❌ No Pandas, Scikit-learn, Numpy, or functions like .mean(), .std(), etc.



