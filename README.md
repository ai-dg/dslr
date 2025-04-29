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

### ■ Installation & Usage
1. Clone this repository

```bash
git clone https://github.com/your-username/dslr.git
cd dslr
```

2. Launch data visualization scripts:

```bash
python3 describe.py dataset_train.csv
python3 histogram.py dataset_train.csv
python3 scatter_plot.py dataset_train.csv
python3 pair_plot.py dataset_train.csv
```

3. Train the classifier:

```bash
python3 logreg_train.py dataset_train.csv
# Then choose method: BGD, SGD or M-Batch
```

4. Predict student houses from test set:

```bash
python3 logreg_predict.py dataset_test.csv
```
This generates houses.csv.

### ▌Bonus Features

✔️ Normalization of features
✔️ Training with multiple optimization strategies
✔️ Export trained θ per house to JSON
✔️ Visualization of pairwise subject correlations
✔️ Calculation of precision vs actual labels


### ▌Example Output

```bash
$ python3 logreg_train.py dataset_train.csv
Choose training method (BGD, SGD, M-Batch): SGD
📉 SGD average Log loss for Gryffindor: 0.045584
📉 SGD average Log loss for Hufflepuff: 0.058805
📉 SGD average Log loss for Ravenclaw: 0.069896
📉 SGD average Log loss for Slytherin: 0.048540

$ python3 logreg_predict.py dataset_test.csv
✅ Results exported into houses.csv
✅ Accuracy: 98.25%
```

### ▌Summary
This project teaches you how to:

- Manually analyze and visualize a dataset,
- Implement your own gradient descent,
- Train and apply a multiclass classifier,
- Evaluate and optimize performance (log-loss, accuracy),
- Work like a real data scientist... muggle or not.

### 📜 License
Project developed at **42 Paris** as part of the **AI/ML discovery curriculum.**
For educational purposes only.
Do not publish derivative work without referencing the original subject.

