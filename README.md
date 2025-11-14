# 🧠 Advertising Click Prediction using Logistic Regression  

## 📌 Project Overview  
This project predicts whether a user will **click on an online advertisement** based on their demographics and online activity.  
By analyzing user behavior and building a **Logistic Regression model**, we identify the key factors that influence ad engagement — enabling data-driven marketing insights.

---

## 🎯 Objective  
To develop a predictive model that determines the **likelihood of a person clicking on an ad** using behavioral and demographic data such as:  
- Time spent on site  
- Daily internet usage  
- Age  
- Gender  
- Area income  

---

## 📊 Dataset Description  
The dataset `advertising.csv` contains information about user behavior and demographics.  
**Key features:**
- `Daily Time Spent on Site`  
- `Age`  
- `Area Income`  
- `Daily Internet Usage`  
- `Male` (0 = Female, 1 = Male)  
- `Clicked on Ad` (Target variable: 0 = No, 1 = Yes)  

🧹 **Excluded Columns:**  
`Ad Topic Line`, `City`, `Country`, `Timestamp` — these were removed as they have minimal predictive impact.

---

## 🔍 Exploratory Data Analysis (EDA)  
Performed exploratory data analysis using **Pandas**, **Seaborn**, and **Matplotlib** to visualize relationships between variables.  

**Key Insights:**
- Strong positive correlation between **Daily Time Spent** and **Internet Usage**  
- Negative correlation between **Age** and **Ad Clicks** (younger users click more)  
- Gender had a smaller effect on ad click probability  

A **ydata-profiling report** was also generated for detailed statistical exploration.

---

## ⚙️ Model Building  
Model: **Logistic Regression (Scikit-learn)**  
- Training-Testing Split: 70% / 30%  
- Iterations: 1000 for stable convergence  
- Features used:  
  `['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']`

**Code Snippet:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

