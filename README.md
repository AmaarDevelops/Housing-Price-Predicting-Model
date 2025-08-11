🏠 House Price Prediction

This project implements a **supervised learning model** to predict median house values in California districts using the dataset provided in **Chapter 2** of *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron.  

It demonstrates a complete **Machine Learning pipeline** — from data exploration and preprocessing to model training, evaluation, and fine-tuning.

---

## 📂 Dataset
The dataset contains information about:
- Median income
- Housing median age
- Average rooms
- Average bedrooms
- Population
- Households
- Latitude & Longitude
- Median house value (target)

The dataset was fetched from the book’s repository and stored locally for analysis.

---

## ⚙️ Project Workflow
1. **Data Loading & Exploration**  
   - Loaded CSV data into Pandas DataFrame  
   - Checked for missing values & outliers  
   - Visualized distributions and correlations  

2. **Data Preprocessing**  
   - Handled missing values  
   - Feature scaling using `StandardScaler`  
   - Categorical encoding (if applicable)  
   - Train-test split  

3. **Model Selection & Training**  
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  

4. **Evaluation**  
   - Used **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**  
   - Compared models' performance  

5. **Model Tuning**  
   - Applied Grid Search for hyperparameter optimization  
   - Cross-validation to improve generalization  

---

## 📊 Results
| Model                  | RMSE Score |
|------------------------|------------|
| Linear Regression      | 68912.668  |
| Decision Tree Regressor| 57964.510      |
| Random Forest Regressor| 48755.086      |


---

## 🚀 Technologies Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## 📌 How to Run
# Clone the repository
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt


📖 References
Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron

