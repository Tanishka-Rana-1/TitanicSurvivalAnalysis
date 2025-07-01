🚢 Titanic Survival Analysis & Prediction


📜 Overview
This project explores the famous Titanic dataset to understand passenger survival patterns through rich Exploratory Data Analysis (EDA) and predictive modeling.
We use machine learning techniques like Decision Trees and K-Nearest Neighbors (KNN) to predict survival outcomes, ultimately identifying which model better fits the data.

📂 Dataset
Source: Local CSV (titanic.csv) from Stanford’s cleaned dataset

Features used:
Pclass, Age, Fare, Siblings/Spouses Aboard, Parents/Children Aboard, Sex, Survived

🚀 Key Steps
🔍 1. Data Preprocessing
Removed duplicates.

Created new features:

FamTot: total family size on board

Solo: indicator if traveling alone

AgeGrp & FareGrp: binned age & fare into categories

Removed outliers using the IQR method on Fare and Age.

📊 2. Exploratory Data Analysis (EDA)
Used matplotlib and seaborn to reveal insights:

Pie & Stacked Bar: more males on board; females survived more.

Line & Violin plots: children had highest survival; family size impacted survival.

Hexbin & Heatmap: showed 1st class clustered around higher fares, strong correlation between fare/class & survival.

Point plots: survival varied sharply by ticket class and fare.

Pairplot & Subplots: multi-angle look at feature interactions.

🤖 3. Machine Learning Models
Decision Tree (depth=5)

K-Nearest Neighbors (k optimized from 1 to 20)

🔬 4. Evaluation
Compared models on accuracy score using test data.

Visualized model performance with scatter & line plots.


🏆 Results
Model	Settings	Accuracy
Decision Tree	depth=5	~80.0%
KNN	k=16 (best)	~82.7%

✅ Conclusion:
The KNN model with k=16 neighbors performed slightly better, suggesting that local patterns among passengers provided more robust survival predictions.
