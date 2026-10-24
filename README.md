# Car Price Prediction with Machine Learning

Task 3 of the **CodeAlpha Data Science Internship**.

A regression model that predicts the selling price (in lakhs) of a used vehicle from its age, original price, kilometres driven, fuel type, seller type, transmission and number of previous owners.

## Dataset
`car_data.csv` has 301 rows and 9 columns (299 after removing 2 duplicates). There are no missing values. Note that it includes motorcycles and scooters as well as cars.

## Approach
1. Loaded and inspected the data, removed duplicates.
2. Feature engineering: created `Car_Age` (2018 minus year). Dropped `Car_Name` (98 unique values, too many for ~300 rows).
3. Explored the data with distributions, scatter plots, boxplots and a correlation heatmap.
4. Built one pipeline that log-transforms the skewed features (`Present_Price`, `Driven_kms`), scales numbers, one-hot encodes categories, and log-transforms the target. Everything is learned from training data only, so there is no data leakage.
5. Compared 4 models with 5-fold cross-validation, then evaluated all of them on a held-out 20% test set.
6. Checked feature importance with permutation importance.

## Results
| Model | CV R² | Test R² | Test MAE (lakhs) |
|---|---|---|---|
| Linear Regression | 0.955 | 0.890 | 0.97 |
| Ridge Regression | 0.953 | 0.891 | 0.96 |
| Gradient Boosting | 0.894 | 0.844 | 0.94 |
| Random Forest | 0.894 | 0.663 | 1.31 |

Linear Regression was chosen by cross-validation. It predicts selling price with an average error of about 1 lakh.

## Key insights
- **Present price** is by far the most important feature, followed by **car age**. The other features add very little.
- Log-transforming the target and skewed features was essential. Without it, the linear models were very unstable because of extreme values (a Land Cruiser at 92.6 lakhs, a scooter with 500,000 km).
- Tree-based models did worse here, since the dataset is small and has a few very expensive vehicles outside the range they learn from.
- The largest errors are on older, expensive cars. The model does not know the brand, which affects how quickly a car loses value.

## Limitations
- Small dataset (~300 rows), mixing cars and motorcycles.
- Car name/brand not used. Adding brand information and more data would likely improve accuracy.

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Car_Price_Prediction.ipynb
```
Keep `car_data.csv` in the same folder as the notebook.

## Files
- `Car_Price_Prediction.ipynb`: full analysis and model
- `car_data.csv`: dataset
