# hackathon_project
This project aims to predict the likelihood of individuals receiving the h1n1 flu vaccine and the seasonal flu vaccine. Using a dataset provided, I have developed a model to predict the probability of vaccine uptake.

## Approach

1. **Data Preprocessing**:
   - Encoded categorical variables to numeric formats.
   - Scaled numerical features to ensure uniformity.

2. **Modeling**:
   - Utilized a `MultiOutput RandomForestClassifier` to handle the multilabel nature of the problem.
   - Trained the model on the processed training data.
   - Evaluated the model using the ROC AUC score to ensure balanced performance across both target variables.

3. **Evaluation**:
   - Achieved robust performance with ROC AUC scores of 0.85 for xyz vaccine prediction and 0.83 for seasonal flu vaccine prediction, yielding a mean ROC AUC score of 0.84.

RESULT

ROC AUC for xyz_vaccine: 0.8251032658481036
ROC AUC for seasonal_vaccine: 0.8470950025550955
Mean ROC AUC: 0.8360991342015995
Submission file created successfully!
