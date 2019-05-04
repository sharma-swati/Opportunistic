# CS205 Final Project

## The Task:
In this project we will try and build a predicitve model to being positive to cancer or malignancy.
- Using the features available at [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015), investigate what features you would like to use in your model. 
- Under the tab "Data, Documentation, Codebooks, SAS Code" choose one of the categories: "Demographics Data", "Dietary Data", "Examination Data", "Laboratory Data", or "Questionnaire Data". 
- For each of these categories there is a list of sub-categories available. Clicking the "Doc File" link will provide the information about features in that sub-category and feature names. 
**For example**: Our target of focus (cancer or malignancy) is found under Questionnaire Data -> Medical Conditions -> MCQ220 - Ever told you had cancer or malignancy. Under this question the variable name **MCQ220** is what we will use as an identifier in our code.

## File list:
- **nhanes.py:** implementation of the data preprocessing logic as well as definition an example dataset.
- **Demo_Dataset.ipynb:** Jupyter notebook file to demonstrate the basic usage of sample dataset.

## How to use:
1) Download [raw data files](https://drive.google.com/file/d/1hFp7O747408D8t5442f0Sjit7wXKXI1z/view?usp=sharing) and decompress them.
2) Install Python 3 and the following packages: joblib, numpy, pandas, matplotlib, scipy, sklearn, jupyter, pytorch.
3) Use Demo_Dataset.ipynb to see an example on how to use the predefined task.
4) Expand nhanes.py to define new tasks by following the implementation logic of the provided sample.


##### For a detailed explanation of the methods used here for the cost-sensetive health dataset, please refer to: ["Nutrition and Health Data for Cost-Sensitive Learning"](https://arxiv.org/abs/1902.07102)
