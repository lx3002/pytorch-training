import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import QuantileTransformer



df= pd.read_csv('C:\\Users\\PC\\Desktop\\New folder (7)\\ml\\diabetes.csv')





