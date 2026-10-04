from sklearn.preprocessing import OneHotEncoder , StandardScaler 
from sklearn.compose import ColumnTransformer
import pandas as pd

class feature_engineering:
    def __init__(self):
        self.preprocessor = None
        

    def preprocessing(self, data , train_data = None):

        if train_data is not None:
        
            num = train_data.select_dtypes(include=['int64','float64']).columns

            cat = train_data.select_dtypes(include='object').columns

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(),num ),
                    ("cat",OneHotEncoder(handle_unknown="ignore"), cat)
                ]
            )

            preprocessed_train_data = self.preprocessor.fit_transform(train_data)
            preprocessed_data = self.preprocessor.transform(data)
            return preprocessed_train_data, preprocessed_data

        else:
            if self.preprocessor is None:
                 raise ValueError(
                    "Preprocessor is not fitted. "
                    "Please provide train_data first."
                )
            preprocessed_data = self.preprocessor.transform(data)

            return preprocessed_data

        




