#------------------------------------------------------------------------------
# Hands-On Lab: Intro to Snowpark ML - pink diamond version
# Script:       model_training.py
# Author:       Joe Burns, Tyler White, Rafa Arranz 
# Last Updated: 3/30/2024
#------------------------------------------------------------------------------
# Snowpark for Python
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
import snowflake.snowpark.functions as F
from snowflake.snowpark import types as T

# Snowpark ML
import snowflake.ml.modeling.preprocessing as snowml #MMS,OE,OHE
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.metrics.correlation import correlation
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.registry import Registry
from snowflake.ml.modeling.metrics import mean_absolute_percentage_error, mean_squared_error
# from snowflake.ml._internal.utils import identifier

# data science libs
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

# Misc
#import json
# import joblib
import ast
# import logging
# import re
import numpy as np

# warning suppresion
import warnings; warnings.simplefilter('ignore')

# Edit the connection.json before creating the session object below
# Create Snowflake Session object
# connection_parameters = json.load(open('/Users/rarranz/Documents/programming_directory/CAKE_POC/secrets/connection.json'))
# session = Session.builder.configs(connection_parameters).create()
# session.sql_simplifier_enabled = True

# snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
# snowpark_version = VERSION

# # Current Environment Details
# print('\nConnection Established with the following parameters:')
# print('User                        : {}'.format(snowflake_environment[0][0]))
# print('Role                        : {}'.format(session.get_current_role()))
# print('Database                    : {}'.format(session.get_current_database()))
# print('Schema                      : {}'.format(session.get_current_schema()))
# print('Warehouse                   : {}'.format(session.get_current_warehouse()))
# print('Snowflake version           : {}'.format(snowflake_environment[0][1]))
# print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))

# --------------------------------------------------------------------------------------------------

# MODEL VERSIONING FUNCTION - need to package it and import it from common.py 

def get_next_version(reg, model_name) -> str:
    """
    Returns the next version of a model based on the existing versions in the registry.

    Args:
        reg: The registry object that provides access to the models.
        model_name: The name of the model.

    Returns:
        str: The next version of the model in the format "V_<version_number>".

    Raises:
        ValueError: If the version list for the model is empty or if the version format is invalid.
    """
    models = reg.show_models()
    if models.empty:
        return "V_1"
    elif model_name not in models["name"].to_list():
        return "V_1"
    max_version_number = max(
        [
            int(version.split("_")[-1])
            for version in ast.literal_eval(
                models.loc[models["name"] == model_name, "versions"].values[0]
            )
        ]
    )
    return f"V_{max_version_number + 1}"


# --------------------------------------------------------------------------------------------------

#MAIN PRE-PROCESSING + MODEL TRAINING FUNCTION - execution runtime on M = 2.5min

def main(session: Session) -> str:
    _ = session.sql("ALTER WAREHOUSE ML_HOL_WH SET WAREHOUSE_SIZE = MEDIUM WAIT_FOR_COMPLETION = TRUE").collect()
    # It first reads data from a Snowflake table named ‘DIAMONDS’ into a dataframe diamonds_df.
    diamonds_df = session.table('"ML_HOL_DB"."ML_HOL_SCHEMA".Diamonds')

    #Input and Output column definitions
    # Define the label column
    output_actual = ['PRICE']
    # # What are the numeric columns? Define Snowflake numeric types
    numeric_types = [T.DecimalType, T.DoubleType, T.FloatType, T.IntegerType, T.LongType]
    # # Get numeric columns
    cols_to_scale = [col.name for col in diamonds_df.schema.fields if type(col.datatype) in numeric_types]
    # remove the output and mormalzie the categorical columns
    cols_to_scale.remove('PRICE')
    scale_cols_output = [col + '_NORM' for col in cols_to_scale]
    # # What are the categorical columns? Define Snowflake categorical types
    categorical_types = [T.StringType]
    # # Get categorical columns
    cat_cols = [col.name for col in diamonds_df.schema.fields if type(col.datatype) in categorical_types]
    # print(cat_cols)

    # We COULD ordinally encode all of these columns. However for the sake of the demo we will OHE 'CUT'.
    cols_to_ohe = ['CUT']
    ohe_cols_output = [col + '_OHE' for col in cols_to_ohe]
    # So I could typcially just list the smaller of the two sets (ones to OE or OHE) and then do list subraction to set the larger list dynamically.
    # I do this so I only have to explicitly define the smaller of the two categorical lists (either OHE or OE)
    set1 = set(cat_cols)
    set2 = set(cols_to_ohe)
    cols_to_oe = list(set1 - set2)
    oe_cols_output = [col + '_OE' for col in cols_to_oe]
    
    # DEFINE PIPELINE with 4 steps: a MinMaxScaler, an OrdinalEncoder, A OneHotEncoder, and a GridSearchCV for an XGBRegressor.
    # dictionary to define order of categories for ordinal encoding of CLARITY and COLOR features
    categories = {
    "CLARITY": np.array(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]),
    "COLOR": np.array(['D', 'E', 'F', 'G', 'H', 'I', 'J']),
    }
    # dictionary that specifies XGBRegresseor model parameters
    parameters = {
    "n_estimators":[100, 200, 300, 400, 500],
    "learning_rate":[0.1, 0.2, 0.3, 0.4, 0.5],
    }
    #MinMax scale 'CARAT', 'DEPTH', 'TABLE_PCT', 'X', 'Y', 'Z'
    snowml_mms = snowml.MinMaxScaler(
                    clip=True,
                    input_cols=cols_to_scale,
                    output_cols=scale_cols_output,
                )
    #Ordinal Encode COLOR, CLARITY features
    snowml_oe = snowml.OrdinalEncoder(
                    input_cols=cols_to_oe,
                    output_cols=oe_cols_output,
                    categories=categories,
                    drop_input_cols=True,
                )
    #One Hot Encode CUT feature
    snowml_ohe = snowml.OneHotEncoder(
                    input_cols=cols_to_ohe,
                    output_cols=ohe_cols_output,
                    drop_input_cols=True,
                )
    #GridSearchCV uses cross validation to perform hyperparameter tuning on XGBRegressor model
    #GridSearchCV will search over the specified parameter grid for the XGBRegressor and find the best parameters based on the MAPE.
    snowml_gscv = GridSearchCV(
                    estimator=XGBRegressor(),
                    param_grid=parameters,
                    n_jobs = -1,
                    scoring="neg_mean_absolute_percentage_error",
                    label_cols=output_actual,
                    output_cols=['PREDICTED_PRICE']
                )
    #Pipeline chaning together pre-procesisng steps and model training step
    pipeline = Pipeline(
    [("MMS", snowml_mms), ("OE", snowml_oe), ("OHE", snowml_ohe),("XGBReg_GSCV", snowml_gscv)]
    )
    # Split the data into training and testing sets
    train_df, test_df = diamonds_df.random_split([0.8, 0.2], seed=42)
    #Pipeline is fitted on the training data 
    pipeline.fit(train_df)
    #Fitted model makes predictions on the test data.
    result_df = pipeline.predict(train_df)
    #Display predictions
    # result_df.show()

    # ACCURACY METRICS
    LABEL_COLUMNS = ['PRICE']
    OUTPUT_COLUMNS = ['PREDICTED_PRICE']
    # Use Snowpark ML metrics to calculate the mean absolute percentage error (MAPE) and mean squared error (MSE) of the model’s predictions.
    metrics = {
        "Mean absolute percentage error": mean_absolute_percentage_error(
            df=result_df, 
            y_true_col_names=LABEL_COLUMNS, 
            y_pred_col_names=OUTPUT_COLUMNS
        ),
        "Mean squared error": mean_squared_error(
            df=result_df, 
            y_true_col_names=LABEL_COLUMNS, 
            y_pred_col_names=OUTPUT_COLUMNS
        ),
    }
    #View accuracy metrics
    result_df.select([*LABEL_COLUMNS, *OUTPUT_COLUMNS]).show()
    # print(f'''Mean absolute percentage error: {metrics["Mean Absolute Percentage Error"]}''')
    # print(f'''Mean squared error: {metrics["Mean squared error"]}''')


    #  Registry object is created and used to log the model and its performance metrics
    reg = Registry(session=session, database_name="ML_HOL_DB", schema_name="ML_HOL_SCHEMA")
    
    reg.log_model(
        model_name="DIAMONDS_PRICE_PREDICTION",
        version_name=get_next_version(reg, "DIAMONDS_PRICE_PREDICTION"),
        model=pipeline,
        metrics=metrics,
    )
    #reg_df = reg.show_models()
    # _ = session.sql("ALTER WAREHOUSE ML_HOL_WH SET WAREHOUSE_SIZE = XSMALL WAIT_FOR_COMPLETION = TRUE").collect()
    # The function returns a string indicating that the model training is complete and provides the MAPE of the model.
    mape = metrics["Mean absolute percentage error"]
    version_name=get_next_version(reg, "DIAMONDS_PRICE_PREDICTION")
    return "The DIAMONDS_PRICE_PREDICTION Model is trained and logged! The accuracy of the model is {} mape. The current model version is {}".format(mape, version_name)
    


#--------------------------------------------------------------------------------------------------

# For local debugging - The if __name__ == '__main__': block at the end is used to run the main function when the script is run directly.  
# The snowpark_utils.get_snowpark_session() function is used to get a Snowpark session, which is needed to interact with Snowflake. The session is closed at the end to free up resources.
# Be aware you may need to type-convert arguments if you add input parameters

if __name__ == '__main__':
    # Add the utils package to our path and import the snowpark_utils function
    import os, sys
    current_dir = os.getcwd()
    parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(parent_parent_dir)

    from utils import snowpark_utils
    session = snowpark_utils.get_snowpark_session()

    if len(sys.argv) > 1:
        print(main(session, *sys.argv[1:]))  # type: ignore
    else:
        print(main(session))  # type: ignore

    session.close()