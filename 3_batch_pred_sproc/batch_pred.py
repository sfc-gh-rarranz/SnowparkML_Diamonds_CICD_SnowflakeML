# Snowpark for Python
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.functions import udf
import snowflake.snowpark.functions as F

# Snow ML stuffs
from snowflake.ml.registry import Registry

# Misc
import json

# --------------------------------------------------------------------------------------------------
## MAIN FUNCTION TO RUN OPTIMAL MODEL FOR BATCH INFERENCE ON TEST DATA - execution runtime on M = 2.5min
def main(session: Session) -> str:
    # Edit the connection.json before creating the session object below
    # Create Snowflake Session object
    # connection_parameters = json.load(open('connection.json'))
    # session = Session.builder.configs(connection_parameters).create()
    # session.sql_simplifier_enabled = True

    # first creates a Registry object, which is used to manage models in Snowflake.
    reg = Registry(session=session, database_name="ML_HOL_DB", schema_name="ML_HOL_SCHEMA")
    # retrieves all the models in the registry and stores them in reg_df.
    reg_df = reg.show_models()
    # specifies the name of the model to use for prediction
    model_name = 'DIAMONDS_PRICE_PREDICTION'
    # Retrieves the specified model and its versions.
    reg_df = reg.get_model(model_name).show_versions()
    # # Adds a new column "Mean Absolute Percentage Error" to reg_df by extracting it from the metadata of each version.
    reg_df["Mean absolute percentage error"] = reg_df["metadata"].apply(
        lambda x: json.loads(x)["metrics"]["Mean absolute percentage error"]
     )
    # # sorts the versions by accuracy in descending order and selects the version with the highest accuracy as the deployed version.
    best_model = reg_df.sort_values(by="Mean absolute percentage error", ascending=False)
    deployed_version = best_model["name"].iloc[0]

    # sets the default version of the model to the deployed version.
    m = reg.get_model(model_name)
    m.default = deployed_version
    mv = m.default
    # define the dataset to run inference on
    inference_df =  session.table('"ML_HOL_DB"."ML_HOL_SCHEMA".Diamonds')
    #target =  session.table('"ML_HOL_DB"."ML_HOL_SCHEMA".Diamonds_Predictions')

    #run batch inference on test dataframe
    remote_prediction = mv.run(inference_df, function_name='predict') # https://docs.snowflake.com/en/developer-guide/snowpark-ml/reference/latest/api/model/snowflake.ml.model.ModelVersion
    remote_prediction.show()
    #run batch inference on single record in test dataframe
    remote_prediction = mv.run(inference_df.limit(1), function_name='predict')
    remote_prediction.show()
    remote_prediction.write.save_as_table("Diamonds_Predictions", mode="overwrite")
    

    # Finally, it returns a message indicating that the batch prediction is complete.
    # you could return the number of rows inserted or updated here if you would like by getting length of F.when_matched() and F.when_not_matched()
    return "Batch prediction complete! Prediction results merged to 'Diamonds_Predictions' table!"


# --------------------------------------------------------------------------------------------------
## WIP - STREAMS & TASKS TO FEED NEW DATA TO MODEL FOR BATCH INFERENCE

# # Retrieves the source and target tables from the database.
# source = session.table('DB.SCHEMA.STREAM_ON_TABLE')
# target = session.table('DB.SCHEMA.DESTINATION_TABLE')

# # runs the prediction function of the model on the source table and stores the predictions in preds.
# preds =  mv.run(source, function_name="predict")

# # prepares the updates to be made to the target table. 
# # The updates include all columns in the predictions that do not contain ‘METADATA’ in their names, and a ‘META_UPDATED_AT’ column that records the current timestamp.
# # TODO: Is the if clause supposed to be based on "META_UPDATED_AT"?
# cols_to_update = {c: source[c] for c in source.schema.names if "METADATA" not in c} # don't push these metdata column downstream
# metadata_col_to_update = {"META_UPDATED_AT": F.current_timestamp()} # push this metdata column downstream
# updates = {**cols_to_update, **metadata_col_to_update}

# # merges the predictions into the target table DIM_CUSTOMER based on the ‘ID_COL’ column. 
# # If a row in the target table matches a row in the predictions, it updates the row with the updates; if not, it inserts a new row with the updates
# target.merge(remote_prediction, target['CARAT'] == source['CARAT'] & (target['COLOR'] == source['COLOR']) & (target['CLARITY'] == source['CLARITY']) 
    #              & (target['DEPTH'] == source['DEPTH']) & (target['TABLE_PCT'] == source['TABLE_PCT']) & (target['PRICE'] == source['PRICE']) 
    #              & (target['CUT'] == source['CUT']) & (target['X'] == source['X']) & (target['Y'] == source['Y']) & (target['Z'] == source['Z']), 
    #                  [F.when_matched().update(updates), F.when_not_matched().insert(updates)])
    

# --------------------------------------------------------------------------------------------------
# For local debugging
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