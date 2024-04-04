#------------------------------------------------------------------------------
# Hands-On Lab: Intro to Snowpark ML - pink diamond version
# Script:       load_clean_data.py
# Author:       Joe Burns, Tyler White, Rafa Arranz 
# Last Updated: 3/17/2024
#------------------------------------------------------------------------------

# Snowpark for Python
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
import json

# Edit the connection.json before creating the session object below
# Create Snowflake Session object
connection_parameters = json.load(open('./secrets/connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

#------------------------------------------------------------------------------
#DATA LOADING - step 3 in prototype v3

# Create a Snowpark DataFrame that is configured to load data from the CSV file
# We can now infer schema from CSV files.
diamonds_df = session.read.options({"field_delimiter": ",",
                                    "field_optionally_enclosed_by": '"',
                                    "infer_schema": True,
                                    "parse_header": True}).csv("@DIAMONDS_ASSETS")
# SNOWFLAKE ADVANTAGE: Snowpark Dataframe API 
# SNOWFLAKE ADVANTAGE: Schema inference

#------------------------------------------------------------------------------
# DATA CLEANING - step 5 in prototype v3 (skip step 4)

#UPPERCASE ALL COLUMN NAMES
for colname in diamonds_df.columns:
    if colname == '"table"':
       new_colname = "TABLE_PCT"
    else:
        new_colname = str.upper(colname)
    diamonds_df = diamonds_df.with_column_renamed(colname, new_colname)

#UPPERCASE STRING VALUES (CUT)
def fix_values(columnn):
    return F.upper(F.regexp_replace(F.col(columnn), '[^a-zA-Z0-9]+', '_'))

for col in ["CUT"]:
    diamonds_df = diamonds_df.with_column(col, fix_values(col))

#Preview cleaned data
diamonds_df.show()

#------------------------------------------------------------------------------
# DATA SERVING - step 6 in prototype v3

#write diamonds data to Snowflake table
diamonds_df.write.save_as_table("Diamonds", mode="overwrite")
# SNOWFLAKE ADVANTAGE: Snowflake Tables (not file-based)

print("Created Diamonds table with diamonds_df inferred schema!")


#------------------------------------------------------------------------------
# INSERT NEW DIAMONDS DATA IN BATCH PRED SPROC STEP
