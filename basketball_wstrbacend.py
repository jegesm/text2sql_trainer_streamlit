import vanna as vn
import os

import pandas as pd
import pymssql
import streamlit_backend as sb

from erd_viewer_mssql_streamlit_module import Session, ERDViewer

def read_dict_from_file(file_path):
    import json

    # Read the dictionary from a file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Print the dictionary
    return data


## Define connection details
# This opens a connection to the MS SQL server
# read connection info from file
conn_info = read_dict_from_file('/v/wfct0p/API-tokens/basketball-mssql-connection-info.json')
database_name = conn_info['database']
model_name = conn_info['model_name']

token_path = "/v/wfct0p/API-tokens/vn_api_key.token"

# Connect to the database
conn = pymssql.connect(server=conn_info['server'], 
    user=conn_info['user'], 
    password=conn_info['password'], 
    database=conn_info['database'])
session = Session(database=database_name, connection=conn)

# schema_ddl = ""
# with open('../basketball-db-tables-create-Full.sql') as f:
#     schema_ddl = f.read()

## Initialize Vanna.ai
if  not os.path.exists(database_name+"-graph.png"):
    print("Creating ERD")
    erd = ERDViewer(session=session)
    graph = erd.get_graph()

app = sb.run(session=session,
    token_path=token_path,
    model_name=model_name, 
    title=database_name,
    erd_path=database_name+"-graph.png")

# Add schema
#if schema_ddl:
#    vn.add_ddl(schema_ddl)
# 
# app.run()