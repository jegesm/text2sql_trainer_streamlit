import vanna as vn
import os

import pandas as pd
import psycopg2
from copy import deepcopy

import streamlit as st
from st_clickable_images import clickable_images
from code_editor import code_editor
from streamlit.components.v1 import html
from streamlit_float import *
from pandas.errors import DatabaseError
import base64

from erd_viewer_mssql_streamlit_module import *

# @st.cache_data(show_spinner="Fetching training data...")
# def _get_training_data(_vn) -> pd.DataFrame:
#     return _vn.get_training_data()

# @st.cache_data(show_spinner="Generating sample questions ...")
# def generate_questions_cached():
#     return vn.generate_questions()


# @st.cache_data(show_spinner="Generating SQL query ...")
# def generate_sql_cached(question: str):
#     return vn.generate_sql(question=question)


# @st.cache_data(show_spinner="Running SQL query ...")
# def run_sql_cached(sql: str):
#     return vn.run_sql(sql=sql)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    return vn.get_plotly_figure(plotly_code=code, df=df)


# @st.cache_data(show_spinner="Generating followup questions ...")
# def generate_followup_cached(question, df):
#     return vn.generate_followup_questions(question=question, df=df)

btn_settings_editor_btns = [
# {
#     "name": "copy",
#     "feather": "Copy",
#     "hasText": True,
#     "alwaysOn": True,
#     "commands": ["copyAll"],
#     "style": {"top": "0rem", "right": "0.4rem"}
#   },
    {
    "name": "update",
    "feather": "RefreshCw",
    "primary": True,
    "hasText": True,
    "showWithIcon": True,
    "commands": ["submit"],
    "style": {"bottom": "0rem", "right": "0.4rem"}
  }]

class StreamlitBackend():

    def __init__(self, **kwargs):
        self.df = None
        if "session" in kwargs:
            self.session = kwargs['session']
        if "title" in kwargs:
            self.title = kwargs['title']
        if "graph" in kwargs:
            self.graph = kwargs['graph']
        if "erd_path" in kwargs:
            self.erd_path = kwargs['erd_path']
        self.model_name = kwargs['model_name']
        self.token_path = kwargs['token_path']
        self.sql = None
        self.question = None
        
        self.vn = vn
        # self.training_data = _get_training_data(vn)
        self._set_token()
        self.vn.run_sql = self.run_sql
        self.vn.set_model(self.model_name)        

    def _set_token(self):
        with open(self.token_path) as f:
            token = f.read().strip()
        self.vn.set_api_key(token)
        return True

    def run_sql(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.session.connection)

    def check_sql_validity(self, sql: str) -> None:
        try:
            df = self.vn.run_sql(sql)
            st.write(self.df)
            st.success("SQL query is valid!")
        except Exception as e:
            st.error("SQL query is invalid!")
            st.error(e)
            st.stop()
        
        return df

    def upload_training_data(file_csv, append=True) -> None:
        self.df = pd.read_csv(file_csv)
        if not append and vn.get_training_data().shape[0] > 0:
            for id in vn.get_training_data().id:
                self.vn.remove_training_data(id)
        for idx, row in df.iterrows():
            if row['training_data_type'] == "sql":
                self.vn.train(question=row['question'], sql=row['content'])

    def download_training_data(self) -> None:
        training_data = self.vn.get_training_data()
        if training_data.shape[0]:
            training_data.drop(axis=1, columns=['id'], inplace=True)
        return training_data.to_csv(index=False)
        
    def run(self):
        # Init Streamlit
        st.set_page_config(layout="wide")
        if "generated_sql" not in st.session_state:
            st.session_state["generated_sql"] = ""
            gen_sql = st.session_state["generated_sql"]
        if "question" not in st.session_state:
            st.session_state["question"] = ""
        if "df" not in st.session_state:
            st.session_state["df"] = pd.DataFrame()
        if "pl_code" not in st.session_state:
            st.session_state["pl_code"] = ""
        if "pl_instr_area" not in st.session_state:
            st.session_state["pl_instr_area"] = ""
            
            
        modal = Modal(key="ERD",title="Schema Viewer", max_width=1600)
        
        # Create tabs
        tabs = ["Ask/Train", "Retrieve/Modify Training Data"]
    
        with st.sidebar:
            st.title(self.title)
            selected_tab = st.radio("Select a tab", tabs)
            open_modal = st.button(label="Schema Viewer")
            # e_image = base64.b64encode(open(self.erd_path, "rb").read()).decode()
            # image = f"data:image/jpeg;base64,{e_image}"
            # clicked = clickable_images([image],
            #     titles=["Schema Viewer"],
            #         div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            #             img_style={"margin": "5px", "height": "200px"},
            #             )
            retrieve_train_data = st.download_button(label="Download Training Data",  
                    data = self.download_training_data(), 
                    file_name="training_data.csv", mime="text/csv")

            if open_modal:
            #if clicked > -1:
                #clicked = -1
                with modal.container():
                    #st.graphviz_chart(self.graph, use_container_width=True)
                    #st.components.v1.html(open("graph.svg", "r").read(), width=800, height=600)
                    #st.image(open("graph.svg", "r").read(), use_column_width='auto')
                    st.image(self.erd_path, use_column_width='auto')                        

#             if st.session_state["question"] and st.session_state["generated_sql"] and not st.session_state['df'].empty:
#                 plotly_code = generate_plotly_code_cached(question=st.session_state["question"], sql=st.session_state["generated_sql"], df=st.session_state['df'])
#                 pl_code = code_editor(plotly_code, lang="python")
#                 fig = generate_plot_cached(plotly_code, st.session_state['df'])
#                 st.plotly_chart(fig)
                
                
    ## Check already existing training data
        if selected_tab == "Retrieve/Modify Training Data":
            col1, col2= st.columns([1,1])
                
            with col1:
                st.write("Upload new training data")
                file_csv = st.file_uploader(label="",  type=['csv'])
                append = st.checkbox("Append to current training data?")
                if file_csv is not None:
                    self.upload_training_data(file_csv, append=append)
                    st.success("Training data uploaded successfully!")
                    train_data = self.vn.get_training_data()
                    st.info(f"Training data statistics: {train_data.shape}")
            with col2:
                with st.form("form_remove_train_data"):
                    rows_to_remove = st.text_input("Enter row numbers to remove (comma-separated)")
                    remove_train_data = st.form_submit_button("Remove row(s) from training data")
            
            st.subheader("Retrieve the model's training data")
            train_data = self.vn.get_training_data()
            st.info(f"Training data statistics: {train_data.shape[0]} rows")
            tr_df = st.dataframe(train_data)

            if remove_train_data:
                for id in rows_to_remove.split(","):
                    rid = train_data.loc[int(id),'id']
                    self.vn.remove_training_data(rid)
                st.success("Training data removed successfully!")
                tr_df.write(self.vn.get_training_data())


        elif selected_tab == "Ask/Train":
            
            
            st.subheader("Ask new questions to the model")
#             print(st.session_state)

            with st.form("add_to_train_data"):
                to_train_data = st.form_submit_button("Add to training data")
#                 print(to_train_data, st.session_state["question"], st.session_state["generated_sql"])
                if to_train_data and st.session_state["question"] and st.session_state["generated_sql"]:
                    self.vn.train(question=st.session_state["question"], sql=st.session_state["generated_sql"])
                    st.success("Training data added successfully!")
                elif to_train_data and not (st.session_state["question"] and st.session_state["generated_sql"]):
                    st.warning("Please enter both a question and an SQL query")
                    #st.stop()

            with st.form("ask_generate_sql"): 
                st.session_state.question = st.text_area("Ask a question", key="q_area")
                generate_sql = st.form_submit_button("Generate SQL!", help="Submits a question to the model and returns with a SQL query.")
                if generate_sql:
                    with st.spinner('Wait for it...'):
                        gen_sql = self.vn.generate_sql(question=st.session_state.question)
                        st.session_state["generated_sql"] = gen_sql
#                 st.write("FDFDFDF", st.session_state["generated_sql"])
                #count players in each division

                edit_sql = {}
                sql_col1, sql_col2 = st.columns(2)
                with sql_col1:
#                     edit_sql = st.text_area("SQL Query Editor", key="sql_area", value=gen_sql)
                    edit_sql = code_editor(st.session_state["generated_sql"], lang="sql", buttons=btn_settings_editor_btns)
                    if edit_sql['type'] == "submit" and len(edit_sql['text']) != 0:
                        st.session_state["generated_sql"] = edit_sql['text']
                
                

                submit_sql = st.form_submit_button("Submit SQL!", help="Submits a query to the database server and returns with a DataFrame.")
#                 st.write("FDFDFDF", st.session_state["generated_sql"], "\n-----------\n",edit_sql)
                                
                if submit_sql:
#                     print("FDFDFDF", st.session_state["code"], "\n-----------\n",edit_sql)
                    with st.spinner('Wait for it...'):
                            st.session_state['df'] = self.vn.run_sql(st.session_state["generated_sql"])
            
                with sql_col2:
                    if not st.session_state["df"].empty:
                        st.write(st.session_state['df'])
                                    
                generate_plotly = st.form_submit_button("Generate Plotly!", help="Creates a plotly code for the DataFrame.")
                
                plotting_question = st.session_state["question"] 
                if st.session_state["pl_instr_area"]:
                    plotting_question = st.session_state["question"] + "; " + st.session_state["pl_instr_area"]
                    
                if generate_plotly and st.session_state["generated_sql"] and st.session_state["question"] and not st.session_state["df"].empty:
                    st.session_state["pl_code"] = generate_plotly_code_cached(question=plotting_question, sql=st.session_state["generated_sql"], df=st.session_state['df'])
                    
                st.write(plotting_question)
                edit_pl_code = {}
                pl_code_col1, pl_code_col2 = st.columns(2)
                with pl_code_col1:                    
                    edit_pl_code = code_editor(st.session_state["pl_code"], lang="python", buttons=btn_settings_editor_btns)

                with pl_code_col2:
                    additional_plotting_instructions = st.text_area("Additional plotting context",  key="pl_instr_area")  
                
#                 st.write("PLCODE",edit_pl_code, st.session_state["pl_code"])
                    
                if edit_pl_code['type'] == "submit" and len(edit_pl_code['text']) != 0:
                    st.session_state["pl_code"] = edit_pl_code['text']
                    
                generate_plotly = st.form_submit_button("Submit Plotly!", help="Creates a plotly figure from the DataFrame.")
                if generate_plotly and st.session_state["pl_code"]:
                    fig = generate_plot_cached(st.session_state["pl_code"], st.session_state['df'])
                    st.plotly_chart(fig)
#             st.write(st.session_state)

