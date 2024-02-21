import vanna as vn
import os

import pandas as pd
import psycopg2
from copy import deepcopy

import streamlit as st
from streamlit_javascript import st_javascript
from pandas.errors import DatabaseError

from erd_viewer_mssql_streamlit_module import *

# @st.cache_data(show_spinner="Fetching training data...")
# def _get_training_data(_vn) -> pd.DataFrame:
#     return _vn.get_training_data()

class StreamlitBackend():


    def __init__(self, **kwargs):
        self.df = None
        if "session" in kwargs:
            self.session = kwargs['session']
        if "title" in kwargs:
            self.title = kwargs['title']
        if "graph" in kwargs:
            self.graph = kwargs['graph']
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
        
    def run(self):
        # Init Streamlit
        st.set_page_config(layout="wide")
        modal = Modal(key="ERD",title="Schema Viewer", max_width=1600)
        
        # Create tabs
        tabs = ["Ask/Train", "Retrieve/Modify Training Data"]

        with st.sidebar:
            st.title(self.title)
            selected_tab = st.radio("Select a tab", tabs)
            open_modal = st.button(label="Schema Viewer")
            if open_modal:
                with modal.container():
                    #st.graphviz_chart(self.graph, use_container_width=True)
                    #st.components.v1.html(open("graph.svg", "r").read(), width=800, height=600)
                    #st.image(open("graph.svg", "r").read(), use_column_width='auto')
                    st.image("graph.png", use_column_width='auto')
                    #st.write("""<figure><embed type="image/svg+xml" src="graph.svg" /></figure>""", unsafe_allow_html=True, use_column_width='auto')
            
    ## Check already existing training data
        if selected_tab == "Retrieve/Modify Training Data":
            col1, col2= st.columns([1,1])
            with col1:
                st.write("Download training data in csv format")
                training_data = self.vn.get_training_data()
                if training_data.shape[0]:
                    training_data.drop(axis=1, columns=['id'], inplace=True)
                retrieve_train_data = st.download_button(label="Download",  data = training_data.to_csv(index=False), file_name="training_data.csv", mime="text/csv")
            with col2:
                append = st.checkbox("Append to current training data?")
                file_csv = st.file_uploader(label="Upload training data",  type=['csv'])
                if file_csv is not None:
                    self.upload_training_data(file_csv, append=append)
                    st.success("Training data uploaded successfully!")
                    train_data = self.vn.get_training_data()
                    st.info(f"Training data statistics: {train_data.shape}")
            with st.form("form0"):
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
            button_add_train_data = None
            col_add_train_data = None
            
            def form_callback():
                st.session_state.q_inp = st.session_state.q_area
                self.question = st.session_state.q_inp
                with out_col2:
                    with st.spinner('Wait for it...'):
                        self.sql = self.vn.generate_sql(question=self.question)
                        st.code(self.sql, language="sql")
                        st.session_state["generated_sql"] = self.sql

                        if not st.session_state.only_sql:
                            try:
                                answer = self.vn.run_sql(self.sql)
                            except DatabaseError as e:
                                st.error(e)
                                st.stop()
                                st.rerun()
                            st.write(answer)
                print(self.sql)
                st.session_state.new_sql = self.sql.replace("\n", " ")

            with st.form("form2"): 
                if "generated_sql" not in st.session_state:
                    st.session_state["generated_sql"] = ""

                input_col1, input_col2 = st.columns([1,1])
                out_col1, out_col2 = st.columns([1,1])

                sql_area = input_col1.text_area("SQL Query", value=st.session_state.generated_sql, key="sql_area")
                self.sql = input_col1.text_input("", value=st.session_state.sql_area.replace("\n", " "), key='new_sql')
                q_area = input_col2.text_area("Ask a question", key="q_area")
                self.question = input_col2.text_input("", value=st.session_state.q_area, key="q_inp", disabled=True)
                
                col1, col2= st.columns([1,1])
                with col1:
                    input_sql = st.form_submit_button("Submit SQL!", help="Submits a query and returns with a DataFrame.")
                with col2:
                    c1, c2, c3 = st.columns([1,1,2])
                    with c2:
                        only_sql = st.checkbox("Only SQL!", key="only_sql")
                    with c1:
                        ask = st.form_submit_button("Submit question!", help="creates a query and executes it if valid.", on_click=form_callback)
                    
                if input_sql:
                    with out_col1:
                        with st.spinner('Wait for it...'):
                            answer = self.vn.run_sql(self.sql)
                            st.write(answer)

                # if ask:
                #    this one filled up through a callback
            
            with st.form("add_to_train_data"):
                to_train_data = st.form_submit_button("Add to training data")
                if to_train_data and self.question and self.sql:
                    self.vn.train(question=self.question, sql=self.sql)
                    st.success("Training data added successfully!")
                elif not self.question or not self.sql:
                    st.error("Please enter both a question and an SQL query")
                    st.stop()


