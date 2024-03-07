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
from st_keyup import st_keyup
from pandas.errors import DatabaseError
import base64

from erd_viewer_mssql_streamlit_module import *


@st.cache_data(show_spinner="Fetching training data...")
def get_training_data(counter) -> pd.DataFrame:
    return vn.get_training_data()

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached(question):
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    return vn.generate_sql(question=question)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    return vn.run_sql(sql=sql)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, df):
    return vn.generate_followup_questions(question=question, df=df)

def upload_training_data(file_csv, append=True) -> None:
    df = pd.read_csv(file_csv)
    if not append and vn.get_training_data().shape[0] > 0:
        for id in vn.get_training_data().id:
            vn.remove_training_data(id)
    for idx, row in df.iterrows():
        if row['training_data_type'] == "sql":
            vn.train(question=row['question'], sql=row['content'])

def download_training_data() -> None:
    training_data = vn.get_training_data()
    if training_data.shape[0]:
        training_data.drop(axis=1, columns=['id'], inplace=True)
    return training_data.to_csv(index=False)            

def new_run_sql(sql, session):
    return pd.read_sql_query(sql, session.connection)

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


def run(session, token_path, model_name, title, erd_path):
    
    counter=0
    
    with open(token_path) as f:
        token = f.read().strip()
    vn.set_api_key(token)
    vn.set_model(model_name)
    vn.run_sql = new_run_sql

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
        st.title(title)
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
                data = download_training_data(), 
                file_name="training_data.csv", mime="text/csv")
        
        if selected_tab == "Ask/Train":
            show_similar_questions = st.checkbox("Show similar questions from the training data?")
            show_followup_questions = st.checkbox("Show Followup questions?")
            with st.form("add_to_train_data"):
                to_train_data = st.form_submit_button("Add to training data")
    #                 print(to_train_data, st.session_state["question"], st.session_state["generated_sql"])
                if to_train_data and st.session_state["question"] and st.session_state["generated_sql"]:
                    vn.train(question=st.session_state["question"], sql=st.session_state["generated_sql"])
                    counter+=1
                    st.success("Training data added successfully!")
                elif to_train_data and not (st.session_state["question"] and st.session_state["generated_sql"]):
                    st.warning("Please enter both a question and an SQL query")
                #st.stop()


        if open_modal:
        #if clicked > -1:
            #clicked = -1
            with modal.container():
                #st.graphviz_chart(self.graph, use_container_width=True)
                #st.components.v1.html(open("graph.svg", "r").read(), width=800, height=600)
                #st.image(open("graph.svg", "r").read(), use_column_width='auto')
                st.image(erd_path, use_column_width='auto')                        

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
                upload_training_data(file_csv, append=append)
                counter+=1
                st.success("Training data uploaded successfully!")
                train_data = get_training_data(counter)
                st.info(f"Training data statistics: {train_data.shape}")
        with col2:
            with st.form("form_remove_train_data"):
                rows_to_remove = st.text_input("Enter row numbers to remove (comma-separated)")
                remove_train_data = st.form_submit_button("Remove row(s) from training data")

        st.subheader("Retrieve the model's training data")
        train_data = get_training_data(counter)
        st.info(f"Training data statistics: {train_data.shape[0]} rows")
        tr_df = st.dataframe(train_data)

        if remove_train_data:
            for id in rows_to_remove.split(","):
                rid = train_data.loc[int(id),'id']
                vn.remove_training_data(rid)
            st.success("Training data removed successfully!")
            tr_df.write(get_training_data(counter))


    elif selected_tab == "Ask/Train":

        train_data = get_training_data(counter)
        st.header("Ask new questions to the model")        

        if show_similar_questions:
            new_question = st_keyup("Ask a question", value=st.session_state.question, debounce=None)
            if new_question:
                filtered = train_data[train_data.question.str.lower().str.contains(new_question.lower(), na=False)]
            else:
                filtered = train_data

            st.write(len(filtered), "similar questions found")
            st.write(filtered)
        else:
            new_question = st.text_area("Ask a question", value=st.session_state.question, key="q_area")
            
        with st.form("generate_sql"): 

            if new_question:
                st.session_state.question = new_question

            generate_sql = st.form_submit_button("Generate SQL!", help="Submits a question to the model and returns with a SQL query.")
            if generate_sql:
                with st.spinner('Wait for it...'):
                    gen_sql = vn.generate_sql(question=st.session_state.question)
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
                        st.session_state['df'] = vn.run_sql(st.session_state["generated_sql"], session)

            with sql_col2:
                if not st.session_state["df"].empty:
                    st.write(st.session_state['df'])
                    
        if show_followup_questions and st.session_state.question and not st.session_state['df'].empty:
            st.write(generate_followup_cached(st.session_state.question, st.session_state['df']))

        with st.form("generate_plotly"): 
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

