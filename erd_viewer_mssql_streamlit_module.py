"""
Created By:    Cristian Scutaru and David Visontai
Creation Date: Sep 2023
Company:       XtractPro Software
"""
import re
import streamlit as st
from streamlit_modal import Modal
import pymssql
import graphviz

class Session():
    def __init__(self, **kwargs):
        if 'database' in kwargs:
            self.database = kwargs['database']
        if 'connection' in kwargs:
            self.connection = kwargs['connection']
            self.cursor = self.connection.cursor()
        else:
            print("No connection provided")
        
        # We need a cursor that contains the state of our query(s)
        self.cursor = self.connection.cursor()
        
    def close(self):
        self.cursor.close()
        self.connection.close()

    def __del__(self):
        self.close()

    def sql(self, query):
        self.cursor.execute(query)
        return self.cursor


class Theme:
    def __init__(self, color, fillcolor, fillcolorC,
            bgcolor, icolor, tcolor, style, shape, pencolor, penwidth):
        self.color = color
        self.fillcolor = fillcolor
        self.fillcolorC = fillcolorC
        self.bgcolor = bgcolor
        self.icolor = icolor
        self.tcolor = tcolor
        self.style = style
        self.shape = shape
        self.pencolor = pencolor
        self.penwidth = penwidth


class Table:
    def __init__(self, name, comment):
        self.name = name
        self.comment = comment if comment is not None and comment != 'None' else ''
        self.label = None

        self.columns = []           # list of all columns
        self.uniques = {}           # dictionary with UNIQUE constraints, by name + list of columns
        self.pks = []               # list of PK columns (if any)
        self.fks = {}               # dictionary with FK constraints, by name + list of FK columns


    @classmethod
    def getClassName(cls, name, useUpperCase, withQuotes=True):
        if re.match("^[A-Z_0-9]*$", name) == None:
            return f'"{name}"' if withQuotes else name
        return name.upper() if useUpperCase else name.lower()


    def getName(self, useUpperCase, withQuotes=True):
        return Table.getClassName(self.name, useUpperCase, withQuotes)


    def getColumn(self, name):
        for column in self.columns:
            if column.name == name:
                return column
        return None


    def getUniques(self, name, useUpperCase):
        constraint = self.uniques[name]
        uniques = [column.getName(useUpperCase) for column in constraint]
        ulist = ", ".join(uniques)

        if useUpperCase:
            return (f',\n  CONSTRAINT {Table.getClassName(name, useUpperCase)}\n'
                + f"    UNIQUE ({ulist})")
        return (f',\n  constraint {Table.getClassName(name, useUpperCase)}\n'
            + f"    unique ({ulist})")


    def getPKs(self, useUpperCase):
        pks = [column.getName(useUpperCase) for column in self.pks]
        pklist = ", ".join(pks)
        pkconstraint = self.pks[0].pkconstraint

        if useUpperCase:
            return (f',\n  CONSTRAINT {Table.getClassName(pkconstraint, useUpperCase)}\n'
                + f"    PRIMARY KEY ({pklist})")
        return (f',\n  constraint {Table.getClassName(pkconstraint, useUpperCase)}\n'
            + f"    primary key ({pklist})")


    def getFKs(self, name, useUpperCase):
        constraint = self.fks[name]
        pktable = constraint[0].fkof.table

        fks = [column.getName(useUpperCase) for column in constraint]
        fklist = ", ".join(fks)
        pks = [column.fkof.getName(useUpperCase) for column in constraint]
        pklist = ", ".join(pks)

        if useUpperCase:
            return (f"ALTER TABLE {self.getName(useUpperCase)}\n"
                + f"  ADD CONSTRAINT {Table.getClassName(name, useUpperCase)}\n"
                + f"  ADD FOREIGN KEY ({fklist})\n"
                + f"  REFERENCES {pktable.getName(useUpperCase)} ({pklist});\n\n")
        return (f"alter table {self.getName(useUpperCase)}\n"
            + f"  add constraint {Table.getClassName(name, useUpperCase)}\n"
            + f"  add foreign key ({fklist})\n"
            + f"  references {pktable.getName(useUpperCase)} ({pklist});\n\n")


    # outputs a CREATE TABLE statement for the current table
    def getCreateTable(self, useUpperCase):
        if useUpperCase:
            s = f"CREATE OR REPLACE TABLE {self.getName(useUpperCase)} ("
        else:
            s = f"create or replace table {self.getName(useUpperCase)} ("
        
        first = True
        for column in self.columns:
            if first: first = False
            else: s += ","
            s += column.getCreateColumn(useUpperCase)

        if len(self.uniques) > 0:
            for constraint in self.uniques:
                s += self.getUniques(constraint, useUpperCase)
        if len(self.pks) >= 1:
            s += self.getPKs(useUpperCase)
        
        s += "\n)"
        if self.comment != '':
            comment = self.comment.replace("'", "''")
            s += f" comment = '{comment}'" if not useUpperCase else f" COMMENT = '{comment}'"
        return s + ";\n\n"


    def getDotShape(self, theme, showColumns, showTypes, useUpperCase):
        fillcolor = theme.fillcolorC if showColumns else theme.fillcolor
        colspan = "2" if showTypes else "1"
        tableName = self.getName(useUpperCase, False)
        s = (f'  {self.label} [\n'
            + f'    fillcolor="{fillcolor}" color="{theme.color}" penwidth="1"\n'
            + f'    label=<<table style="{theme.style}" border="0" cellborder="0" cellspacing="0" cellpadding="1">\n'
            + f'      <tr><td bgcolor="{theme.bgcolor}" align="center"'
            + f' colspan="{colspan}"><font color="{theme.tcolor}"><b>{tableName}</b></font></td></tr>\n')

        if showColumns:
            for column in self.columns:
                name = column.getName(useUpperCase, False)
                if column.ispk: name = f"<u>{name}</u>"
                if column.fkof != None: name = f"<i>{name}</i>"
                if column.nullable: name = f"{name}*"
                if column.identity: name = f"{name} I"
                if column.isunique: name = f"{name} U"
                datatype = column.datatype
                if useUpperCase: datatype = datatype.upper()

                if showTypes:
                    s += (f'      <tr><td align="left"><font color="{theme.icolor}">{name}&nbsp;</font></td>\n'
                        + f'        <td align="left"><font color="{theme.icolor}">{datatype}</font></td></tr>\n')
                else:
                    s += f'      <tr><td align="left"><font color="{theme.icolor}">{name}</font></td></tr>\n'

        return s + '    </table>>\n  ]\n'


    def getDotLinks(self, theme):
        s = ""
        for constraint in self.fks:
            #print(constraint)
            fks = self.fks[constraint]
            fk1 = fks[0]
            dashed = "" if not fk1.nullable else ' style="dashed"'
            arrow = "" if fk1.ispk and len(self.pks) == len(fk1.fkof.table.pks) else ' arrowtail="crow"'
            s += (f'  {self.label} -> {fk1.fkof.table.label}'
                + f' [ penwidth="{theme.penwidth}" color="{theme.pencolor}"{dashed}{arrow} ]\n')
        return s


class Column:
    def __init__(self, table, name, comment):
        self.table = table
        self.name = name
        self.comment = comment if comment is not None and comment != 'None' else ''
        self.nullable = True
        self.datatype = None        # with (length, or precision/scale)
        self.identity = False

        self.isunique = False
        self.ispk = False
        self.pkconstraint = None
        self.fkof = None            # points to the PK column on the other side


    def getName(self, useUpperCase, withQuotes=True):
        return Table.getClassName(self.name, useUpperCase, withQuotes)

    # outputs the column definition in a CREATE TABLE statement, for the parent table
    def getCreateColumn(self, useUpperCase):
        nullable = "" if self.nullable or (self.ispk and len(self.table.pks) == 1) else " not null"
        if useUpperCase: nullable = nullable.upper()
        identity = "" if not self.identity else " identity"
        if useUpperCase: identity = identity.upper()
        pk = ""     # if not self.ispk or len(self.table.pks) >= 2 else " primary key"
        if useUpperCase: pk = pk.upper()
        datatype = self.datatype
        if useUpperCase: datatype = datatype.upper()
        
        comment = self.comment.replace("'", "''")
        if comment != '': comment = f" COMMENT '{comment}'" if useUpperCase else f" comment '{comment}'"

        return f"\n  {self.getName(useUpperCase)} {datatype}{nullable}{identity}{pk}{comment}"

class ERDViewer():
    def __init__(self, session):
        self.session = session
        self.themes = self.getThemes()
        self.theme = self.themes["Common Gray"]
        self.showColumns = True
        self.showTypes = True
        self.useUpperCase = False
        
    # @st.cache_resource(show_spinner="Reading metadata...")
    def importMetadata(self):
        tables = {}
        #suffix = f"in schema {Table.getClassName(database, False)}"

        # get tables
        query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
        #query = "SELECT t.name AS TableName, ep.value AS TableComment FROM sys.tables t LEFT JOIN sys.extended_properties ep ON ep.major_id = t.object_id AND ep.minor_id = 0 AND ep.name = 'MS_Description'"
        r_tables = self.session.sql(query).fetchall()
        for r_table in r_tables:
            if r_table[0] == "sysdiagrams": continue
            tableName = r_table[0]
            table = Table(tableName, "No Comment")
            tables[tableName] = table
            table.label = f"n{len(tables)}"

        # get table columns
        query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS" # WHERE TABLE_NAME = {tableName}"
        r_cols = self.session.sql(query).fetchall()
        #print(results)
        for col in r_cols:
            tableName = col[2]
            colName = col[3]
            if tableName in tables:
                table = tables[tableName]

                #name = str(row["column_name"])
                column = Column(table, colName, "No comment")
                table.columns.append(column)

                #column.identity = str(row["autoincrement"]) != ''
                column.nullable = bool(col[6])
                column.datatype = col[7]
                if col[8]: # length
                    column.datatype += f"({col[8]})"
        
        for tableName in tables.keys():
            # get UNIQUE constraints
            # query = f"EXEC sp_pkeys {tableName}"
            # r_pkeys = self.session.sql(query).fetchall()
            # #print("Pkeys", r_pkeys)
            # for pkeys in r_pkeys:
            #     constraint = pkeys[-1]
            #     if constraint not in table.uniques:
            #         table.uniques[constraint] = []
            #     table.uniques[constraint].append(column)
            #     column.isunique = True

            # get PKs
            query = f"EXEC sp_pkeys {tableName}"
            r_pkeys = self.session.sql(query).fetchall()
            #print("Pkeys\n", r_pkeys)
            for pkey in r_pkeys: 
                column.ispk = True
                column.pkconstraint = pkey[-1]
                pos = pkey[-2] - 1
                table.pks.insert(pos, column)

            # get FKs
            query = f"EXEC sp_fkeys {tableName}"
            r_fkeys = self.session.sql(query).fetchall()
            for fkeys in r_fkeys:
                #print("Fkeys\n",fkeys)
                pktableName = fkeys[2]
                fktableName = fkeys[6]
                if pktableName in tables and fktableName in tables:
                    pktable = tables[pktableName]
                    pkcolumn = pktable.getColumn(fkeys[3])
                    fktable = tables[fktableName]
                    fkcolumn = fktable.getColumn(fkeys[7])

                        # add a constraint (if not there) with the current FK column
                        # if str(row["pk_schema_name"]) == str(row["fk_schema_name"]):
                    constraint = str(fkeys[11])
                    if constraint not in fktable.fks:
                        fktable.fks[constraint] = []
                    fktable.fks[constraint].append(fkcolumn)

                    fkcolumn.fkof = pkcolumn
                    #print(f"{fktable.name}.{fkcolumn.name} -> {pktable.name}.{pkcolumn.name}")
            
        return tables

    #@st.cache_resource(show_spinner="Reading metadata...")
    def createGraph(self, tables):
        s = ('digraph {\n'
            + '  graph [ rankdir="LR" bgcolor="#ffffff" ]\n'
            + f'  node [ style="filled" shape="{self.theme.shape}" gradientangle="180" ]\n'
            + '  edge [ arrowhead="none" arrowtail="none" dir="both" ]\n\n')

        for name in tables:
            s += tables[name].getDotShape(self.theme, self.showColumns, self.showTypes, self.useUpperCase)
        s += "\n"
        for name in tables:
            s += tables[name].getDotLinks(self.theme)
        s += "}\n"
        return s



    def getThemes(self):
        return {
            "Common Gray": Theme("#6c6c6c", "#e0e0e0", "#f5f5f5",
                "#e0e0e0", "#000000", "#000000", "rounded", "Mrecord", "#696969", "1"),
            "Blue Navy": Theme("#1a5282", "#1a5282", "#ffffff",
                "#1a5282", "#000000", "#ffffff", "rounded", "Mrecord", "#0078d7", "2"),
            #"Gradient Green": Theme("#716f64", "#008080:#ffffff", "#008080:#ffffff",
            #    "transparent", "#000000", "#000000", "rounded", "Mrecord", "#696969", "1"),
            #"Blue Sky": Theme("#716f64", "#d3dcef:#ffffff", "#d3dcef:#ffffff",
            #    "transparent", "#000000", "#000000", "rounded", "Mrecord", "#696969", "1"),
            "Common Gray Box": Theme("#6c6c6c", "#e0e0e0", "#f5f5f5",
                "#e0e0e0", "#000000", "#000000", "rounded", "record", "#696969", "1")
        }
            

    # def getSchema(self):
    #     names = []
    #     if database != "":
    #         query = f"show schemas in database {Table.getClassName(database, False)}"
    #         results = session.sql(query).fetchall()
    #         for row in results:
    #             schemaName = str(row["name"])
    #             if schemaName != "INFORMATION_SCHEMA":
    #                 names.append(schemaName)
    #     sel = 0 if "PUBLIC" not in names else names.index("PUBLIC")
    #     return st.sidebar.selectbox('Schema', tuple(names), index=sel, 
    #         help="Select a schema for the current database")

    def save_graph_to_html(self, graph):
            output_file = 'graph'
            #graph.format = 'html'
            graphviz.Source(graph).render(output_file, format='svg')
            graphviz.Source(graph).render(output_file, format='png')
            #graphviz.render(graph, outfile=output_file)
            print(f"Graph saved to {output_file}")

    def get_graph(self):
        # schema = getSchema(database)
        
        # with st.spinner('Reading metadata...'):
        #     tables = importMetadata(self.session.database)
        # if len(tables) == 0:
        #     st.write("Found no tables in the current database and schema.")
        # else:
        #     with st.spinner('Generating diagram and script...'):
        #         graph_name = f"{self.session.database}_graph.dot"
        #         try:
        #             with open(graph_name, "r") as f:
        #                 graph = f.read()
        #             print(f"graph for Database {self.session.database} found. Reading it ...")
        #         except:
        #             print(f"graph for Database {self.session.database} not found\n Creating it")
        #             #graph = createGraph(tables, themes[theme], showColumns, showTypes, useUpperCase)
        #             graph = createGraph(tables, self.themes[theme])
        #             with open(graph_name, "w") as f:
        #                 f.write(graph)
                
        #         #st.graphviz_chart(graph, use_container_width=True)
        tables = self.importMetadata()
        graph_name = f"{self.session.database}_graph.dot"
        try:
            with open(graph_name, "r") as f:
                graph = f.read()
            print(f"graph for Database {self.session.database} found. Reading it ...")
            self.save_graph_to_html(graph)    
        except:
            print(f"graph for Database {self.session.database} not found\n Creating it")
            #graph = createGraph(tables, themes[theme], showColumns, showTypes, useUpperCase)
            graph = self.createGraph(tables)
            with open(graph_name, "w") as f:
                f.write(graph)
            self.save_graph_to_html(graph)    
            
        return graph
                
        


        
        
        