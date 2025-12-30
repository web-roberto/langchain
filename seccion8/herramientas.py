from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import streamlit as st

python_repl = PythonREPL()

tool = Tool(
    name="Python_repl",
    func=python_repl.run,
    description="Ejecuta codigo en Python en un interprete para calculos o logica matematica."
)

output = tool.run("st.write(2+2)")
st.write(output)
