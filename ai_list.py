import streamlit as st
# https://docs.streamlit.io/develop/quick-reference/cheat-sheet

st.set_page_config(layout="wide")

st.subheader("Roberto's- Artificial Intelligence Apps")
st.write('70 Apps by Roberto')
st.balloons()
st.snow()
st.toast("Loading...")


st.subheader("Machine Learning")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### MACHINE LEARNING
   {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/1_Introducci%C3%B3n%20a%20NumPy.ipynb",
    "text": "Practice of Machine Learning -> Introduction to NumPy",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/2_Introducci%C3%B3n%20a%20Pandas.ipynb",
    "text": "Practice of Machine Learning -> Introduction to Panda",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/3_Introducci%C3%B3n%20a%20Matplotlib.ipynb",
    "text": "Practice of Machine Learning -> Introduction to Matplotlib",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/4_Regresi%C3%B3n%20Lineal%20-%20Predicci%C3%B3n%20del%20coste%20de%20un%20incidente%20de%20seguridad.ipynb",
    "text": "Practice of Machine Learning -> Linear Regression: Cost of a security incident",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/5_Regresi%C3%B3n%20Log%C3%ADstica%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb",
    "text": "Practice of Machine Learning App -> Logistic Regression: SPAM Detection",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/6_Visualizaci%C3%B3n%20del%20conjunto%20de%20datos.ipynb",
    "text": "Practice of Machine Learning -> Visualization of the dataset",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/7_Divisi%C3%B3n%20del%20conjunto%20de%20datos.ipynb",
    "text": "Practice of Machine Learning -> Division of the data set",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/8_Preparaci%C3%B3n%20del%20conjunto%20de%20datos.ipynb",
    "text": "Practice of Machine Learning  -> Data set preparation",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/9_Creaci%C3%B3n%20de%20Transformadores%20y%20Pipelines%20personalizados.ipynb",
    "text": "Machine Learning  -> Creation of custom transformers and pipelines",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/10_Evaluaci%C3%B3n%20de%20resultados.ipynb",
    "text": "Practice of Machine Learning  -> Evaluation of results",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/11_Support%20Vector%20Machine%20-%20Detecci%C3%B3n%20de%20URLs%20maliciosas.ipynb",
    "text": "Practice of Machine Learning  -> Support Vector Machine (SVM)",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/12_%C3%81rboles%20de%20decisi%C3%B3n%20-%20Detecci%C3%B3n%20de%20malware%20en%20Android.ipynb",
    "text": "Practice of Machine Learning  -> Decision trees",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/13_Random%20Forests%20-%20Detecci%C3%B3n%20de%20Malware%20en%20Android.ipynb",
    "text": "Practice of Machine Learning  -> Random Forest",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/14_T%C3%A9cnicas%20de%20selecci%C3%B3n%20de%20caracter%C3%ADsticas.ipynb",
    "text": "Practice of Machine Learning  -> Features selection",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/15_PCA%20-%20Extracci%C3%B3n%20de%20caracter%C3%ADsticas.ipynb",
    "text": "Practice of Machine Learning  -> Principal Component Analysis (PCA)",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/16_T%C3%A9cnicas%20de%20selecci%C3%B3n%20del%20modelo.ipynb",
    "text": "Practice of Machine Learning  -> Model selection",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/17_KMEANS%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb",
    "text": "Machine Learning  -> KMEANS - Detection of fraudulent bank transactions",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/18_DBSCAN%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb",
    "text": "Machine Learning  -> DBSCAN - Detection of fraudulent bank transactions",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/19_Naive%20Bayes%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb",
    "text": "Practice of Machine Learning  -> Naive Bayes - SPAM Detection",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/20_Distribuci%C3%B3n%20Gaussiana%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb",    
    "text": "Practice of Machine Learning  -> Gaussian Distribution",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/21_Isolation%20Forest%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb",
    "text": "Practice of Machine Learning  -> Isolation Forest",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/22_Redes%20Neuronales%20Artificiales%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb",
    "text": "Practice of Machine Learning  -> Artificial Neural Networks - Detection of fraudulent bank transactions",
    "fondo": "darkmagenta" }, # 
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 
  
st.subheader("Generative AI")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### GENERATIVE AI
   {"href": "https://github.com/web-roberto/generative_ai/tree/main/seccion4-Prompts",
    "text": "Generative AI and LLM -> Prompt Engineering documentation",
    "fondo": "yellowgreen" }, # seccion4-Prompts
    {"href": "",
    "text": "Generative AI and Large Language Model (LLM) Apps-> Practice with DeepSeek",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "Generative AI and LLM) Apps-> Practice with Claude",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "Generative AI and LLM Apps-> Practice with Gemini",
    "fondo": "red" }, # 
    {"href": "",
    "text": "Generative AI and LLM Apps-> email management and automated response",
    "fondo": "yellowgreen" }, # 75
    {"href": "",
    "text": "Generative AI and LLM Apps-> Chatbot",
    "fondo": "darkmagenta" }, # 80
    {"href": "",
    "text": "Generative AI and LLM Apps-> malicious email detection with ChatGPT and GPT",
    "fondo": "blue" }, # 92
    {"href": "",
    "text": "Generative AI and LLM Apps-> Connecting my GPTs to the real world: intelligent URL analysis",
    "fondo": "red" }, # 95
    {"href": "",
    "text": "Generative AI and LLM Apps-> Implementing a Chatbot with ChatGPT and GPT",
    "fondo": "yellowgreen" }, # 97
    {"href": "",
    "text": "Generative AI and LLM Apps-> Integration of Chatbot with external services",
    "fondo": "darkmagenta" }, # 99
    {"href": "",
    "text": "Generative AI, LLM and Recurrent Neural Networks. Apps-> Integration of Chatbot with external service",
    "fondo": "blue" }, # 99
    {"href": "",
    "text": "Generative AI, LLM and Recurrent Neural Networks. Apps-> Text generation with LSTM",
    "fondo": "red" }, # 106
    {"href": "",
    "text": "Generative AI, LLM and Transformers. Apps-> Text Generation with Transformers",
    "fondo": "yellowgreen" }, # 121
    {"href": "",
    "text": "Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuned LLM (FLAN-T5 by Google)",
    "fondo": "darkmagenta" }, # 129
    {"href": "",
    "text": "Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuned of an LLM",
    "fondo": "blue" }, # 134
    {"href": "",
    "text": "Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuning of GPT and ChatGPT to call external functions",
    "fondo": "red" }, # 141
    {"href": "",
    "text": "Generative AI, LLM, Pre-training and Fine-Tunning, PEFT (Parameter Efficient Fine-tunning), LoRA (Low-Rank Adaption. Apps-> PEFT of LLAMA2 (Meta) with QLoRA",
    "fondo": "yellowgreen" }, # 146, 147
    {"href": "",
    "text": "Generative AI, Fine-Tunning, RLHF (Reinforcement Learning from Human Feeback) and PPO (Proximal Policy Optimization). Apps-> RLHF of LLAMA2 (TinyLlama)",
    "fondo": "darkmagenta" }, # 155-157
    {"href": "",
    "text": "Generative AI, LLM, GANs (Generative Adversarial Networks). Apps-> To change a person's face in a video",
    "fondo": "blue" }, # 163
    {"href": "",
    "text": "Generative AI, LLM, Diffusion Models, CLIP (Contrastive Language-Image Pre-training). Apps-> Automatic Generations fo Images. DALL-E 3",
    "fondo": "red" }, # 169
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 

# GENERATIVE AI
# https://github.com/web-roberto/generative_ai/tree/main/seccion4-Prompts
#st.error("Generative AI and LLM -> Prompt Engineering documentation") # seccion4-Prompts
#st.error("Generative AI and Large Language Model (LLM) Apps-> Practice with DeepSeek")  # ...
#st.success("Generative AI and LLM) Apps-> Practice with Claude")  # ...
#st.warning("Generative AI and LLM Apps-> Practice with Gemini")  # ...
#st.info("Generative AI and LLM Apps-> email management and automated response")  # 75
#st.error("Generative AI and LLM Apps-> Chatbo")  # 80
#st.success("Generative AI and LLM Apps-> malicious email detection with ChatGPT and GPT")  # 92
#st.warning("Generative AI and LLM Apps-> Connecting my GPTs to the real world: intelligent URL analysis")  # 95
#st.info("Generative AI and LLM Apps-> Implementing a Chatbot with ChatGPT and GPT")  # 97
#st.error("Generative AI and LLM Apps-> Integration of Chatbot with external services")  # 99
#st.success("Generative AI, LLM and Recurrent Neural Networks. Apps-> Integration of Chatbot with external service")  # 99
#st.warning("Generative AI, LLM and Recurrent Neural Networks. Apps-> Text generation with LSTM")  #106
#st.info("Generative AI, LLM and Transformers. Apps-> Text Generation with Transformers")  # 121
#st.error("Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuned LLM (FLAN-T5 by Google)")  # 129
#st.success("Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuned of an LLM")  # 134
#st.warning("Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuning of GPT and ChatGPT to call external functions")  # 141
#st.info("Generative AI, LLM, Pre-training and Fine-Tunning, PEFT (Parameter Efficient Fine-tunning), LoRA (Low-Rank Adaption. Apps-> PEFT of LLAMA2 (Meta) with QLoRA")  # 146,147
#st.error("Generative AI, Fine-Tunning, RLHF (Reinforcement Learning from Human Feeback) and PPO (Proximal Policy Optimization). Apps-> RLHF of LLAMA2 (TinyLlama)")  # 155-157
#st.success("Generative AI, LLM, GANs (Generative Adversarial Networks). Apps-> To change a person's face in a video")  # 163
#st.warning("Generative AI, LLM, Diffusion Models, CLIP (Contrastive Language-Image Pre-training). Apps-> Automatic Generations fo Images. DALL-E 3")  # 169


st.subheader("Deep Learning")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### DEEP LEARNING
   {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
    {"href": "",
    "text": "",
    "fondo": "darkmagenta" }, # 
    {"href": "",
    "text": "",
    "fondo": "blue" }, # 
    {"href": "",
    "text": "",
    "fondo": "red" }, # 
    {"href": "",
    "text": "",
    "fondo": "yellowgreen" }, # 
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 


# DEEP LEARNING
st.success("Deep Learning: M-P NEURON App -> Implementation")
st.warning("Deep Learning: M-P NEURON App -> Diagnosis of breast cancer with one M-P neuron")
st.info("Deep Learning: PRECEPTRON App -> Visualization of the perceptron decision boundary ")
st.error("Deep Learning: PRECEPTRON App -> Classification of Images")
st.warning("Deep Learning: PRECEPTRON MULTILAYER App -> Decision boundary of the multilayer perceptron")
st.info("Deep Learning: loss and optimization functions. App -> Classification of images of handwritten digits")
st.error("Deep Learning: DEEP ARTIFICIAL NEURONAL NETWORK App -> Audio Classification")
st.success("Deep Learning: KERAS App -> Implementation of Artificial Neuronal Network")
st.warning("Deep Learning: KERAS App -> Training of a Artificial Neuronal Network")
st.info("Deep Learning: KERAS App -> Decision limit of a deep artificial neural network")
st.error("Deep Learning: KERAS App -> Audio Classification with Keras")
st.success("Deep Learning: KERAS App -> Regression with Keras")
st.warning("Deep Learning: Activation Functions App -> Show activation functions")
st.info("Deep Learning: Activation Functions App -> No use of activation functions")
st.error("Deep Learning: Activation Functions App -> Feelings classification")
st.success("Deep Learning: Optimization Functions App -> Mini-Batch Gradient Descent")
st.warning("Deep Learning: Optimization Functions App -> Stochastic Gradient Descent")
st.info("Deep Learning: Optimization Functions App -> Exponentially Weighted Moving Average")
st.error("Deep Learning: Optimization Functions App -> Momentum Gradient Descent")
st.success("Deep Learning: Optimization Functions App -> RMSprop")
st.warning("Deep Learning: Optimization Functions App -> Adam Optimization")
st.info("Deep Learning: Hyperparams App -> Selection of Hyperparams and text classification")
st.error("Deep Learning:Optimization Functions App -> Mini-Batch Gradient Descent")
st.warning("Deep Learning: TENSORFLOW App -> Tensor and operations")  
st.warning("Deep Learning: TENSORFLOW App -> Create a custom loss function")  
st.info("Deep Learning: TENSORFLOW App -> Create a custom component")  
st.error("Deep Learning: TENSORFLOW App -> Create a custom layer")  


  
st.subheader("AI with LangChain")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### LANGCHAIN
    {"href": "",
    "text": "### LANGCHAIN ######",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion4",
    "text": "LangChain App -> Feelings Analysis",
    "fondo": "darkmagenta" }, # -28-39 seccion4
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion4",
    "text": "LangChain App -> System for Evaluating Curriculums and Candidates with Generative AI",
    "fondo": "blue" }, # 46,48,49  seccion4
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion5",
    "text": "LangChain and RAG(Retrieval Augmented Generation) App -> Legal Assistance and Contract Evaluation System",
    "fondo": "red" }, #  60s seccion5
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion6",
    "text": "LangGraph App -> Meeting Processing System",
    "fondo": "yellowgreen" }, # seccion6
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion6",
    "text": "LangGraph: Human-in-the-Loop, Checkpoints App -> Helpdesk with AI, Langgraph and RAG",
    "fondo": "darkmagenta" }, # seccion6
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion7",
    "text": "LangGraph: Memory and Context Management. APP -> Vectorial Memoroy with Langgraph",
    "fondo": "blue" }, # seccion7
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion7",
    "text": "LangGraph: Memory and Context Management. APP -> Multi-user Chat with Advanced Memory",
    "fondo": "red" }, # 103-120 seccion6
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion8",
    "text": "AI Agents and external tools. Apps -> Multi-Agent Security Operations Center",
    "fondo": "yellowgreen" }, # 135 seccion8
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion8",
    "text": "AI Agents and external tools. Apps ->Internet Search with LLM and Tavily",
    "fondo": "darkmagenta" }, # 138 seccion8
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion8",
    "text": "AI Agents and external tools. Apps -> Cybersecurity Analisys with LLM and VirusTotal",
    "fondo": "blue" }, #  seccion8
    {"href": "https://github.com/web-roberto/langchain/tree/main/seccion8",
    "text": "AI Agents and external tools. Apps -> Tasks Implementation of Thread Intelligence",
    "fondo": "red" }, # 143-147 seccion8

]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True)


st.subheader("N8N Comming")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### N8N
   {"href": " N8N",
    "text": " N8N",
    "fondo": "yellowgreen" }, # 
    {"href": " N8N",
    "text": " N8N",
    "fondo": "darkmagenta" }, # 
    {"href": " N8N",
    "text": " N8N",
    "fondo": "blue" }, # 
    {"href": " N8N",
    "text": " N8N",
    "fondo": "red" }, # 
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 


st.subheader("MCP Comming")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### MCP
   {"href": "MCP",
    "text": "MCP",
    "fondo": "yellowgreen" }, # 
    {"href": "MCP",
    "text": "MCP",
    "fondo": "darkmagenta" }, # 
    {"href": "MCP",
    "text": "MCP",
    "fondo": "blue" }, # 
    {"href": "MCP",
    "text": "MCP",
    "fondo": "red" }, # 
    {"href": "MCP",
    "text": "MCP",
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 



