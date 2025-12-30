import streamlit as st
# https://docs.streamlit.io/develop/quick-reference/cheat-sheet

st.set_page_config(layout="wide")

st.subheader("Roberto's- Artificial Intelligence Apps")
st.write('70 Apps by Roberto')
st.balloons()
st.snow()
st.toast("Loading...")


col1, col2, col3= st.columns([9,2,2], gap="small", vertical_alignment="top", border=True, width="stretch")

with col1:
    st.success("⭐⭐⭐ Top App ->  MULTICHAT MULTIUSER -> Tecnologies....")
    st.success("LangChain App -> Feelings Analysis") #  -28-39 seccion4
    st.warning("LangChain App -> System for Evaluating Curriculums and Candidates with Generative AI") #  46,48,49  seccion4
    st.info("LangChain and RAG(Retrieval Augmented Generation) App -> Legal Assistance and Contract Evaluation System") #  60s seccion5
    st.error("LangGraph App -> Meeting Processing System") # seccion6
    st.success("LangGraph: Human-in-the-Loop, Checkpoints App -> Helpdesk with AI, Langgraph and RAG ")  # seccion6
    st.warning("LangGraph: Memory and Context Management. APP -> Vectorial Memoroy with Langgraph")  # seccion7
    st.info("LangGraph: Memory and Context Management. APP -> Multi-user Chat with Advanced Memory")  # 103-120 seccion6
    st.error("AI Agents and external tools. Apps -> Multi-Agent Security Operations Center")  # 135 seccion8
    st.success("AI Agents and external tools. Apps ->Internet Search with LLM and Tavily ")  # 138 seccion8
    st.warning("AI Agents and external tools. Apps -> Cybersecurity Analisys with LLM and VirusTotal")  # seccion8
    st.info("AI Agents and external tools. Apps -> Tasks Implementation of Thread Intelligence")  # 143-147 seccion8
    
    st.error("Generative AI and Large Language Model (LLM) Apps-> Practice with DeepSeek")  # ...
    st.success("Generative AI and LLM) Apps-> Practice with Claude")  # ...
    st.warning("Generative AI and LLM Apps-> Practice with Gemini")  # ...
    st.info("Generative AI and LLM Apps-> email management and automated response")  # 75
    st.error("Generative AI and LLM Apps-> Chatbo")  # 80
    st.success("Generative AI and LLM Apps-> malicious email detection with ChatGPT and GPT")  # 92
    st.warning("Generative AI and LLM Apps-> Connecting my GPTs to the real world: intelligent URL analysis")  # 95
    st.info("Generative AI and LLM Apps-> Implementing a Chatbot with ChatGPT and GPT")  # 97
    st.error("Generative AI and LLM Apps-> Integration of Chatbot with external services")  # 99
    st.success("Generative AI, LLM and Recurrent Neural Networks. Apps-> Integration of Chatbot with external service")  # 99
    st.warning("Generative AI, LLM and Recurrent Neural Networks. Apps-> Text generation with LSTM")  #106
    st.info("Generative AI, LLM and Transformers. Apps-> Text Generation with Transformers")  # 121
    st.error("Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuned LLM (FLAN-T5 by Google)")  # 129
    st.success("Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuned of an LLM")  # 134
    st.warning("Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Fine-tuning of GPT and ChatGPT to call external functions")  # 141
    st.info("Generative AI, LLM, Pre-training and Fine-Tunning, PEFT (Parameter Efficient Fine-tunning), LoRA (Low-Rank Adaption. Apps-> PEFT of LLAMA2 (Meta) with QLoRA")  # 146,147
    st.error("Generative AI, Fine-Tunning, RLHF (Reinforcement Learning from Human Feeback) and PPO (Proximal Policy Optimization). Apps-> RLHF of LLAMA2 (TinyLlama)")  # 155-157
    st.success("Generative AI, LLM, GANs (Generative Adversarial Networks). Apps-> To change a person's face in a video")  # 163
    st.warning("Generative AI, LLM, Diffusion Models, CLIP (Contrastive Language-Image Pre-training). Apps-> Automatic Generations fo Images. DALL-E 3")  # 169

    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/1_Introducci%C3%B3n%20a%20NumPy.ipynb
    st.error("Practice of Machine Learning -> Introduction to NumPy")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/2_Introducci%C3%B3n%20a%20Pandas.ipynb
    st.success("Practice of Machine Learning -> Introduction to Panda")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/3_Introducci%C3%B3n%20a%20Matplotlib.ipynb
    st.warning("Practice of Machine Learning -> Introduction to Matplotlib")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/4_Regresi%C3%B3n%20Lineal%20-%20Predicci%C3%B3n%20del%20coste%20de%20un%20incidente%20de%20seguridad.ipynb
    st.info("Practice of Machine Learning -> Linear Regression: Cost of a security incident")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/5_Regresi%C3%B3n%20Log%C3%ADstica%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb
    st.error("Practice of Machine Learning App -> Logistic Regression: SPAM Detection")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/6_Visualizaci%C3%B3n%20del%20conjunto%20de%20datos.ipynb
    st.success("Practice of Machine Learning -> Visualization of the dataset")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/7_Divisi%C3%B3n%20del%20conjunto%20de%20datos.ipynb
    st.warning("Practice of Machine Learning -> Division of the data set")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/8_Preparaci%C3%B3n%20del%20conjunto%20de%20datos.ipynb
    st.error("Practice of Machine Learning  -> Data set preparation.")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/9_Creaci%C3%B3n%20de%20Transformadores%20y%20Pipelines%20personalizados.ipynb
    st.error("Machine Learning  -> Creation of custom transformers and pipelines" )
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/10_Evaluaci%C3%B3n%20de%20resultados.ipynb
    st.error("Practice of Machine Learning  -> Evaluation of results")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/11_Support%20Vector%20Machine%20-%20Detecci%C3%B3n%20de%20URLs%20maliciosas.ipynb
    st.error("Practice of Machine Learning  -> Support Vector Machine (SVM)")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/12_%C3%81rboles%20de%20decisi%C3%B3n%20-%20Detecci%C3%B3n%20de%20malware%20en%20Android.ipynb
    st.error("Practice of Machine Learning  -> Decision trees")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/13_Random%20Forests%20-%20Detecci%C3%B3n%20de%20Malware%20en%20Android.ipynb
    st.error("Practice of Machine Learning  -> Random Forest")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/14_T%C3%A9cnicas%20de%20selecci%C3%B3n%20de%20caracter%C3%ADsticas.ipynb
    st.error("Practice of Machine Learning  -> Features selection")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/15_PCA%20-%20Extracci%C3%B3n%20de%20caracter%C3%ADsticas.ipynb
    st.error("Practice of Machine Learning  -> Principal Component Analysis (PCA)")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/16_T%C3%A9cnicas%20de%20selecci%C3%B3n%20del%20modelo.ipynb
    st.error("Practice of Machine Learning  -> Model selection")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/17_KMEANS%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb
    st.error("Machine Learning  -> KMEANS - Detection of fraudulent bank transactions")
    #https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/18_DBSCAN%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb
    st.error("Machine Learning  -> DBSCAN - Detection of fraudulent bank transactions")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/19_Naive%20Bayes%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb
    st.error("Practice of Machine Learning  -> Naive Bayes - SPAM Detection")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/19_Naive%20Bayes%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb
    st.error("Practice of Machine Learning  -> Gaussian Distribution - Detection of fraudulent bank transactions")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/21_Isolation%20Forest%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb
    st.error("Practice of Machine Learning  -> Isolation Forest")
    # https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/22_Redes%20Neuronales%20Artificiales%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb
    st.error("Practice of Machine Learning  -> Artificial Neural Networks - Detection of fraudulent bank transactions")
    
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

    st.error("N8N: -> ")  
    st.error("N8N: -> ")  
    st.error("N8N: -> ")  
    st.error("N8N: -> ")  
    
    st.error("MCP: -> ")  
    st.error("MCP: -> ")  
    st.error("MCP: -> ")  
    st.error("MCP: -> ")  

with col2:
    st.link_button("Code", "")
    st.write('')
    st.link_button("Codeb", "https://github.com/web-roberto/langchain/tree/main/seccion4")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion4")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion5")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion6")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion6")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion7")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion7")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion8")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion8")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion8")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/langchain/tree/main/seccion8")
    st.write('')
    
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/1_Introducci%C3%B3n%20a%20NumPy.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/2_Introducci%C3%B3n%20a%20Pandas.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/3_Introducci%C3%B3n%20a%20Matplotlib.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/4_Regresi%C3%B3n%20Lineal%20-%20Predicci%C3%B3n%20del%20coste%20de%20un%20incidente%20de%20seguridad.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/5_Regresi%C3%B3n%20Log%C3%ADstica%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/6_Visualizaci%C3%B3n%20del%20conjunto%20de%20datos.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/7_Divisi%C3%B3n%20del%20conjunto%20de%20datos.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/8_Preparaci%C3%B3n%20del%20conjunto%20de%20datos.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/9_Creaci%C3%B3n%20de%20Transformadores%20y%20Pipelines%20personalizados.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/10_Evaluaci%C3%B3n%20de%20resultados.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/11_Support%20Vector%20Machine%20-%20Detecci%C3%B3n%20de%20URLs%20maliciosas.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/12_%C3%81rboles%20de%20decisi%C3%B3n%20-%20Detecci%C3%B3n%20de%20malware%20en%20Android.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/13_Random%20Forests%20-%20Detecci%C3%B3n%20de%20Malware%20en%20Android.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/14_T%C3%A9cnicas%20de%20selecci%C3%B3n%20de%20caracter%C3%ADsticas.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/15_PCA%20-%20Extracci%C3%B3n%20de%20caracter%C3%ADsticas.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/16_T%C3%A9cnicas%20de%20selecci%C3%B3n%20del%20modelo.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/17_KMEANS%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/18_DBSCAN%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb")
    st.write('https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/19_Naive%20Bayes%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/19_Naive%20Bayes%20-%20Detecci%C3%B3n%20de%20SPAM.ipynb")
    st.write('')
    st.link_button("Code", "https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/21_Isolation%20Forest%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb")
    st.write('') 
    st.link_button("Code", "    https://github.com/web-roberto/MachineLearning_python/blob/main/casos_practicos_machine_learning/22_Redes%20Neuronales%20Artificiales%20-%20Detecci%C3%B3n%20de%20transacciones%20bancarias%20fraudulentas.ipynb")
    st.write('') 

with col3:
    st.link_button("Run App", "")
    st.write('')
    st.link_button("Run App", "")
    st.write('')
    st.link_button("Run App", "")
    st.write('')
    st.link_button("Run App", "")
    st.write('')
    st.link_button("Run App", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
    st.link_button("Code Github", "")
    st.write('')
