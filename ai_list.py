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
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/seccion5-Deepseek-Claude-Gemini/Prompts_tarea_DeepSeek.txt",
    "text": "Generative AI and Large Language Model (LLM) Apps-> Practice with DeepSeek",
    "fondo": "darkmagenta" }, # seccion5- 38
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/seccion5-Deepseek-Claude-Gemini/Prompts_tarea_Claude.txt",
    "text": "Generative AI and LLM) Apps-> Practice with Claude",
    "fondo": "blue" }, #  seccion5-  43
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/seccion5-Deepseek-Claude-Gemini/Prompts_tarea_Gemini.txt",
    "text": "Generative AI and LLM Apps-> Practice with Gemini",
    "fondo": "red" }, # seccion5- 51
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/tree/main/seccion6-prompts-avanzados/Advanced-Data-Analisys",
    "text": "Prompts -> Advanced Data Analisys",
    "fondo": "yellowgreen" }, # seccion6- 57
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/seccion6-prompts-avanzados/Prompts_busqueda_Internet_chatGPT4.txt",
    "text": "Prompts search in intermet",
    "fondo": "red" }, # seccion6- 62
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/seccion6-prompts-avanzados/Prompts_gpts_external_services.txt",
    "text": "Prompts external services",
    "fondo": "red" }, # seccion6- 63
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion8/94.Detecci%C3%B3n_correos_maliciosos_GPT_Parte2.ipynb",
    "text": "Generative AI and LLM Apps-> malicious email detection with ChatGPT and GPT",
    "fondo": "yellowgreen" }, # seccion8- 93
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion8/102.Hiperparametros_LLMs_GPT.ipynb",
    "text": "Generative AI and LLM Apps-> Hyperparameters LLM",
    "fondo": "blue" }, # seccion8- 94
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/tree/main/generative_ai_seccion8/96.gpt_con_mundo_exterior_analisis_url",
    "text": "Generative AI and LLM Apps-> Connecting my GPTs to the real world: intelligent URL analysis",
    "fondo": "red" }, # seccion8-96
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion8/97.Implementacion_ChatBot_GPT.ipynb",
    "text": "Generative AI and LLM Apps-> Implementing a Chatbot with ChatGPT and GPT",
    "fondo": "yellowgreen" }, # seccion8-97
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion8/101.Funciones_GPT_ChatGPT_Integraci%C3%B3n_servicios_externos.ipynb",
    "text": "Generative AI and LLM Apps-> Integration of Chatbot with external services",
    "fondo": "darkmagenta" }, # seccion8-99
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/tree/main/generative_ai_seccion9/106.GeneracionTextoLSTM",
    "text": "Generative AI, LLM and Recurrent Neural Networks. Apps-> Text generation with LSTM",
    "fondo": "red" }, # seccion9-106
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/tree/main/generative_ai_seccion9/121.GeneracionTextoConTransformers",
    "text": "Generative AI, LLM and Transformers. Apps-> Text Generation with Transformers",
    "fondo": "yellowgreen" }, # seccion9-121
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion10/instruction_fine_tuning_llm_base.py",
    "text": "Generative AI, LLM,  Pre-training and Fine-Tunning. Apps-> ->  Instruction Fine-tuning sobre un LLM Base",
    "fondo": "darkmagenta" }, # seccion10-129,130
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion10/130.Base_LLM_vs_Fine_Tuned_LLM.ipynb",
    "text": "Generative AI, LLM, Pre-training and Fine-Tunning. Apps-> Base LLM vs Fine-tuned LLM",
    "fondo": "blue" }, # seccion10-134
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion10/139.Fine_Tuning_GPT_4_ChatGPT.ipynb",
    "text": "Fine-tuning de GPT 4 y ChatGPT",
    "fondo": "yellowgreen" }, # seccion10-137
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion10/141.Funciones_y_Fine_Tuning_GPT_4_ChatGPT.ipynb",
    "text": "Fine-tuning for GPT 4 and ChatGP to call external functions",
    "fondo": "blue" }, # seccion10-139
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion10/148.convert_llama_weights_to_hf.py",
    "text": "Convert Llama weights to hf",
    "fondo": "red" }, # seccion10-141
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion10/146.PEFT_QLoRA_LLAMA_2.ipynb",
    "text": "Generative AI, LLM, Pre-training and Fine-Tunning, PEFT (Parameter Efficient Fine-tunning), LoRA (Low-Rank Adaption. Apps-> PEFT of LLAMA2 (Meta) with QLoRA",
    "fondo": "yellowgreen" }, # seccion10-146, 147
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion11/156.Reinforcement_Learning_Human_Feedback_PPO_LLAMA.ipynb",
    "text": "Generative AI, Fine-Tunning, RLHF (Reinforcement Learning from Human Feeback) and PPO (Proximal Policy Optimization). Apps-> RLHF of LLAMA2 (TinyLlama)",
    "fondo": "darkmagenta" }, # seccion11-155-157
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion12/163.Generative_adversarial_Networks_modificacion_videos.ipynb",
    "text": "Generative AI, LLM, GANs (Generative Adversarial Networks). Apps-> To change a person's face in a video",
    "fondo": "blue" }, # seccion12-163
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion12/171.introduccion_stable_diffusion.py",
    "text": " Introduction to Stable Diffusion",
    "fondo": "red" }, # seccion12-169
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/tree/main/generative_ai_seccion12/169",
    "text": "Generative AI, LLM, Diffusion Models, CLIP (Contrastive Language-Image Pre-training). Apps-> Automatic Generations fo Images. DALL-E 3",
    "fondo": "darkmagenta" }, # seccion12-170
    {"href": "https://github.com/web-roberto/generative_artificial_intelligence/blob/main/generative_ai_seccion13/179.fast_stable_diffusion_AUTOMATIC1111.ipynb",
    "text": "Fast Stable Diffusion AUTOMATIC1111",
    "fondo": "yellowgreen" }, # seccion13-179
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 

st.subheader("Deep Learning")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### DEEP LEARNING
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/18.Intuiciones%20sobre%20End-to-End%20Learning.ipynb",
    "text": "Deep Learning: End-to-End intuitions",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/28.Implementando%20la%20Neurona%20M-P.ipynb ",
    "text": "Deep Learning: M-P NEURON App -> Implementation",
    "fondo": "yellowgreen" }, # 28
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/29.Diagn%C3%B3stico%20de%20c%C3%A1ncer%20de%20mama%20con%20el%20Perceptr%C3%B3n.ipynb",
    "text": "Deep Learning: M-P NEURON App -> Diagnosis of breast cancer with one M-P neuron",
    "fondo": "darkmagenta" }, # 29
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/36.Visualizaci%C3%B3n%20del%20l%C3%ADmite%20de%20decisi%C3%B3n%20del%20Perceptr%C3%B3n.ipynb",
    "text": "Deep Learning: PRECEPTRON App -> Visualization of the perceptron decision boundary",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/39.Clasificaci%C3%B3n%2Bde%2Bim%C3%A1genes%2Bcon%2Bel%2BPerceptr%C3%B3n.ipynb",
    "text": "Deep Learning: PRECEPTRON App -> Classification of Images",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/46.Visualizando%20el%20l%C3%ADmite%20de%20decisi%C3%B3n%20del%20Perceptr%C3%B3n%20Multicapa.ipynb",
    "text": "Deep Learning: PRECEPTRON MULTILAYER App -> Decision boundary of the multilayer perceptron",
    "fondo": "yellowgreen" }, # 46
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/57.Clasificaci%C3%B3n%20de%20im%C3%A1genes%20con%20el%20Perceptr%C3%B3n%20Multicapa.ipynb",
    "text": "Deep Learning: loss and optimization functions. App -> Classification of images of handwritten digits",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/tree/main/DeepLearning/66.Clasificaci%C3%B3n%2Bde%2Baudio%2Bcon%2Bel%2BPerceptr%C3%B3n%2BMulticapa%2BI.ipynb",
    "text": "Deep Learning: Audio Classification",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/tree/main/DeepLearning/67.Clasificaci%C3%B3n%2Bde%2Baudio%2Bcon%2Bel%2BPerceptr%C3%B3n%2BMulticapa%2BII.ipynb",
    "text": "Deep Learning: Audio Classification with the Multilayer Perceptron",
    "fondo": "red" }, # 67
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/81.Implementando%20una%20RNA%20con%20Keras.ipynb",
    "text": "Deep Learning: KERAS App -> Implementation of Artificial Neuronal Network",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/83.Limite%20de%20decisi%C3%B3n%20de%20una%20RNA%20profunda.ipynb",
    "text": "Deep Learning: KERAS App -> Decision limit of a deep artificial neural network",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/84.Clasificaci%C3%B3n%20de%20audio%20con%20Keras.ipynb",
    "text": "Deep Learning: KERAS App -> Audio Classification with Keras",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/85.Regresi%C3%B3n%20con%20Keras.ipynb",
    "text": "Deep Learning: KERAS App -> Regression with Keras",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/tree/main/DeepLearning/91.Visualizaci%C3%B3n%2Bde%2Blas%2Bfunciones%2Bde%2Bactivaci%C3%B3n.ipynb",
    "text": "Deep Learning: Activation Functions App -> Show activation functions",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/tree/main/DeepLearning/93.%C2%BFPor%2Bqu%C3%A9%2Butilizar%2Buna%2Bfunci%C3%B3n%2Bde%2Bactivaci%C3%B3n_.ipynb",
    "text": "Deep Learning: Activation Functions App -> Why to use of activation functions",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/95.Clasificaci%C3%B3n%20de%20sentimientos.ipynb",
    "text": "Deep Learning: Activation Functions App -> Feelings classification",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/95.Clasificaci%C3%B3n%20de%20sentimientos.ipynb",
    "text": "Deep Learning -> Feelings Classification ",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/99.Clasificaci%C3%B3n%20de%20texto%20con%20Mini-Batch%20Gradient%20Descent.ipynb",
    "text": "Deep Learning: Optimization Functions App -> Mini-Batch Gradient Descent",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/101.Clasificaci%C3%B3n%20de%20texto%20con%20Stochastic%20Gradient%20Descent.ipynb",
    "text": "Deep Learning -> Stochastic Gradient Descent",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/tree/main/DeepLearning/104.Exponentially%2Bweighted%2Bmoving%2Baverage.ipynb",
    "text": "Deep Learning: Optimization Functions App -> Exponentially Weighted Moving Average",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/106.Clasificaci%C3%B3n%20de%20texto%20con%20Momentum%20Gradient%20Descent.ipynb",
    "text": "Deep Learning: Optimization Functions App -> Momentum Gradient Descent",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/108.Clasificaci%C3%B3n%20de%20texto%20con%20RMSprop.ipynb",
    "text": "Deep Learning: Optimization Functions App -> RMSprop",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/110.Clasificaci%C3%B3n%20de%20texto%20con%20Adam.ipynb",
    "text": "Deep Learning: Optimization Functions App -> Adam Optimization",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/113.Selecci%C3%B3n%20de%20hiperpar%C3%A1metros%20con%20Keras%20Tuner.ipynb",
    "text": "Deep Learning: Hyperparams App -> Selection of Hyperparams with Teras Turner",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/117.Selecci%C3%B3n%20de%20hiperpar%C3%A1metros%20y%20clasificaci%C3%B3n%20de%20texto.ipynb",
    "text": "Deep Learning: Hyperparams App -> Selection of Hyperparams and text classification",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/120.Tensores%20y%20operaciones%20con%20tensores.ipynb",
    "text": "Deep Learning: TENSORFLOW App -> Tensor and operations",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/121.Creaci%C3%B3n%20de%20una%20funci%C3%B3n%20de%20error%20personalizada.ipynb",
    "text": "Deep Learning: TENSORFLOW App -> Create a custom loss function",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/122.Creaci%C3%B3n%20de%20otros%20componentes%20personalizados%20con%20Tensorflow.ipynb",
    "text": "Deep Learning: TENSORFLOW App -> Create a custom component",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/deep_learning_roberto/blob/main/DeepLearning/123.Creaci%C3%B3n%20de%20capas%20personalizadas%20con%20Tensorflow.ipynb",
    "text": "Deep Learning: TENSORFLOW App -> Create a custom layer",
    "fondo": "blue" }, # 
     {"href": "https://github.com/web-roberto/deep_learning_roberto/tree/main/DeepLearning/132.Clasificaci%C3%B3n%2Bde%2BTweets%2Bde%2Bdesastres%2Bnaturales.ipynb",
    "text": "Deep Learning: TENSORFLOW App -> Classification of Tweets about natural disasters",
    "fondo": "blue" }, # 
]
for link in links:
    st.markdown(f"""<a href="{link['href']}" target="_blank">
        <button style="background-color:{link['fondo']};color:white;padding:0.5em 1em;margin:0.5em;width:100%;border:none;text-align:start;border-radius:8px;cursor:pointer;">
            {link['text']}
        </button></a>""",unsafe_allow_html=True) 
    
st.subheader("AI with LangChain")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### LANGCHAIN
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

st.subheader("MCP")
#greenyellow,lime,orangered,yellowgreen,deeppink,    darkmagenta,blueviolet,red, blue
links = [
    ### MCP
   {"href": "https://github.com/web-roberto/MCP_Model_Control_Protocol/blob/main/3.introduction_claude_desktop_config.json",
    "text": "Introduction to Claude desktop",
    "fondo": "yellowgreen" }, # 
    {"href": "https://github.com/web-roberto/MCP_Model_Control_Protocol/tree/main/seccion4_promps_en_MCP",
    "text": "Promps in MCP",
    "fondo": "darkmagenta" }, # 
    {"href": "https://github.com/web-roberto/MCP_Model_Control_Protocol/tree/main/13.mpc_server",
    "text": "MCP Server",
    "fondo": "blue" }, # 
    {"href": "https://github.com/web-roberto/MCP_Model_Control_Protocol/tree/main/24.gmail_mcp_server_resource_template",
    "text": "gmail MCP Server Resource Template",
    "fondo": "red" }, # 
    {"href": "https://github.com/web-roberto/MCP_Model_Control_Protocol/tree/main/42.using%20local%20LLMs%20with%20an%20MCP%20client",
    "text": "Using local LLMs with an MCP client",
    "fondo": "yellowgreen" }, # 
    {"href": "MChttps://github.com/web-roberto/MCP_Model_Control_Protocol/tree/main/seccion5_Clientes_personalizadosP",
    "text": "Customized Clients",
    "fondo": "darkmagenta" }, # 
    {"href": "Mhttps://github.com/web-roberto/MCP_Model_Control_Protocol/tree/main/seccion7_Seguridad_MCPCP",
    "text": "Security in MCP",
    "fondo": "blue" }, # 
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




