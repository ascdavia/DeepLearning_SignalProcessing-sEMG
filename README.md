<h1 align="center"> Aplicação de Diferentes Algoritmos de Aprendizado Profundo para Classificação de Sinais Biológico </h1>

Esse trabalho visa realizar a classificação de Biosinais (sinais elétricos gerados por seres vivos) utilizando algoritimos de Deep Learning. 

Foram utilizados CNN's (Convolutional Neural Networks), LSTM's (Long Short Term Memory) e o ensemble entre as duas.

Os sinais utilizados são gerados apartir de seis gestos de mão que são utilizados em processos de reabilitação motora. A seguir uma imagem que represta os gestos.

![grasps_en](https://user-images.githubusercontent.com/76635621/182120286-b042691d-41a7-46e7-a4e4-ab10f56e0023.PNG)




## :hammer: Códigos

- `2_Layers_Conv1D`: Diferentes pré-pocessamento de dados aplicados em uma rede com duas camadas convolucionais. 

- `4_Layers_Conv1D`: Diferentes pré-pocessamento de dados aplicados em uma rede com quatro camadas convolucionais. 

- `Wavelet_Transform`: Utilização da Transformada de Wavelet como pré-pocessamento de dados aplicados em quatro redes diferentes, os quais são: CNN, LSTM, CNN+LSTM e CNN+LSTM Multscale. 

- `Criação_DF_Movimentos_e_MatrizX.ipynb`: Como os arquivos da base de dados foram separados por movimento e por pessoa, foi necessário juntar eles formando arquivos .csv para cada movimento e um arquivo para juntar todos os movimentos.

- `Plot_dos_Sinais.ipynb `: Plotagem dos sinais para ter uma visualização de como eles são além de comparar entre si os sinais gerados pelo homem e pela mulher.

