# Deep Learning Train Solution

View the git code with HackMD.

Gradient Problem
---

- Vanishing Gradient Problem 梯度消失 隨著隱藏層數目增加，分類準確度反而下降
- Exploding Gradient Problem 梯度爆炸 初始化權值過大，activation function導數都大於1

Gradient Problem Solution
---

- ReLU、Leaky-ReLU、P-ReLU、R-ReLU、Maxout exchange Sigmoid
- Batch Normalization
- LSTM structure may improve the Gradient Problem in RNN
- Dynamic Change Learning Rate, 當梯度過小->增加學習率 梯度過大->減少學習率
- 神經網路權重標準初始化

Rex Tsou 2021/06/06

###### tags: `DeepLearning` `Python` `Gradient`