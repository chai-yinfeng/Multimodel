# Overall

For the Visual Question Answering (VQA) task, the model needs to generate a correct answer by fusing features from both the given image and the natural language question. Based on the task requirements, the model is designed to take an image and a natural language text as input, and output the answer in natural language. The key challenge lies in understanding both the image content and the text, and providing an appropriate response. A typical model framework includes the following components: an image encoder, a text encoder, a feature fusion module, and an answer generation module.

The image encoder is responsible for extracting high-level visual features from the image. In early frameworks, deep convolutional neural networks (CNNs) such as ResNet are commonly used to obtain these image features. The text encoder transforms the natural language question into an embedded representation, which can be encoded using recurrent neural networks (RNNs) or long short-term memory networks (LSTMs). The feature fusion module, which is the core of multimodal tasks, combines the image and text features. In this framework, stacked attention mechanisms are employed to generate attention weights and fuse the visual and linguistic information. For the answer generation module, we use a simple label classifier instead of a language model, aiming to simplify the output module and allow the model to focus on a smaller output space. Specifically, this module passes the fused features through fully connected layers to generate a probability distribution over the answer classes, and the final answer is produced based on the highest probability.

The overall framework can be expressed as follows: for a given image input $I$ and a natural language question $q$, the model selects the answer $\hat{a}$ with the highest probability from a set of candidate answers extracted from the image content, where $a \in \{a_1, a_2, ..., a_M \}$
$$\hat{a}={\arg\max}_a P(a|I,q)$$
![[VQA pipeline.png]]
Figure. An overview of our model. We use a convolutional neural network based on ResNet to embed the image, and a multi-layer LSTM network to tokenize and embed input text. The concat module concatenate image features and the final state of LSTMs, which used to compute multiple attention distributions over the image features. Then we use image features and attention distribution to fuse a multi-veiw image. This new fusion image will be used to concatenate with the state of the LSTM, and finally be fed to fully connected layers to produce probabilities over answer classes.
# Image Encoder

The image encoder is responsible for extracting high-dimensional features from the image, and this part determines how the image input will be utilized by the model, especially by the feature fusion module. In the base framework, we use ResNet (Residual Network) as the image feature extractor. The model extracts the final layer's convolutional features through a pre-trained ResNet, and these features are then used as the input to the model.

For a given input image III, this module outputs a 3D tensor $\phi$ with dimensions $14 \times 14 \times 2048$ , where the $2048$ dimension represents the visual information for that specific region of the image.
$$\phi=\text{CNN}(I)$$
# Text Encoder

The text encoder transforms the natural language question into a feature representation. In this case, we use an LSTM (Long Short-Term Memory) network to handle the sequential information of the question. While traditional LSTM networks are effective at capturing positional features in sequences, their efficiency is significantly impacted due to the need for sequential processing.

For a given natural language question $q$, this module outputs an embedded representation of the question, denoted as $s$, which is the final hidden state of the LSTM network. The process involves two sub-steps: word embedding and LSTM encoding. First, the text encoder converts each word in the question into a distributed representation through an embedding layer, generating a sequence of word embeddings $E_q = \{e_1, e_2, ..., e_P\}$, where $e_i \in \mathbb{R}^D$, and each word vector has a length of $D$ (with a default value of $300$ dimensions). The word embedding sequence is then passed through the LSTM, and the final hidden state $s$ is produced as the encoded representation of the question.
$$s = \text{LSTM}(E_q)$$
# Feature Fusion

The core of the feature fusion module is the **stacked attention mechanism**, which assigns weights to the image features based on the text features. This mechanism captures important regions in the image that are related to the question, resulting in a fused feature view that is used for the final answer generation.

First, this module calculates attention weights $\alpha_{c,l}$​ across the spatial dimensions of the image features. The attention weights are computed using a convolution operation based on the text features $s$ and the image features $\phi$. Next, the attention weights are used to perform a weighted average over the image features, achieving multi-view fusion and generating several image feature views $x_1, x_2, ..., x_C$​. Finally, the generated image feature views are concatenated with the text feature representation sss, which will be used for the subsequent classification.

The attention weights $\alpha_{c,l}$​ are derived from the convolution of the text and image features and are normalized for each view $c = 1, 2, ..., C$. In the formula, $⁡\exp$ represents the softmax function applied to the vectors to obtain the normalized output.
$$\alpha_{c,l}\propto\exp F_c(s,\phi_l)$$
 $$\sum_{l=1}^L \alpha_{c,l}=1$$
 Each image feature $\text{x}_c$​ is the weighted average of the image features $\phi$ across all spatial locations $l = \{1, 2, ..., L\}$.
 $$\text{x}_c=\sum_l \alpha_{c,l}\cdot \phi_l$$
 In practice, $F = [F_1, F_2, ..., F_C]$ is modeled using two layers of convolution, where $F_i$​ shares parameters in the first layer. Different attention distributions are generated solely by relying on different initializations.
# Label CLassifier

The model's output module in this framework adopts a simple label classification approach. The classifier component takes the fused feature vector and passes it through fully connected layers to output the probability distribution over the possible answers. The concatenated image and text features from the feature fusion module are processed by two fully connected layers, generating a probability for each answer class. Then, the cross-entropy loss function is used to calculate the difference between the model’s predictions and the ground truth answers, guiding the training process through supervision.

We concatenate the image features from the feature fusion step with the text embeddings obtained from the LSTM, aiming for the classifier to predict the answer with the highest probability.
$$P(a_i|I,q)\propto \exp G_i(\text{x},s)$$
Here, $\text{x} = [\text{x}_1, \text{x}_2, ..., \text{x}_C]$ represents the attention-weighted feature views, and $G = [G_1, G_2, ..., G_M]$ denotes the two-layer fully connected network mentioned earlier.

The cross-entropy loss function is expressed as follows:
$$L=\frac{1}{K}\sum_{k=1}^K -\log P(a_k|I,q)$$