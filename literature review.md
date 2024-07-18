## Related Work

Xu et al. (2015) in "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" introduce a model that incorporates visual attention mechanisms—both soft and hard attention—to dynamically focus on different parts of images, significantly enhancing caption quality and contextual adaptability, validated on standard datasets like Flickr8k and Microsoft COCO.

Rennie et al. (2017) in "Self-critical Sequence Training for Image Captioning" employ the REINFORCE algorithm with a unique self-assessment strategy, using the model's test-time output as a baseline to improve REINFORCE's application in RL and enhance performance in non-differentiable metric optimization on the MSCOCO dataset.

## Content (completed part)

1502.03044v3: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

1612.00563v2: Self-critical Sequence Training for Image Captioning

### Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

In the fields of computer vision and natural language processing, automatic image caption generation has always been a challenging research topic. Xu et al. (2015), in their paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention," introduced a neural network model incorporating a visual attention mechanism that significantly enhances the quality and relevance of image descriptions.

Early methods of image caption generation relied on simple visual features and predefined templates. With the advancement of deep learning technologies, researchers began to use a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs) in an encoder-decoder architecture to generate descriptions, as seen in the studies by Vinyals et al. (2015) and Karpathy and Fei-Fei (2015). However, these methods typically produce static descriptions that lack contextual adaptability.

The introduction of attention mechanisms allows the model to dynamically focus on different parts of the image, thereby generating more precise and enriched descriptions. Xu et al. (2015)'s study is pioneering in this field, proposing two types of attention mechanisms: soft attention, which is differentiable and allows for training using standard backpropagation algorithms; and hard attention, which uses a stochastic sampling method optimized through reinforcement learning.

In the experimental section, Xu and colleagues utilized standard image description datasets, such as Flickr8k, Flickr30k, and Microsoft COCO, to validate the effectiveness of the model. The results indicate that the model with the visual attention mechanism surpasses previous methods on multiple evaluation metrics, particularly in handling complex scenes, where it can generate more detailed and accurate descriptions. Moreover, Xu et al. visualized the attention weights and found that the model's learned alignment highly corresponds with human intuition, enhancing the interpretability of this mechanism.

The work of Xu et al. (2015) not only advances the technology of image caption generation but also provides a powerful tool for deepening the understanding of the interaction between vision and language. Their research outcomes demonstrate the importance of attention mechanisms in enhancing machines' capabilities in understanding visual content and generating language, laying a solid foundation for future research.

### Self-critical Sequence Training for Image Captioning

In the task of image caption generation, traditional encoder-decoder architectures often face challenges due to exposure bias during training and the inability to directly optimize non-differentiable sequence-level evaluation metrics. To overcome these limitations, recent research has begun to explore the application of Reinforcement Learning (RL) techniques to image caption generation, enabling direct optimization of evaluation metrics and thus improving model performance.

The work of Rennie et al. (2017) in "Self-critical Sequence Training for Image Captioning" introduces a method known as Self-critical Sequence Training (SCST), which utilizes the REINFORCE algorithm from RL. Unlike traditional applications, SCST employs a unique self-assessment strategy by using the model's output at test time as the reward baseline to normalize rewards during the training process. It dynamically adjusts rewards so that only those samples that perform better than the current test system receive positive weight, while underperforming samples are suppressed.

This self-assessment algorithm allows for more effective baselining of the REINFORCE algorithm for policy-gradient based RL and more effective training on non-differentiable metrics, leading to significant improvements in captioning performance on MSCOCO—the results on the MSCOCO evaluation server establish a new state-of-the-art for the task. The self-critical approach, which normalizes the reward obtained by sampled sentences with the reward obtained by the model under the test-time inference algorithm, is intuitive and avoids the need to estimate both action-dependent and action-independent reward functions.

The research by Rennie et al. not only enhances the quality of image caption generation but also provides an effective strategy for using reinforcement learning to optimize non-differentiable metrics. Overall, the SCST method demonstrates its strong potential and wide applicability both theoretically and in practical applications, providing new directions for future research.