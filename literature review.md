# Related work

## **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**
Xu et al. (2015) in "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" introduce a model that incorporates visual attention mechanisms—both soft and hard attention—to dynamically focus on different parts of images, significantly enhancing caption quality and contextual adaptability, validated on standard datasets like Flickr8k and Microsoft COCO.

**refined**: Xu and colleagues address the challenge of generating contextually relevant and accurate image captions by introducing a visual attention mechanism. Their model utilizes both soft and hard attention to dynamically focus on pertinent parts of images, enhancing the relevance and detail of the generated captions. The model demonstrated superior performance on standard image captioning datasets such as Flickr8k and Microsoft COCO, setting new benchmarks for caption quality and adaptability.

---
##  **Self-critical Sequence Training for Image Captioning**
Rennie et al. (2017) in "Self-critical Sequence Training for Image Captioning" employ the REINFORCE algorithm with a unique self-assessment strategy, using the model's test-time output as a baseline to improve REINFORCE's application in RL and enhance performance in non-differentiable metric optimization on the MSCOCO dataset.

**refined**: Rennie et al. tackle the optimization of non-differentiable metrics in image caption generation, which traditional methods struggle with due to exposure bias and reward sparsity. They employ the REINFORCE algorithm enhanced with a self-critical sequence training strategy, using the model's own output at test time as a baseline for reward normalization. This approach led to significant improvements in captioning performance on the MSCOCO dataset, establishing a new state-of-the-art by effectively training on direct evaluation metrics.

---
## **Ask, Attend and Answer: Exploring Question-Guided Spatial Attention for Visual Question Answering**
Huijuan Xu and Kate Saenko addressed the challenge of spatial reasoning in Visual Question Answering (VQA) tasks by proposing a new model that integrates deep learning and attention mechanisms, named the Spatial Memory Network (SMem-VQA). This model utilizes a multi-hop attention mechanism, iteratively focusing on key areas of the image through a memory network to address specific query requirements. Experiments on standard VQA datasets such as DAQUAR and VQA demonstrate that the SMem-VQA model outperforms existing methods in complex spatial reasoning tasks, particularly in terms of accuracy and relevance to the questions posed.

---
## **Bi-Directional Attention Flow for Machine Comprehension**
Minjoon Seo et al. (2017) in aimed to enhance the comprehension capabilities of machine learning models in question-answering systems by addressing limitations in understanding the context of questions and answers. They introduced the Bi-Directional Attention Flow (BiDAF) model, which employs a novel bi-directional attention mechanism that enriches the mutual interaction between the context and the query for more accurate comprehension. BiDAF demonstrated superior performance, setting new state-of-the-art benchmarks on the SQuAD dataset by significantly improving accuracy in machine comprehension tasks.

---
## **Learning Convolutional Text Representations for Visual Question Answering**
Zhengyang Wang and Shuiwang Ji address the challenge of learning text representations for Visual Question Answering (VQA) tasks, critiquing traditional reliance on Recurrent Neural Networks (RNNs) and proposing the use of Convolutional Neural Networks (CNNs) for more effective text processing. They introduce a novel model named “CNN Inception + Gate,” which incorporates Inception modules and gating mechanisms to extract text features. This model utilizes convolutional kernels of varying sizes to capture essential textual information and employs gates to optimize the flow of information. Experimental results demonstrate that this CNN-based model not only improves the quality of question representations but also significantly enhances overall VQA accuracy, with fewer parameters and faster computation compared to traditional RNN-based models.

---
## **Structured Attentions for Visual Question Answering**
The authors tackle the limited effective receptive field in CNNs for Visual Question Answering (VQA), which struggles with complex spatial relationships between image regions. They introduce a novel visual attention model using a grid-structured Conditional Random Field (CRF) to capture inter-region dependencies, implemented through Mean Field and Loopy Belief Propagation algorithms as recurrent neural network layers. The model significantly improves upon baseline models, enhancing accuracy on the CLEVR dataset by 9.5% and on the VQA dataset by 1.25%.

---
## **iVQA: Inverse Visual Question Answering**
The authors introduced the task of Inverse Visual Question Answering (iVQA), aimed at generating questions corresponding to given image and answer pairs to enhance the model's understanding of image content and language generation capabilities. They developed a model based on a dynamic multimodal attention mechanism, which adjusts focus on image regions during question generation to better integrate image content and provided answers. Experimental results demonstrated that the model can generate diverse and highly relevant questions, validated by a newly proposed ranking-based evaluation metric, confirming its effectiveness and superiority in the iVQA task.

---
## **Meshed-Memory Transformer for Image Captioning**
The authors addressed the challenge of enhancing image captioning by improving the integration of visual and textual data, as existing models were insufficient in capturing complex inter-modal relationships. They proposed the Meshed-Memory Transformer (M2 Transformer), a novel architecture that utilizes a memory-augmented encoder and a meshed decoder to exploit both low- and high-level visual features while incorporating learned a priori knowledge. The M2 Transformer achieved state-of-the-art performance on the COCO dataset, demonstrating its effectiveness in handling novel objects and setting new benchmarks for image captioning tasks.

---
## **X-Linear Attention Networks for Image Captioning**
The authors aim to address how to effectively integrate visual and textual information in image caption generation, specifically how to enhance model performance through high-order feature interactions. A new attention module, the X-Linear Attention Block, is proposed. This module fully utilizes bilinear pooling to capture second-order interactions between either single-modal or multimodal features, and by stacking multiple attention blocks, it simulates higher-order and even infinite-order feature interactions. On the COCO dataset, the newly proposed X-Linear Attention Network (X-LAN) achieved unprecedented performance, particularly on the CIDEr evaluation metric, demonstrating its effectiveness in handling complex multimodal reasoning tasks in image caption generation.

---

## LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding

Xu et al. (2022) introduced LayoutLMv2, a multi-modal pre-training model designed to enhance the understanding of visually-rich documents, particularly in handling interactions between text, layout, and image information. The model integrates an improved Transformer architecture, introducing a spatial-aware self-attention mechanism and new pre-training tasks (such as text-image alignment and matching) to learn deep interactions across modalities. LayoutLMv2 demonstrated significant performance improvements on various visually-rich document understanding tasks, setting new state-of-the-art results on datasets like FUNSD, CORD, and SROIE, showcasing its effectiveness and potential for application.

---

## Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning

In their research, Kuo and Kira address the limitations of relying solely on pre-trained object detectors for image captioning, which often miss crucial contextual relationships and scene details. They propose an enhanced approach utilizing the CLIP model for cross-modal retrieval of textual descriptions that supplement the detected visual content, and an image conditioning module to dynamically adjust feature representations. Their methodology significantly improves the descriptive accuracy, achieving a notable increase of 7.5% in CIDEr and 1.3% in BLEU-4 scores over state-of-the-art models, demonstrating its effectiveness in producing more contextually rich and grounded captions.

---

## Transformer-Based Multi-Modal Proposal and Re-Rank for Wikipedia Image-Caption Matching

In their research, Messina et al. address the challenge of matching images with captions on Wikipedia, where images often lack contextual descriptions. They introduce a two-stage Transformer-based model consisting of a Multi-modal Caption Proposal (MCProp) model for initial filtering and a Caption Re-Rank (CRank) model for fine-tuning the selections. Their approach achieved notable results, securing the fifth position in a Kaggle competition, demonstrating the effectiveness of their cascading models in improving the accessibility and relevance of multimedia content on large online platforms.

---

## Dual Memory Networks: A Versatile Adaptation Approach for Vision-Language Models

The authors aimed to address the challenge of effectively adapting pre-trained vision-language models (like CLIP) to various downstream tasks, specifically focusing on zero-shot, few-shot, and training-free few-shot adaptation scenarios. They introduced the Dual Memory Networks (DMN), comprising dynamic and static memory components, which allow the model to adapt to different task settings by storing and utilizing historical test data and training data features. The DMN outperformed existing methods by over 3% in the zero-shot setting and showed superior results even against methods that use external training data, proving its effectiveness across different scenarios.

---
## MeaCap: Memory-Augmented Zero-shot Image Captioning
The authors aim to address the limitations of zero-shot image captioning methods, which often suffer from hallucinations (imaginary information) or a lack of generalization when trained only on textual data. They propose the MeaCap framework, which introduces a memory-augmented mechanism using an external textual memory to retrieve and filter key concepts highly related to the given image, thereby generating more accurate and world-knowledge-enriched captions. The MeaCap framework outperforms existing methods in reducing hallucinations and improving caption accuracy across multiple benchmarks, particularly in zero-shot, in-domain, and cross-domain settings.

---
## Efficient Vision-Language Pre-training by Cluster Masking
The authors aimed to improve the efficiency and quality of vision-language pre-training by addressing the redundancy and inefficiency in current image masking strategies during contrastive learning. They introduced a cluster masking strategy, which masks clusters of visually similar image patches, thereby enhancing the representation learning by forcing the model to predict words for masked visual structures using context. The proposed method outperforms existing masking strategies like FLIP in terms of representation quality, while also speeding up the training process by reducing the amount of data used per image.

---
## SCSC: Spatial Cross-scale Convolution Module to Strengthen both CNNs and Transformers
The authors aimed to address the limitations of both CNNs and Transformers in capturing diverse spatial features efficiently, specifically targeting the inefficiency of large dense kernels and the challenges in capturing local features with large receptive fields. They introduced the Spatial Cross-scale Convolution Module (SCSC), which utilizes a wide range of kernel sizes and an efficient spatial embedding module to capture both microscopic and macroscopic feature representations while maintaining computational efficiency. The SCSC module demonstrated superior performance across various tasks, improving accuracy while significantly reducing computational costs and model parameters, making traditional CNNs competitive with advanced Transformers.