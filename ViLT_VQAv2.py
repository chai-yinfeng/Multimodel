import os
import os
import json
import re
from typing import Optional
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig


# Define paths for train2014 and val2014
train_images_path = "/root/autodl-tmp/mscoco/train2014"
val_images_path = "/root/autodl-tmp/mscoco/val2014"
questions_train_path = "/root/autodl-tmp/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
annotations_train_path = "/root/autodl-tmp/vqa/v2_mscoco_train2014_annotations.json"
questions_val_path = "/root/autodl-tmp/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
annotations_val_path = "/root/autodl-tmp/vqa/v2_mscoco_val2014_annotations.json"

# load questions
print("loading questions")
data_questions_train = json.load(open(questions_train_path))
data_questions_val = json.load(open(questions_val_path))

questions_train = data_questions_train['questions']
print("Number of train questions:", len(questions_train))
questions_val = data_questions_val['questions']
print("Number of validation questions:", len(questions_val))

# map between image IDs and their corresponding filenames
filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")

# source: https://github.com/allenai/allennlp-models/blob/a36aed540e605c4293c25f73d6674071ca9edfc3/allennlp_models/vision/dataset_readers/vqav2.py#L141
def id_from_filename(filename: str) -> Optional[int]:
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

print("mapping file names")
file_names_train = [f for f in tqdm(listdir(train_images_path)) if isfile(join(train_images_path, f))]
file_names_val = [f for f in tqdm(listdir(val_images_path)) if isfile(join(val_images_path, f))]

# create 2 dictionaries, one that maps filenames to their IDs and one the other way around
filename_to_id_train = {train_images_path + "/" + file: id_from_filename(file) for file in file_names_train}
id_to_filename_train = {v:k for k,v in filename_to_id_train.items()}

filename_to_id_val = {val_images_path + "/" + file: id_from_filename(file) for file in file_names_val}
id_to_filename_val = {v:k for k,v in filename_to_id_val.items()}

# load annotations
data_annotations_train = json.load(open(annotations_train_path))
data_annotations_val = json.load(open(annotations_val_path))

print("loading annotations")
annotations_train = data_annotations_train['annotations']
print("Number of train annotations:", len(annotations_train))
annotations_val = data_annotations_val['annotations']
print("Number of validation annotations:", len(annotations_val))

# Add labels + scores
config = ViltConfig.from_pretrained("/root/autodl-tmp/vilt-b32-finetuned-vqa")

def get_score(count: int) -> float:
    return min(1.0, count / 3)

# train
print("adding labels and scores for train annotations")
for annotation in tqdm(annotations_train):
    answers = annotation['answers']
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1
    labels = []
    scores = []
    for answer in answer_count:
        if answer not in list(config.label2id.keys()):
            continue
        labels.append(config.label2id[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    annotation['labels'] = labels
    annotation['scores'] = scores

# validation
print("adding labels and scores for validation annotations")
for annotation in tqdm(annotations_val):
    answers = annotation['answers']
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1
    labels = []
    scores = []
    for answer in answer_count:
        if answer not in list(config.label2id.keys()):
            continue
        labels.append(config.label2id[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    annotation['labels'] = labels
    annotation['scores'] = scores

# Create PyTorch dataset
class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations, processor, dataset_type):
        self.questions = questions
        self.annotations = annotations
        self.processor = processor
        self.dataset_type = dataset_type

        # Load the appropriate id_to_filename mapping based on dataset type
        if dataset_type == "train":
            self.id_to_filename = id_to_filename_train
        elif dataset_type == "val":
            self.id_to_filename = id_to_filename_val
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}. Use 'train' or 'val'.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        image = Image.open(self.id_to_filename[annotation['image_id']])
        if image.mode != "RGB":# 确保图片拥有3D的维度，即对灰度图像也增添三通道的冗余信息
            image = image.convert("RGB")
        text = questions['question']

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        # add labels
        labels = annotation['labels']
        scores = annotation['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets

        return encoding

processor = ViltProcessor.from_pretrained("/root/autodl-tmp/vilt-b32-finetuned-vqa")# vilt-b32-mlm

train_dataset = VQADataset(questions=questions_train, annotations=annotations_train, processor=processor, dataset_type="train")
val_dataset = VQADataset(questions=questions_val, annotations=annotations_val, processor=processor, dataset_type="val")

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained("/root/autodl-tmp/vilt-b32-finetuned-vqa",# vilt-b32-mlm
    id2label=config.id2label, 
    label2id=config.label2id)
model.to(device)


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=8, shuffle=True)

# start to train
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    """
    多标签任务的准确率计算，预测值只需匹配 labels 中的任意一个正确类别即可视为正确。
    :param outputs: 模型的输出 logits，形状为 [batch_size, 3129]
    :param labels: 多热编码目标张量，形状为 [batch_size, 3129]
    :return: (正确预测的样本数, 总样本数)
    """
    predictions = torch.argmax(outputs.logits, dim=-1)  # 每个样本的预测类别索引
    correct = 0
    total = labels.size(0)  # 总样本数

    for pred, label_row in zip(predictions, labels):
        # 获取 labels 中值为 1 的类别索引
        valid_labels = (label_row == 1).nonzero(as_tuple=True)[0]
        # 检查预测类别是否在有效标签集合中
        if pred.item() in valid_labels:
            correct += 1

    return correct, total

print("start training")

# 初始化日志文件
log_file = "./training_log-1.json"
log_data = []

# 如果日志文件已存在，加载现有数据
try:
    with open(log_file, "r") as f:
        log_data = json.load(f)
except FileNotFoundError:
    pass  # 文件不存在时忽略


num_epochs = 5
for epoch in range(num_epochs):
    # Training loop
    model.train()
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, train_correct, train_total = 0, 0, 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calculate training accuracy
        correct, total = calculate_accuracy(outputs, batch["labels"].to(device))
        train_correct += correct
        train_total += total

    train_accuracy = train_correct / train_total
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_dataloader):.4f}, Training Accuracy: {train_accuracy:.4f}")
    
    # Validation loop
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            val_loss += outputs.loss.item()

            # Calculate validation accuracy
            correct, total = calculate_accuracy(outputs, batch["labels"].to(device))
            val_correct += correct
            val_total += total

    val_accuracy = val_correct / val_total
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 将当前 epoch 的数据记录到字典中
    epoch_log = {
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_dataloader),
        "train_accuracy": train_accuracy,
        "val_loss": val_loss / len(val_dataloader),
        "val_accuracy": val_accuracy
    }
    log_data.append(epoch_log)

    # 将日志写入 JSON 文件
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)