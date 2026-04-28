from datasets import load_dataset
from transformers import ViTImageProcessor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
from torch.utils.data import DataLoader
import torch
from transformers import TrainingArguments, Trainer, ViTForImageClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image
from torch.optim import SGD

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

### dataset
def process_example(example):
    image_path, label = example['text'].split()
    return {
        'image_path': image_path,
        'label': int(label)
    }

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(Image.open(image_path).convert("RGB")) for image_path in examples['image_path']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(Image.open(image_path).convert("RGB")) for image_path in examples['image_path']]
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_dataset = load_dataset('text', data_files='/data/liuji/projects/nsfw_detection/train.txt')
train_dataset = train_dataset.map(process_example)['train']
val_dataset = load_dataset('text', data_files='/data/liuji/projects/nsfw_detection/val.txt')
val_dataset = val_dataset.map(process_example)['train']
id2label = {0:'low', 1:'mid', 2:'high'}
label2id = {'low':0, 'mid':1, 'high':2}


### model
processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]
# size = 384
print('model size: ', size)

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            Resize((size, size)),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize((size, size)),
            # CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )



train_dataset.set_transform(train_transforms)
val_dataset.set_transform(val_transforms)



model = ViTForImageClassification.from_pretrained('Falconsai/nsfw_image_detection',
                                                  image_size=size,
                                                  id2label=id2label,
                                                  label2id=label2id,
                                                  ignore_mismatched_sizes=True)


metric_name = "accuracy"

args = TrainingArguments(
    "run/nsfw_finetune_1e-4_224",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=10,
    learning_rate=1e-4,
    warmup_steps=0,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=50,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
    dataloader_num_workers=8,
    weight_decay=0.01,
    fp16=True,
)

# optimizer = SGD(
#     model.parameters(),
#     lr=args.learning_rate,
#     momentum=0.9
# )
# num_training_steps = len(train_dataset) * args.num_train_epochs
# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=500,
#     num_training_steps=num_training_steps
# )
trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    # optimizers=(optimizer, scheduler)
)
trainer.train()
outputs = trainer.predict(val_dataset)
print(outputs.metrics)
