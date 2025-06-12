# 문장 순서 예측 모델 (기존 train.csv의 ID 컬럼 사용)

import pandas as pd
import re
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab
from itertools import permutations
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. 데이터 불러오기 및 준비
# -----------------------------
df = pd.read_csv("open/train.csv")  # ID 컬럼 포함되어 있음

# 문장 묶기 및 정답 레이블 생성
perm_list = list(permutations([0, 1, 2, 3]))
df["sentences"] = df.apply(lambda row: [row[f"sentence_{i}"] for i in range(4)], axis=1)
df["label"] = df.apply(lambda row: [row[f"answer_{i}"] for i in range(4)], axis=1)
df["class_id"] = df["label"].apply(lambda x: perm_list.index(tuple(x)))

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# -----------------------------
# 2. 토크나이저 및 Vocab 생성
# -----------------------------
def tokenize(text):
    return re.findall(r'\w+', text.lower())

all_tokens = []
for sents in df["sentences"]:
    for sent in sents:
        all_tokens.extend(tokenize(sent))

counter = Counter(all_tokens)
vocab_obj = vocab(counter, specials=['<pad>', '<unk>'])
vocab_obj.set_default_index(vocab_obj['<unk>'])

def numericalize(text):
    return [vocab_obj[token] for token in tokenize(text)]

# -----------------------------
# 3. Dataset 정의
# -----------------------------
class SentenceOrderDataset(Dataset):
    def __init__(self, df, max_len=30):
        self.sentences = df["sentences"].tolist()
        self.labels = df["class_id"].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def pad(self, tokens):
        return tokens[:self.max_len] + [vocab_obj['<pad>']] * (self.max_len - len(tokens))

    def __getitem__(self, idx):
        encoded = [self.pad(numericalize(sent)) for sent in self.sentences[idx]]
        return torch.tensor(encoded), torch.tensor(self.labels[idx])

train_dataset = SentenceOrderDataset(train_df)
val_dataset = SentenceOrderDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# -----------------------------
# 4. 모델 정의 (LSTM 기반)
# -----------------------------
class OrderPredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_obj['<pad>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 24)
        )

    def forward(self, x):
        reps = []
        for i in range(4):
            emb = self.embed(x[:, i])
            _, (h_n, _) = self.lstm(emb)
            reps.append(h_n[-1])
        concat = torch.cat(reps, dim=1)
        return self.fc(concat)

# -----------------------------
# 5. 학습 루프
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OrderPredictionModel(len(vocab_obj)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# -----------------------------
# 6. 예측 및 제출 파일 생성 (기존 ID 사용)
# -----------------------------
model.eval()
preds = []
with torch.no_grad():
    for x, _ in val_loader:
        x = x.to(device)
        out = model(x)
        pred_class = out.argmax(dim=1)
        preds.extend(pred_class.cpu().tolist())

final_orders = [perm_list[i] for i in preds]

submission = pd.DataFrame({
    "ID": val_df["ID"].values,
    "answer_0": [p[0] for p in final_orders],
    "answer_1": [p[1] for p in final_orders],
    "answer_2": [p[2] for p in final_orders],
    "answer_3": [p[3] for p in final_orders]
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 저장 완료 (기존 ID 기준)")