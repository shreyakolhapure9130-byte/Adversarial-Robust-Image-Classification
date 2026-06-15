import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18, resnet34, resnet50
from google.colab import drive, files

# ── GPU check ─────────────────────────────────────────────────────────────────
subprocess.run(["nvidia-smi"], check=False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ── Local Storage Setup ───────────────────────────────────────────────────────
DRIVE_DIR       = "/content"
DRIVE_DATA_PATH = "/content/train.npz"
BEST_MODEL_PATH = "/content/model.pt"
CKPT_PATH       = "/content/checkpoint.pt"

os.makedirs(DRIVE_DIR, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
data   = np.load(DRIVE_DATA_PATH)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

print("Dataset size:", len(images))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

# ── Train/val split ───────────────────────────────────────────────────────────
full_dataset = TensorDataset(images, labels)
n_val   = int(len(full_dataset) * 0.1)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))

loader     = DataLoader(train_set, batch_size=128, shuffle=True,
                        num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set,   batch_size=256, shuffle=False,
                        num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_CLASSES = 9

# pick one of: resnet18, resnet34, resnet50
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# sanity check -- output shape must be (1, 9)
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32).to(DEVICE))
print("Output shape:", out.shape)

# ── Adversarial training helpers ──────────────────────────────────────────────
EPS       = 8 / 255
ALPHA     = 2 / 255
PGD_STEPS = 10

def augment(x):
    if torch.rand(1).item() > 0.5:
        x = x.flip(-1)
    x = F.pad(x, [4, 4, 4, 4], mode='reflect')
    i = torch.randint(0, 8, (1,)).item()
    j = torch.randint(0, 8, (1,)).item()
    return x[:, :, i:i+32, j:j+32]

def pgd_attack(model, x, y, eps, alpha, steps):
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - x, 1 - x)
    delta.requires_grad_(True)
    for _ in range(steps):
        loss = F.cross_entropy(model(x + delta), y)
        loss.backward()
        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(delta.data, 0 - x, 1 - x)
        delta.grad.zero_()
    return (x + delta).detach()

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
EPOCHS    = 100
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Resume from checkpoint if session was interrupted ─────────────────────────
START_EPOCH = 1
best_score  = 0.0

if os.path.exists(CKPT_PATH):
    print("Checkpoint found — resuming...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    best_score  = ckpt["best_score"]
    START_EPOCH = ckpt["epoch"] + 1
    print(f"Resumed from epoch {START_EPOCH - 1} | Best score so far: {best_score:.4f}")
else:
    print("No checkpoint found — starting from scratch.")

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(START_EPOCH, EPOCHS + 1):
    model.train()
    correct = total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 1. Augment the clean images
        x_clean = augment(x)

        # 2. Generate adversarial examples from the clean images
        model.eval()
        x_adv = pgd_attack(model, x_clean, y, EPS, ALPHA, PGD_STEPS)
        model.train()

        # 3. Combine 50% clean and 50% adversarial images into one batch
        x_combined = torch.cat([x_clean, x_adv], dim=0)
        y_combined = torch.cat([y, y], dim=0)

        # 4. Optimize on the mixed batch
        optimizer.zero_grad()
        logits = model(x_combined)
        loss   = criterion(logits, y_combined)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 5. Track training accuracy ONLY on the adversarial half for logging consistency
        adv_logits = logits[x_clean.size(0):]
        correct += (adv_logits.argmax(1) == y).sum().item()
        total   += y.size(0)

    scheduler.step()

    # ── Evaluate every 5 epochs ───────────────────────────────────────────────
    if epoch % 5 == 0 or epoch == 1:
        model.eval()

        clean_correct = clean_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                clean_correct += (model(x).argmax(1) == y).sum().item()
                clean_total   += y.size(0)
        clean_acc = clean_correct / clean_total

        rob_correct = rob_total = 0
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = pgd_attack(model, x, y, EPS, ALPHA, steps=20)
            with torch.no_grad():
                rob_correct += (model(x_adv).argmax(1) == y).sum().item()
            rob_total += y.size(0)
        robust_acc = rob_correct / rob_total

        score = 0.5 * clean_acc + 0.5 * robust_acc
        print(f"Epoch {epoch:3d} | Train adv acc: {correct/total:.3f} | "
              f"Clean: {clean_acc:.3f} | Robust: {robust_acc:.3f} | Score: {score:.3f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  → Saved best model to Drive (score={best_score:.4f})")
    else:
        print(f"Epoch {epoch:3d} | Train adv acc: {correct/total:.3f}")

    # ── Save full checkpoint to Drive after every epoch ───────────────────────
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_score": best_score,
    }, CKPT_PATH)

# ── Download best model to local machine ──────────────────────────────────────
print(f"\nDone. Best score: {best_score:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")
files.download(BEST_MODEL_PATH)