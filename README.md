# Adversarial-Robust-Image-Classification
Adversarially robust image classification using ResNet18 on 3x32x32 images (9 classes). Trained with PGD (ε=8/255, α=2/255, 10 steps) and 50/50 clean-adversarial batches. Uses SGD, cosine annealing, and label smoothing to balance clean accuracy and robustness under unseen attacks.
