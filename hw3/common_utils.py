"""
====================================================================
공통 유틸리티 모듈 (common_utils.py)
====================================================================
실험 A, B, C에서 공통으로 사용하는 함수들을 모아둔 모듈.

포함된 내용:
  1. 재현성 관련: 시드 고정 함수
  2. 학습/평가 루프 (model.eval() + torch.no_grad() 포함)
  3. 학습 기록(History) 로깅
  4. 시각화 유틸리티
     - Loss/Accuracy 곡선
     - Activation 분포 히스토그램
     - Dead ReLU 히트맵
     - Gradient flow 그래프
  5. 정량 비교표 생성

사용법:
  from common_utils import set_seed, train_one_epoch, evaluate, ...

평가기준 대응:
  - "실험 정확성 및 재현성"(20%) → set_seed 사용
  - "학습곡선 분석"(20%) → plot_loss_acc_curves
  - "Activation 분포 분석"(20%) → plot_activation_distribution, plot_dead_neuron_heatmap
  - "Optimizer 비교 분석"(20%) → 동일 함수로 여러 optimizer 결과 비교
====================================================================
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


# ====================================================================
# 1. 재현성 (Reproducibility)
# ====================================================================

def set_seed(seed: int = 42):
    """
    실험 재현성을 위한 시드 고정.

    같은 seed에서 같은 결과가 나와야 함 (평가기준: 재현성 20%).

    Args:
        seed (int): 고정할 시드 값. 기본값 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN의 비결정적 알고리즘 비활성화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """학습 디바이스 반환 (Colab에서 GPU가 켜져 있으면 cuda 사용)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================================
# 2. 학습 / 평가 루프
# ====================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    loss_type='ce'):
    """
    1 epoch 학습 수행.

    Args:
        model: 학습할 모델
        loader: DataLoader (train)
        loss_fn: 손실 함수 (nn.CrossEntropyLoss, nn.MSELoss 등)
        optimizer: 옵티마이저
        device: cpu/cuda
        loss_type: 'ce' (CrossEntropy) 또는 'mse' (MSE+softmax+one-hot)
                   MSE의 경우 출력에 softmax 적용 + 타겟을 one-hot으로 변환해야 함.

    Returns:
        avg_loss: epoch 평균 손실
        accuracy: epoch 정확도(%)
        avg_grad_norm: epoch 평균 gradient L2 norm (gradient flow 분석용)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []

    num_classes = None  # MSE 사용 시 one-hot encoding에 필요

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 입력이 이미지(2D)면 flatten
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        logits = model(inputs)

        # 손실 계산: CE는 logits 그대로, MSE는 softmax + one-hot 필요
        if loss_type == 'ce':
            loss = loss_fn(logits, targets)
        elif loss_type == 'mse':
            if num_classes is None:
                num_classes = logits.size(1)
            probs = torch.softmax(logits, dim=1)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=num_classes
            ).float()
            loss = loss_fn(probs, targets_onehot)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        loss.backward()

        # gradient norm 측정 (모든 파라미터의 L2 norm 합)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(total_norm ** 0.5)

        optimizer.step()

        # 통계
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    avg_grad_norm = float(np.mean(grad_norms))
    return avg_loss, accuracy, avg_grad_norm


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, loss_type='ce'):
    """
    검증/테스트 평가.
    *** model.eval() + torch.no_grad() 활용 (과제 명시 조건) ***

    Args:
        (train_one_epoch와 동일)

    Returns:
        avg_loss, accuracy(%)
    """
    model.eval()  # Dropout, BatchNorm 등을 평가 모드로
    total_loss = 0.0
    correct = 0
    total = 0
    num_classes = None

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)

        logits = model(inputs)

        if loss_type == 'ce':
            loss = loss_fn(logits, targets)
        elif loss_type == 'mse':
            if num_classes is None:
                num_classes = logits.size(1)
            probs = torch.softmax(logits, dim=1)
            targets_onehot = torch.nn.functional.one_hot(
                targets, num_classes=num_classes
            ).float()
            loss = loss_fn(probs, targets_onehot)

        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


def fit(model, train_loader, test_loader, loss_fn, optimizer,
        device, epochs, loss_type='ce', scheduler=None, verbose=True,
        log_every=1):
    """
    전체 학습 루프. 매 epoch마다 train/test 손실·정확도·grad_norm 기록.

    Args:
        scheduler: torch.optim.lr_scheduler 객체 (optional, 예: ExponentialLR)
        log_every: 몇 epoch마다 출력할지

    Returns:
        history (dict): epoch별 metric을 담은 dict
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss':  [], 'test_acc':  [],
        'grad_norm':  [], 'lr': []
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, gnorm = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, loss_type
        )
        te_loss, te_acc = evaluate(
            model, test_loader, loss_fn, device, loss_type
        )

        # scheduler step (ExponentialLR 등)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        history['grad_norm'].append(gnorm)
        history['lr'].append(current_lr)

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"[{epoch:3d}/{epochs}] "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:5.2f}% "
                  f"test_loss={te_loss:.4f} test_acc={te_acc:5.2f}% "
                  f"grad_norm={gnorm:.4f} lr={current_lr:.5f}")

    return history


# ====================================================================
# 3. 시각화: 학습 곡선
# ====================================================================

def plot_loss_acc_curves(histories: dict, title_prefix="",
                         figsize=(14, 4)):
    """
    여러 실험의 loss/accuracy 곡선을 한 그래프에 비교.

    Args:
        histories (dict): {label: history_dict}
            예: {'CrossEntropy': h1, 'MSE': h2}
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for label, h in histories.items():
        epochs = range(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs, h['train_loss'], label=f"{label} (train)")
        axes[0].plot(epochs, h['test_loss'],  label=f"{label} (test)",
                     linestyle='--')
        axes[1].plot(epochs, h['train_acc'], label=f"{label} (train)")
        axes[1].plot(epochs, h['test_acc'],  label=f"{label} (test)",
                     linestyle='--')
        axes[2].plot(epochs, h['grad_norm'], label=label)

    axes[0].set_title(f"{title_prefix} Loss vs Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title(f"{title_prefix} Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].set_title(f"{title_prefix} Gradient Norm vs Epoch")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("||grad||")
    axes[2].set_yscale('log')  # log scale로 vanishing 확인 용이
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ====================================================================
# 4. 시각화: Activation 분포 (실험 B용)
# ====================================================================

def register_activation_hooks(model):
    """
    nn.Sequential 모델의 각 레이어에 forward hook을 걸어 activation을 저장.

    Returns:
        activations (dict): {layer_name: tensor}  -- 매 forward마다 갱신됨
        hooks (list): hook 핸들 (나중에 remove하기 위함)
    """
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    # 활성화 함수 (ReLU, LeakyReLU, Sigmoid) 직후의 출력을 캡처
    target_types = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)
    idx = 0
    for module in model.modules():
        if isinstance(module, target_types):
            name = f"act{idx}_{type(module).__name__}"
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
            idx += 1

    return activations, hooks


def plot_activation_distribution(activations: dict, title="",
                                 bins=50, figsize=None):
    """
    각 레이어 activation 값의 히스토그램을 나란히 그림.

    Args:
        activations: register_activation_hooks가 반환한 dict
                     (forward 한번 돌린 직후 호출해야 데이터가 있음)
    """
    n = len(activations)
    if n == 0:
        print("No activations captured.")
        return
    if figsize is None:
        figsize = (4 * n, 3)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (name, t) in zip(axes, activations.items()):
        vals = t.cpu().numpy().flatten()
        ax.hist(vals, bins=bins, color='steelblue', edgecolor='white')
        ax.set_title(f"{name}\nmean={vals.mean():.3f} std={vals.std():.3f}")
        ax.set_xlabel("activation value")
        ax.grid(alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ====================================================================
# 5. 시각화: Dead ReLU 히트맵 (실험 B용)
# ====================================================================

def compute_dead_ratio(activations: dict, threshold: float = 1e-6):
    """
    각 레이어에서 'dead' 상태인 뉴런의 비율을 계산.
    Dead 뉴런: 모든 입력에 대해 출력이 0(또는 threshold 이하)인 뉴런.

    Returns:
        dead_info (dict): {layer_name: {'ratio': float, 'mask': np.ndarray}}
            mask는 각 뉴런이 dead인지 여부 (True/False)
    """
    dead_info = {}
    for name, t in activations.items():
        # t shape: (batch, num_neurons)
        arr = t.cpu().numpy()
        # 각 뉴런별로, 모든 샘플에 대해 |출력| <= threshold이면 dead
        neuron_max = np.abs(arr).max(axis=0)
        dead_mask = neuron_max <= threshold
        dead_info[name] = {
            'ratio': float(dead_mask.mean()),
            'mask': dead_mask,
            'activation_mean_per_neuron': arr.mean(axis=0),
        }
    return dead_info


def plot_dead_neuron_heatmap(activations: dict, title="Dead Neuron Heatmap"):
    """
    각 레이어의 뉴런 활성화 평균을 히트맵으로 시각화.
    검은(0에 가까운) 부분이 Dead 뉴런 후보.

    레이어마다 뉴런 수가 다를 수 있으므로 레이어별로 한 행씩 그림.
    """
    fig, axes = plt.subplots(len(activations), 1,
                             figsize=(12, 1.5 * len(activations)))
    if len(activations) == 1:
        axes = [axes]

    for ax, (name, t) in zip(axes, activations.items()):
        arr = t.cpu().numpy()  # (batch, num_neurons)
        # 뉴런별 평균 활성화 (1 x num_neurons 형태로 표시)
        neuron_mean = arr.mean(axis=0, keepdims=True)
        im = ax.imshow(neuron_mean, aspect='auto', cmap='hot')
        ax.set_yticks([])
        ax.set_xlabel("Neuron index")
        # dead 비율도 같이 표시
        dead_ratio = (np.abs(arr).max(axis=0) <= 1e-6).mean() * 100
        ax.set_title(f"{name}  (Dead ratio: {dead_ratio:.1f}%)")
        plt.colorbar(im, ax=ax)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ====================================================================
# 6. 시각화: Gradient Flow (레이어별 grad norm)
# ====================================================================

def plot_gradient_flow(model, title="Gradient Flow per Layer"):
    """
    backward() 직후에 호출. 각 파라미터(레이어)별 gradient의 평균 abs 값을
    bar chart로 표시. Vanishing/Exploding gradient 확인용.

    *** backward() 호출 후, optimizer.step() 전에 호출할 것 ***
    """
    layer_names = []
    avg_grads = []
    max_grads = []

    for name, p in model.named_parameters():
        if p.grad is not None and 'weight' in name:
            layer_names.append(name.replace('.weight', ''))
            avg_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    plt.figure(figsize=(10, 4))
    x = np.arange(len(layer_names))
    plt.bar(x - 0.2, avg_grads, width=0.4, label='mean |grad|', alpha=0.8)
    plt.bar(x + 0.2, max_grads, width=0.4, label='max |grad|', alpha=0.8)
    plt.xticks(x, layer_names, rotation=30, ha='right')
    plt.yscale('log')  # log scale로 vanishing 잘 보이게
    plt.ylabel("Gradient magnitude (log)")
    plt.title(title)
    plt.legend(); plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# ====================================================================
# 7. 정량 비교표 생성
# ====================================================================

def summarize_histories(histories: dict, convergence_threshold: float = None):
    """
    여러 실험 결과를 표(DataFrame)로 정리.

    Args:
        histories: {label: history}
        convergence_threshold: '수렴'으로 간주할 train_loss 임계값.
            None이면 각 실험에서 final_loss * 1.05 이하로 처음 도달한 epoch로 계산.

    Returns:
        pandas.DataFrame
    """
    rows = []
    for label, h in histories.items():
        final_test_acc = h['test_acc'][-1]
        best_test_acc = max(h['test_acc'])
        min_train_loss = min(h['train_loss'])

        # 수렴 epoch: train_loss가 (min_loss * 1.05) 이하로 처음 떨어진 epoch
        if convergence_threshold is None:
            thr = min_train_loss * 1.05
        else:
            thr = convergence_threshold
        conv_epoch = next(
            (i + 1 for i, v in enumerate(h['train_loss']) if v <= thr),
            len(h['train_loss'])
        )

        rows.append({
            'experiment': label,
            'final_test_acc(%)':  round(final_test_acc, 2),
            'best_test_acc(%)':   round(best_test_acc, 2),
            'min_train_loss':     round(min_train_loss, 4),
            'convergence_epoch':  conv_epoch,
        })
    return pd.DataFrame(rows)
