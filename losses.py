import torch
from info_nce import InfoNCE
from pytorch_metric_learning.losses import NTXentLoss


CE = torch.nn.CrossEntropyLoss()


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


BCEL = torch.nn.BCEWithLogitsLoss()


def negative_sampling_contrastive_loss(v1, v2, labels):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    eye = torch.diag_embed(labels).to(v1.device)
    return (
        BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye),
        logits.diag() > 0,
    )


INCE = InfoNCE(temperature=0.1)


def info_nce_loss(v1, v2):
    return INCE(v1, v2) + INCE(v2, v1)


NTX = NTXentLoss(temperature=0.1)


def nt_xent_loss(v1, v2):
    logits = torch.cat((v1, v2), dim=0)
    labels = torch.arange(v1.shape[0], device=v1.device)
    labels = torch.cat((labels, labels), dim=0)
    return NTX(logits, labels)
