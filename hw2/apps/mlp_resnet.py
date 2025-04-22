import sys

sys.path.append("../python")
import os
import time

import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )

    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(
    dataloader: ndl.data.DataLoader, model: nn.Module, opt: ndl.optim.Optimizer = None
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    softmaxLoss = nn.SoftmaxLoss()
    if opt:  # training
        model.train()
    else:  # eval
        model.eval()
    total_err = 0
    total_loss = 0.0
    sample_size = 0
    for data, label in dataloader:
        bs = data.shape[0]
        sample_size += bs
        if opt:
            opt.reset_grad()
        logits = model(data)
        loss = softmaxLoss(logits, label)
        total_loss += loss.numpy() * bs
        predicted_class = np.argmax(logits.numpy(), -1)
        err = (predicted_class != label.numpy()).sum().item()
        total_err += err
        if opt:
            loss.backward()
            opt.step()
    res = total_err / sample_size, total_loss / sample_size
    return res

    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
    )
    t_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
    )
    train_dataloader = ndl.data.DataLoader(dataset, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(t_dataset, batch_size, shuffle=False)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    for _ in range(epochs):
        err, loss = epoch(train_dataloader, model, opt)
    test_err, test_loss = epoch(test_dataloader, model)
    return err, loss, test_err, test_loss   
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
