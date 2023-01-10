import argparse
import pdb
import sys

import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel

sys.path.append('./src')
from data.data import CorruptMnist

#pdb.set_trace()

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr, type(lr))
    if type(lr) is not float:
        lr = float(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    # train_set, _ = mnist()
    train_set = CorruptMnist(train=True)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 5
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to("cpu"))
            loss = criterion(preds, y.to("cpu"))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
    torch.save(model.state_dict(), "trained_model.pt")

    plt.plot(loss_tracker, "-")
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.savefig("training_curve.png")

    return model


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    test_set = CorruptMnist(train=False)

    dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)

    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch

        preds = model(x.to("cpu"))
        preds = preds.argmax(dim=-1)

        correct += (preds == y.to("cpu")).sum().item()
        total += y.numel()

    print(f"Test set accuracy {correct/total}")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
