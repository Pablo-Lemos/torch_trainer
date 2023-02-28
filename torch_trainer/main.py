import torch
import numpy as np
from tqdm.auto import trange


def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        x = x.to(device=device, dtype=dtype)
    else:
        print("Wrong data type")
        raise
    return x


class Trainer():
    def __init__(self, model, loss = None, learning_rate = 1e-3, optimizer=None,
                 batch_size=1, device=None, dtype=torch.float32):
        self._model = model

        if not loss:
            self._loss_fn = torch.nn.MSELoss(reduction='mean')
        else:
            self._loss_fn = loss

        if not optimizer:
            self._optimizer = torch.optim.Adam(self._model.parameters(),
                                               lr=learning_rate)
        else:
            self._optimizer = optimizer

        if not device:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

        self._batch_size = batch_size

        self._train_set = None
        self._test_set = None
        self._dtype = dtype

    def add_data(self, x, y, split = 0):
        x = to_tensor(x, self._device, self._dtype)
        y = to_tensor(y, self._device, self._dtype)
        dataset = torch.utils.data.TensorDataset(x, y)

        if split > 0:
            n = len(x)
            size_val = int(split*n)
            size_tr = n-size_val
            self._train_set, self._test_set = torch.utils.data.random_split(
                dataset, [size_tr, size_val])
        else:
            self._train_set = dataset

    def _make_trainloader(self, dataset):
        trainloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self._batch_size,
                                                  shuffle=True)
        return trainloader

    def train(self, num_epochs = 1000, patience = None):
        assert self._train_set is not None, 'No data has been loaded'
        early_stopping = patience is not None
        if early_stopping and (self._test_set is None):
            print('Trying to use early stopping, but no test data exists')
            raise
        losses_tr = []
        losses_val = []
        trainloader = self._make_trainloader(self._train_set)

        if early_stopping:
            num_steps_no_improv = 0
            best_loss = np.infty

        t = trange(num_epochs, desc='Batch', leave=True)
        for epoch in t:
            losses = []
            for x, y in trainloader:
                y_pred = self._model(x)
                loss = self._loss_fn(y_pred, y)

                self._model.zero_grad()
                loss.backward()
                self._optimizer.step()

                losses.append(loss.item())

            t.set_description(f"Epoch {epoch}")
            if self._test_set is None:
                t.set_postfix(loss = np.average(losses))
            else:
                x_test, y_test = self._test_set.dataset.tensors
                y_pred = self._model(x_test)
                test_loss = self._loss_fn(y_pred, y_test).item()
                t.set_postfix(loss=np.average(losses), test_loss=test_loss)

            if early_stopping:
                if test_loss < best_loss:
                    best_loss = test_loss
                    num_steps_no_improv = 0
                elif np.isfinite(test_loss):
                    num_steps_no_improv += 1

                if (num_steps_no_improv > patience):
                    print(f"Early stopping after {epoch} epochs.")
                    return None


if __name__ == "__main__":
    import os
    path = "/Users/pl332/Desktop/Chang/"
    pk_0 = np.load(os.path.join(path, "hod.quijote_LH.z0p5.lowz_sgc.p0k.npy"))
    thetas = np.load(os.path.join(path,
                                  "hod.quijote_LH.z0p5.lowz_sgc.thetas.npy"))

    model = torch.nn.Sequential(
        torch.nn.Linear(100, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )

    trainer = Trainer(model)
    trainer.add_data(x=pk_0, y=thetas, split=0.2)
    trainer.train(num_epochs=5)