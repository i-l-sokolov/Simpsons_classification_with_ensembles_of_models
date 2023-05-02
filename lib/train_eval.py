from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch


def fit_epoch(model, optimizer, criterion, train_dataloader, device, desc):
    train_loss = 0
    acc = 0
    model.train()
    train_bar = tqdm(train_dataloader, position=1, leave=False, desc=desc)
    for X, y in train_bar:

        X = X.to(device, dtype=torch.float)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(X)

        loss = criterion(y_pred, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        #         writer.add_scalar('Losses/train', loss.item(), step)
        acc += accuracy_score(y_pred.argmax(axis=1).detach().cpu().numpy(), y.detach().cpu().numpy())

    train_bar.close()

    train_loss /= len(train_dataloader)
    acc /= len(train_dataloader)
    return train_loss, acc


def eval_model(model, val_dataloader, criterion, device, desc):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        acc = 0
        wrong = []

        val_bar = tqdm(val_dataloader, position=1, leave=False, desc=desc)

        for X,y in val_bar:

            X = X.to(device, dtype=torch.float)
            y = y.to(device)
            y_preds = model(X)
            val_loss += criterion(y_preds, y).item()
            acc += accuracy_score(y_preds.argmax(axis=1).detach().cpu().numpy(),y.detach().cpu().numpy())
            wrong += y[torch.argmax(y_preds,axis=1) != y
                       ]
        val_bar.close()

    val_loss /= len(val_dataloader)
    acc /= len(val_dataloader)
    return val_loss, acc, wrong


def model_choosing(model, epochs, name, optimizer, criterion, scheduler, device, val_dataloader, train_dataloader):
    val_loss, best_acc, wrong = eval_model(model, val_dataloader, criterion, device, desc=f'Priming validation {name}')
    torch.save(model.state_dict(), f'../trained_models/{name}.pth')
    best_epoch = 0
    best_wrong = wrong

    epochs_bar = tqdm(range(epochs), desc=f'Fitting {name}', leave=True, position=0)

    for epoch in epochs_bar:

        train_loss, train_acc = fit_epoch(model, optimizer, criterion, train_dataloader, device, desc=f'Train {name} {epoch} epoch best_acc {best_acc:.4f}')

        val_loss, acc, wrong = eval_model(model, val_dataloader,criterion,device, desc=f'Valid {name} {epoch} epoch best_acc {best_acc:.4f}')
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_wrong = wrong
            torch.save(model.state_dict(), f'../trained_models/{name}.pth')

    epochs_bar.close()

    print(f'The best result for {name} was achieved on {best_epoch+1} epoch with valid acc {best_acc}')
    return best_acc, best_wrong