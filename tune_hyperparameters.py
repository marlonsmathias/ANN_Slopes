import wandb

from neural_network import *

if __name__ == '__main__':

    wandb.init()

    file_path = "../Data/01_Inicial_v8.xlsx"

    D = Data_loader(file_path,seed=0)

    input_size1 = 5
    output_size1 = 2
    input_size2 = 3
    output_size2 = 2

    NN = Neural_network(input_size1,output_size1,input_size2,output_size2)

    NN.set_norms(D.X1_norm,D.X2_norm,D.Y1_norm,D.Y2_norm)

    # Set hyperparameters
    n_epochs1 = 1000
    batch_size1 = 80
    n_epochs2 = 1000
    batch_size2 = 80
    n_epochs3 = 1000
    batch_size3 = 80

    lr1 = 10.**wandb.config.lr10
    lr2 = 10.**wandb.config.lr10
    lr3 = 10.**wandb.config.lr10

    scheduler_step1 = 100
    scheduler_gamma1 = wandb.config.gamma

    scheduler_step2 = 100
    scheduler_gamma2 = wandb.config.gamma

    scheduler_step3 = 100
    scheduler_gamma3 = wandb.config.gamma

    layers1 = [wandb.config.n1 for _ in range(wandb.config.d1)]
    layers2 = [wandb.config.n2 for _ in range(wandb.config.d2)]

    dropout1 = 0.2
    dropout2 = 0.2

    # Pre-train the first network
    NN.init_net1(layers=layers1, dropout=dropout1)

    loss1_train_pre = []
    loss1_val_pre = []

    opt1 = torch.optim.Adam(NN.net1.parameters(),lr=lr1)
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=scheduler_step1, gamma=scheduler_gamma1)

    for epoch in range(n_epochs1):

        NN.net1.train()
        opt1.zero_grad()

        loss_train_epoch = 0
        loss_val_epoch = 0

        for X1, Y1_target in D.get_batch1(batch_size=batch_size1):
            Y1_pred = NN.net1(X1)
            loss = torch.nn.functional.mse_loss(Y1_target, Y1_pred, reduction='sum')
            loss.backward()
            opt1.step()

            loss_train_epoch += loss.item()

        NN.net1.eval()
        with torch.no_grad():
            for X1, Y1_target in D.get_batch1(batch_size=batch_size1,validation=True):
                Y1_pred = NN.net1(X1)
                loss = torch.nn.functional.mse_loss(Y1_target, Y1_pred, reduction='sum')

                loss_val_epoch += loss.item()

        loss_train_epoch = loss_train_epoch/(D.n_train*output_size1)
        loss_val_epoch = loss_val_epoch/(D.n_val*output_size1)

        loss1_train_pre.append(loss_train_epoch)
        loss1_val_pre.append(loss_val_epoch)

        sch1.step()

        wandb.log({
            'epoch': epoch, 
            'train_loss': loss_train_epoch, 
            'val_loss': loss_val_epoch
        })

        if epoch%50 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_train_epoch}, Validation loss: {loss_val_epoch}')


    # Train the second network
    NN.init_net2(layers=layers2, dropout=dropout2)

    loss1_train = []
    loss1_val = []
    loss2_train = []
    loss2_val = []
    loss_train = []
    loss_val = []

    opt2 = torch.optim.Adam(NN.net2.parameters(),lr=lr2)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=scheduler_step2, gamma=scheduler_gamma2)

    for epoch in range(n_epochs2):
        NN.train()
        opt2.zero_grad()

        loss_train_epoch = 0
        loss_val_epoch = 0

        loss1_train_epoch = 0
        loss1_val_epoch = 0

        loss2_train_epoch = 0
        loss2_val_epoch = 0

        for X, Y_target in D.get_batch2(batch_size=batch_size2):
            Y_pred = NN(X)
            loss1 = torch.nn.functional.mse_loss(Y_target[:,:output_size1], Y_pred[:,:output_size1], reduction='sum')
            loss2 = torch.nn.functional.mse_loss(Y_target[:,output_size1:], Y_pred[:,output_size1:], reduction='sum')
            loss = loss1 + loss2

            loss.backward()
            opt2.step()

            loss_train_epoch += loss.item()
            loss1_train_epoch += loss1.item()
            loss2_train_epoch += loss2.item()

        NN.eval()
        with torch.no_grad():
            for X, Y_target in D.get_batch2(batch_size=batch_size2,validation=True):
                Y_pred = NN(X)
                loss1 = torch.nn.functional.mse_loss(Y_target[:,:output_size1], Y_pred[:,:output_size1], reduction='sum')
                loss2 = torch.nn.functional.mse_loss(Y_target[:,output_size1:], Y_pred[:,output_size1:], reduction='sum')
                loss = loss1 + loss2

                loss_val_epoch += loss.item()
                loss1_val_epoch += loss1.item()
                loss2_val_epoch += loss2.item()

        loss1_train_epoch = loss1_train_epoch/(D.n_train2*output_size1)
        loss1_val_epoch = loss1_val_epoch/(D.n_val2*output_size1)
        loss2_train_epoch = loss2_train_epoch/(D.n_train2*output_size2)
        loss2_val_epoch = loss2_val_epoch/(D.n_val2*output_size2)
        loss_train_epoch = loss_train_epoch/(D.n_train2*(output_size1+output_size2))
        loss_val_epoch = loss_val_epoch/(D.n_val2*(output_size1+output_size2))

        loss1_train.append(loss1_train_epoch)
        loss1_val.append(loss1_val_epoch)
        loss2_train.append(loss2_train_epoch)
        loss2_val.append(loss2_val_epoch)
        loss_train.append(loss_train_epoch)
        loss_val.append(loss_val_epoch)

        sch2.step()

        wandb.log({
            'epoch': epoch+1000, 
            'train_loss': loss_train_epoch, 
            'val_loss': loss_val_epoch
        })

        if epoch%50 == 0:
            print(f'Epoch: {epoch+1000}, Loss: {loss_train_epoch} ({loss1_train_epoch} + {loss2_train_epoch}), Validation loss: {loss_val_epoch} ({loss1_val_epoch} + {loss2_val_epoch})')


    opt3 = torch.optim.Adam(NN.parameters(),lr=lr3)
    sch3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=scheduler_step3, gamma=scheduler_gamma3)

    for epoch in range(n_epochs3):
        NN.train()
        opt3.zero_grad()

        loss_train_epoch = 0
        loss_val_epoch = 0

        loss1_train_epoch = 0
        loss1_val_epoch = 0

        loss2_train_epoch = 0
        loss2_val_epoch = 0

        for X, Y_target in D.get_batch2(batch_size=batch_size3):
            Y_pred = NN(X)
            loss1 = torch.nn.functional.mse_loss(Y_target[:,:output_size1], Y_pred[:,:output_size1], reduction='sum')
            loss2 = torch.nn.functional.mse_loss(Y_target[:,output_size1:], Y_pred[:,output_size1:], reduction='sum')
            loss = loss1 + loss2

            loss.backward()
            opt3.step()

            loss_train_epoch += loss.item()
            loss1_train_epoch += loss1.item()
            loss2_train_epoch += loss2.item()

        NN.eval()
        with torch.no_grad():
            for X, Y_target in D.get_batch2(batch_size=batch_size3,validation=True):
                Y_pred = NN(X)
                loss1 = torch.nn.functional.mse_loss(Y_target[:,:output_size1], Y_pred[:,:output_size1], reduction='sum')
                loss2 = torch.nn.functional.mse_loss(Y_target[:,output_size1:], Y_pred[:,output_size1:], reduction='sum')
                loss = loss1 + loss2

                loss_val_epoch += loss.item()
                loss1_val_epoch += loss1.item()
                loss2_val_epoch += loss2.item()

        loss1_train_epoch = loss1_train_epoch/(D.n_train2*output_size1)
        loss1_val_epoch = loss1_val_epoch/(D.n_val2*output_size1)
        loss2_train_epoch = loss2_train_epoch/(D.n_train2*output_size2)
        loss2_val_epoch = loss2_val_epoch/(D.n_val2*output_size2)
        loss_train_epoch = loss_train_epoch/(D.n_train2*(output_size1+output_size2))
        loss_val_epoch = loss_val_epoch/(D.n_val2*(output_size1+output_size2))

        loss1_train.append(loss1_train_epoch)
        loss1_val.append(loss1_val_epoch)
        loss2_train.append(loss2_train_epoch)
        loss2_val.append(loss2_val_epoch)
        loss_train.append(loss_train_epoch)
        loss_val.append(loss_val_epoch)

        sch3.step()

        wandb.log({
            'epoch': epoch+2000, 
            'train_loss': loss_train_epoch, 
            'val_loss': loss_val_epoch
        })

        if epoch%50 == 0:
            print(f'Epoch: {epoch+2000}, Loss: {loss_train_epoch} ({loss1_train_epoch} + {loss2_train_epoch}), Validation loss: {loss_val_epoch} ({loss1_val_epoch} + {loss2_val_epoch})')

    model_name = f'models/model_lr{wandb.config.lr10}_g{wandb.config.gamma}_n{wandb.config.n1}_d{wandb.config.d1}_n{wandb.config.n2}_d{wandb.config.d2}.pt'

    NN.save(model_name)