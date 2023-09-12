if __name__ == '__main__':
    import torch
    import numpy as np
    from dataset_dementia import MTDTFUNDUS
    from sklearn.model_selection import train_test_split
    import json
    from tensorboardX import SummaryWriter
    from torch import optim
    import torch.nn.functional as F
    from metric.eval_metric import compute_acc, compute_auc
    from model.demographic_combine import net_cmr_mtdt
    import argparse

    parser = argparse.ArgumentParser(description='Only Demographic')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--n_cpus', type=int, default=6)
    parser.add_argument('--num_mtdt', type=int, default=36)

    args = parser.parse_args(['--num_mtdt', '36', '--n_classes', '1', ])

    torch.manual_seed(0)

    writer = SummaryWriter('results/log/fundus_demo')
    device = torch.device('cuda:0')

    model = net_cmr_mtdt(args)
    model = model.to(device)

    # Create datasets and loaders
    with open('../DATA/CSV/OCTmeasure/new1.json', 'r') as file:
        img_mask_list = json.loads(file.readline())
    train_filelists, val_filelists = train_test_split(img_mask_list, test_size=0.1, random_state=100)
    print("Total Nums: {}, train: {}, val: {}".format(len(img_mask_list), len(train_filelists), len(val_filelists)))
    train_data = MTDTFUNDUS(filelists=train_filelists)
    val_data = MTDTFUNDUS(filelists=val_filelists)

    train_batch_size = 4
    val_batch_size = 4

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=train_batch_size,
        num_workers=6,
        shuffle=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=val_batch_size,
        num_workers=6,
        shuffle=False)

    lr = 1e-3  # 学习率
    epochs = 500  # 训练轮次
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器

    # 把训练的过程定义为一个函数
    def train(model, optimizer, train_data, val_data, test_data,
              epochs):  # 输入：网络架构，优化器，损失函数，训练集，验证集，测试集，轮次
        best_auc, best_epoch = 0, 0  # 输出验证集中准确率最高的轮次和准确率
        model.train()
        for epoch in range(epochs):
            # epoch_train_losses = np.zeros(1, dtype=np.float32)
            # epoch_train_auc = np.zeros(1, dtype=np.float32)
            print('============第{}轮 train ============'.format(epoch + 1))
            avg_train_loss_list = []
            avg_train_auc_list = []
            train_list = []
            for steps, (x, features, y) in enumerate(train_loader):  # for x,y in train_data
                x = x / 255.
                x = x.to(device)
                y = y.to(device)
                features = features.to(device)

                logits = model(x, features)  # 数据放入网络中
                probab = torch.squeeze(torch.sigmoid(logits))
                loss = F.binary_cross_entropy(probab, y)  # 得到损失值
                optimizer.zero_grad()  # 优化器先清零，不然会叠加上次的数值
                loss.backward()  # 后向传播
                optimizer.step()
                avg_train_loss_list.append(loss.cpu().detach().numpy())

                train_acc = compute_acc(probab, y, train_batch_size)
                train_list.append(train_acc)

                if torch.all(y == 1) or torch.all(y == 0):
                    # print(epoch, steps, y)
                    continue
                auc = compute_auc(probab, y)
                avg_train_auc_list.append(auc)

            epoch_train_acc = np.mean(train_list)
            epoch_avg_train_losses = np.mean(avg_train_loss_list)
            epoch_avg_train_auc = np.mean(avg_train_auc_list)
            print('Epoch {}, losses {}  cls_auc {}  cls_acc{}'.format(epoch, epoch_avg_train_losses, epoch_avg_train_auc,
                                                                    epoch_train_acc))
            writer.add_scalar('train_acc/epoch', epoch_train_acc, epoch)
            writer.add_scalar('train_loss/epoch', epoch_avg_train_losses, epoch)
            writer.add_scalar('train_auc/epoch', epoch_avg_train_auc, epoch)

            ######################
            ##### Validation #####
            ######################
            if (epoch +1)  % 5 == 0:  # 这里可以设置每两次训练验证一次
                model.eval()
                # correct = 0
                avg_loss_list = []
                avg_acc_list = []
                avg_auc_list = []
                # total = len(val_loader.dataset)
                with torch.no_grad():
                    for step, (x, features, y) in enumerate(val_loader):
                        x = x / 255.
                        x = x.to(device)
                        y = y.to(device)
                        features = features.to(device)

                        logits = model(x, features)
                        probab = torch.squeeze(torch.sigmoid(logits))
                        loss = F.binary_cross_entropy(probab, y)
                        avg_loss_list.append(loss.cpu().detach().numpy())
                        val_acc = compute_acc(probab, y, val_batch_size)
                        avg_acc_list.append(val_acc)

                        # class_prob = torch.softmax(logits, axis=1)
                        # final_label = torch.argmax(class_prob, dim=1)
                        if torch.all(y == 1) or torch.all(y == 0):
                            # print(y)
                            continue
                        auc = compute_auc(probab, y)
                        avg_auc_list.append(auc)

                avg_val_loss = np.array(avg_loss_list).mean()
                avg_val_auc = np.array(avg_auc_list).mean()
                avg_val_acc = np.array(avg_acc_list).mean()

                # if val_acc > best_acc:  # 判断每次在验证集上的准确率是否为最大
                if avg_val_auc > best_auc:
                    best_epoch = epoch
                    # best_acc = avg_val_acc
                    best_auc = avg_val_auc
                    torch.save(model.state_dict(),
                               'results/model/fundus_demo/best_{}_{}.pth'.format(best_auc, best_epoch))  # 保存验证集上最大的准确率
                print('===========================eval ====================')
                # print('best acc:', best_acc, 'best_epoch:', best_epoch)
                print('Epoch {}, losses {}  cls_auc {}  cls_acc{}'.format(epoch, avg_val_loss, avg_val_auc, avg_val_acc))
                writer.add_scalar('val_acc/epoch', avg_val_acc, epoch)
                writer.add_scalar('val_loss/epoch', avg_val_loss, epoch)
                writer.add_scalar('val_auc/epoch', avg_val_auc, epoch)


    # 训练以及验证测试函数
    train(model=model, optimizer=optimizer, train_data=train_loader, val_data=val_loader,
          test_data=val_filelists, epochs=epochs)