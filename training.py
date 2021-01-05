def main():
    import pandas as pd
    model = FaceNetModel(pretrained=pretrain)
    model.to(device)
    triplet_loss = TripletLoss.to(device)


    model.unfreeze_all()


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    model = torch.nn.DataParallel(model) # ?? not sure if needed

    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        print(80 * '=')
        print('Epoch [{}/{}]'.format(epoch, args.num_epochs + start_epoch - 1))

        time0 = time.time()
        data_loaders, data_size = get_dataloader(args.train_root_dir, args.valid_root_dir,
                                                 args.train_csv_name, args.valid_csv_name,
                                                 args.num_train_triplets, args.num_valid_triplets,
                                                 args.batch_size, args.num_workers)

        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        print(f'  Execution time                 = {time.time() - time0}')
    print(80 * '=')


def save_last_checkpoint(state):
    torch.save(state, 'log/last_checkpoint.pth')


def train_valid(model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):
    for phase in ['train', 'valid']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1) ##dont understand
                    if len(hard_triplets[0]) == 0:
                        continue #überspringt schleife wenn diese bedingung erfüllt ist
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

                # pos_hard_cls = pos_cls[hard_triplets]
                # neg_hard_cls = neg_cls[hard_triplets]

                model.module.forward_classifier(anc_hard_img)
                model.module.forward_classifier(pos_hard_img)
                model.module.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = '_'.join(map(str, scheduler.get_lr()))
        layers = '+'.join(args.unfreeze.split(','))
        write_csv(f'log/{phase}.csv', [time, epoch, np.mean(accuracy), avg_triplet_loss, layers, args.batch_size, lr])

        if phase == 'valid':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.module.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  })
            save_if_best({'epoch': epoch,
                          'state_dict': model.module.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, np.mean(accuracy))
