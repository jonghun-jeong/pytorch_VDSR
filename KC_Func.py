def adjust_leaning_rate(optimizer,epoch):
    lr = opt.lr *(0.1**(epoch//opt.step))
    return lr

def train(retraining_data_loader, optimizer, new_model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

     print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    new_model.train()
    for iteration, batch in enumerate(retraing_data_loader,1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(new_model(input), target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(new_model.parameters(),opt.clip)
        optimizer.step()

        if iteration%100 == 0:
            print("===> Epoch[{}] ({}/{]) : Loss: (:.10f)".format(epoch,iteration,len(training_data_loader), loss.data))

def save_checkpoint(new_model,epoch):
    model_out_path = "checkpoint/KC_model/" + "KC_model_epoch_{}.pth".format(epoch)
    state = {"epoch" : epoch, "new_model": new_model}
    if not os.path,exits("checkpoint/KC_model/"):
        os.mkdirs("checkpoint/KC_model/")

    torch.save(state, model_out_path)

    print("checkpoint saved to {}".format(model_out_path))

