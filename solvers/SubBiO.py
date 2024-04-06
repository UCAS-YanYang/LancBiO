import torch
from utils.evaluation import *
from utils.dataset import *
import wandb
import time
import torch.nn as nn


def subbio_solver(args, train_loader, test_loader, val_loader):
    '''
    SubBiO:
        LancBiO: Dynamic Lanczos-aided Bilevel Optimization via Krylov Subspace.
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # data preprocessing
    batch_num = args.training_size//args.batch_size

    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(train_loader):
        images_list.append(images)
        labels = nositify(labels, args.noise_rate, args.num_classes)
        labels_list.append(labels)


    images_val_list, labels_val_list = [], []
    for index, (images, labels) in enumerate(val_loader):
        images_val_list.append(images)
        labels_val_list.append(labels)
    train_index_list = torch.randperm(batch_num)
    
    # initial variables: x,y,v
    parameters = torch.randn((args.num_classes, 785), requires_grad=True,device=device)
    parameters = nn.init.kaiming_normal_(parameters, mode='fan_out')
    lambda_x = torch.zeros((args.training_size), requires_grad=True, device=device)
    v = torch.rand((parameters.numel(),1),device=device)

    val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
    test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
    # wandb log
    if args.track:
        wandb.log({"global_step":0, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": 0, "res":5})

    total_time = 0

    for epoch in range(args.epochs):
        start_time = time.time() 
        inner_lr = args.alpha/torch.pow(torch.tensor(1+epoch),args.a)
        args.outer_lr = args.beta/torch.pow(torch.tensor(1+epoch),args.b)
        train_index_list = torch.randperm(batch_num)
        for index in range(args.iterations):
            index_rn = train_index_list[index%batch_num]
            images, labels = images_list[index_rn], labels_list[index_rn]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = labels.to(device)
            weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
            output = out_f(images, parameters)
            inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f,False).detach()
            parameters = parameters - inner_lr*inner_update


        # update v, update x
        val_index = torch.randperm(args.validation_size//args.batch_size)
        val_data_list, ind_yy, ind_xy = build_val_data(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)
        hparams_yy = lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size]
        hparams_xy = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
        v, outer_update, res = subbio(v,parameters, hparams_yy,  hparams_xy, val_data_list, args, out_f, reg_f,epoch)
        weight = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
                
        outer_update = torch.squeeze(outer_update)
        with torch.no_grad():
            weight = weight - args.outer_lr*outer_update
            lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size] = weight

            end_time = time.time()
            total_time = end_time-start_time + total_time
            
            if epoch % args.test_fre == 0:
                val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
                test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
                # wandb log
                if args.track:
                    wandb.log({"global_step":epoch+1, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": total_time,"res": res})

def subbio(v,params, hparams_yy, hparams_xy, val_data_list, args, out_f, reg_f,epoch):
    data_list, labels_list = val_data_list
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    F_y = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    G_gradient = torch.reshape(params, [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])

    G_y_p_v = torch.matmul(G_gradient, v)
    subspace_add = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1)

    # construct subspace using shimit orgothor
    W = torch.zeros((F_y.shape[0],2)).to('cuda')
    W[:,0] = (F_y/torch.norm(F_y,2)).squeeze().detach()
    W[:,1] = (subspace_add - torch.matmul(F_y.squeeze(),subspace_add)*F_y.squeeze()).squeeze()
    W[:,1] = W[:,1]/torch.norm(W[:,1],2).detach()

    WTb = W.T@F_y
    product_temp = torch.zeros_like(W)
    for i in range(2):
        G_y_p_v = torch.matmul(Gy_gradient, W[:,i])
        product_temp[:,i] = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1)
    WTAW = W.T@product_temp


    sub_hess_p_Fy = torch.linalg.solve(WTAW,WTb)
    v = (W@sub_hess_p_Fy).detach()

    if (epoch % args.test_fre == 0):
        G_y_p_v = torch.matmul(Gy_gradient, v)
        A_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1,1)
        res = torch.norm(A_v-F_y.reshape([-1,1]),2)
        # wandb.log({"global_step":epoch+1,"res": res})
    else:
        res = 0

    torch.autograd.grad(G_y_p_v, params)

    # Gyx_gradient
    output = out_f(data_list[2], params)
    Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams_xy, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v), hparams_xy)[0]
    outer_update = -Gyx_gradient 
    return v, outer_update, res
