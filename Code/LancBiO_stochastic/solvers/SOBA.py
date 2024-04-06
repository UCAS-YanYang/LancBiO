import torch
from utils.dataset import nositify,build_val_data_soba
from utils.evaluation import loss_train_avg,test_avg,out_f,reg_f,gradient_fy,gradient_gy
import wandb
import time
import torch.nn as nn



def soba_solver(args, train_loader, test_loader, val_loader):
    '''
    SOBA:
        M. Dagréou, P. Ablin, S. Vaiter and T. Moreau, "A framework for bilevel
        optimization that enables stochastic and global variance reduction
        algorithms", NeurIPS 2022.
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
    
    # initial variables: x,y,v
    parameters = torch.randn((args.num_classes, 785), requires_grad=True,device=device)
    parameters = nn.init.kaiming_normal_(parameters, mode='fan_out')
    lambda_x = torch.zeros((args.training_size), requires_grad=True, device=device)

    val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
    test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
    # wandb log
    if args.track:
        wandb.log({"global_step":0, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": 0, "res":5})
    v = torch.rand((parameters.numel(),1),device=device).reshape(-1)
    
    total_time = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        inner_lr = args.inner_lr

        # update v, update x
        val_index = torch.randperm(args.validation_size//args.batch_size)
        val_data_list, ind_yy = build_val_data_soba(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)
        hparams_yy = lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size]
        v, outer_update, inner_update, res = sobabio(v,parameters, hparams_yy,  val_data_list, args, out_f, reg_f,epoch)
        weight = lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size]
        
        parameters = parameters - (inner_lr*inner_update.reshape((10,785))).detach()
        outer_update = torch.squeeze(outer_update)
        with torch.no_grad():
            weight = weight - args.outer_lr*outer_update
            lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size] = weight

            end_time = time.time()
            total_time = end_time-start_time + total_time
            
            if epoch % args.test_fre == 0:
                val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
                test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
                # wandb log
                if args.track:
                    wandb.log({"global_step":epoch+1, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": total_time,"res": res})


def sobabio(v,params, hparams_yy, val_data_list, args, out_f, reg_f, epoch):
    '''
        subroutine to approximate the Hessian inverse vector product in SOBA
        args:
            Gy_gradient: the stored \nabla_y G, i.e., the stored computational graph
    '''
    data_list, labels_list = val_data_list
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    F_y = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

    # for SOBA, a single batch can be used to perform all updates at once, i.e., store the computational graph only once
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy, output, reg_f).reshape([-1])
    
    # reuse Gy_gradient to compute G_yy@v
    G_y_p_v = torch.matmul(Gy_gradient, v)
    G_yy_p_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1)


    if (epoch % args.test_fre == 0):
        G_y_p_v = torch.matmul(Gy_gradient, v)
        A_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1,1)
        res = torch.norm(A_v-F_y.reshape([-1,1]),2)
        # wandb.log({"global_step":epoch+1,"res": res})
    else:
        res = 0

    # reuse Gy_gradient to compute G_xy@v
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v.detach()), hparams_yy)[0]
    outer_update = -Gyx_gradient

    v = (v - args.eta*(G_yy_p_v.reshape(-1) - F_y.reshape(-1))).detach()

    return v, outer_update, Gy_gradient, res