import torch
from utils.dataset import nositify,build_val_data_Qsample
from utils.evaluation import loss_train_avg,test_avg,out_f,reg_f,gradient_fy,gradient_gy
import wandb
import time
import torch.nn as nn



def amigo_gd_solver(args, train_loader, test_loader, val_loader):
    '''
    AmIGO:  
        M. Arbel and J. Mairal. "Amortized Implicit Differentiation for Stochastic
        Bilevel Optimization". ICLR 2022.
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
    v = torch.rand((parameters.numel(),1),device=device).reshape(-1)

    val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
    test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
    # wandb log
    if args.track:
        wandb.log({"global_step":0, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": 0, "res":5})
        
    
    total_time = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        inner_lr = args.inner_lr

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
        val_data_list, Qsamples_list, ind_yy, ind_xy = build_val_data_Qsample(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)
        v, outer_update,res = amigo_gd(v,parameters, lambda_x, val_data_list, Qsamples_list, ind_yy, ind_xy, args, out_f, reg_f,epoch)
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


def amigo_gd(v,params, lambda_x, val_data_list, Qsamples_list, ind_yy, ind_xy, args, out_f, reg_f,epoch):
    '''
    subroutine to approximate the Hessian inverse vector product by GD
    '''
    data_list, labels_list = val_data_list
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    F_y = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

    # As stated in AmIGO, it samples a new batch for each Q-iteration
    Qsamples_images, Qsamples_labels = Qsamples_list
    for s in range(args.hessian_q):
        hparams_yy = lambda_x[ind_yy[s]*args.batch_size: (ind_yy[s]+1)*args.batch_size]
        output = out_f(Qsamples_images[s], params)
        Gy_gradient = gradient_gy(args, Qsamples_labels[s], params, Qsamples_images[s], hparams_yy, output, reg_f).reshape([-1]) 
        G_y_p_v = torch.matmul(Gy_gradient, v)
        G_yy_p_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1)
        v = (v - args.eta*(G_yy_p_v.reshape(-1) - F_y.reshape(-1))).detach()

    if (epoch % args.test_fre == 0):
        G_y_p_v = torch.matmul(Gy_gradient, v)
        A_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1,1)
        res = torch.norm(A_v-F_y.reshape([-1,1]),2)
        # wandb.log({"global_step":epoch+1,"res": res})
    else:
        res = 0

    # Gyx_gradient
    hparams_xy = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_xy, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v), hparams_xy)[0]
    outer_update = -Gyx_gradient 
    
    return v, outer_update, res