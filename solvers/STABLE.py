import torch
from utils.evaluation import *
from utils.dataset import *
import wandb
import time
import torch.nn as nn



def stable_solver(args, train_loader, test_loader, val_loader):
    '''
    STABLE:
        T. Chen, et al. "A single-timescale method for stochastic bilevel optimization." PMLR, 2022.
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

    # initial variables: x,y  
    parameters = torch.randn((args.num_classes, 785), requires_grad=True,device=device)
    parameters = nn.init.kaiming_normal_(parameters, mode='fan_out')
    lambda_x = torch.zeros((args.training_size), requires_grad=True, device=device)

    val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
    test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
    # wandb log
    if args.track:
        wandb.log({"global_step":0, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": 0, "res":5})

    total_time = 0
    for epoch in range(args.epochs):
        start_time = time.time() 
        if epoch == 0:
            params_old = parameters
            H_xy = torch.zeros([args.batch_size, 7850],device=device)
            H_yy = torch.zeros([7850, 7850],device=device)
        beta_k = args.inner_lr
        alpha_k = args.outer_lr
        tao = 0.5
        
        val_index = torch.randperm(args.validation_size//args.batch_size)
        data_list = build_val_data(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)

        datas, ind_yy, ind_xy = data_list
        hparams = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
        if epoch==0:
            hparams_old = hparams
        else:
            hparams_old = hparams_old[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
        hparams_yy = lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size]

        params_old, parameters, outer_update, H_xy, H_yy = stable(args, datas, params_old, parameters, 
            hparams, hparams_old, H_xy, H_yy, tao, beta_k, alpha_k, out_f, reg_f,hparams_yy)

        hparams_old = lambda_x
        
        weight = hparams
        res = 0    

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


def stable(args, train_data_list, params_old, params, hparams, hparams_old, H_xy_old, H_yy_old, 
        tao, beta_k, alpha_k, out_f, reg_f,hparams_yy):
    data_list, labels_list = train_data_list
    # fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()
    v_temp = v_0

    # Hessian
    z_list = []
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy, output, reg_f) 

    G_gradient = torch.reshape(params, [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
    # G_gradient = torch.reshape(params[0], [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
    
    for _ in range(args.hessian_q):
    # for _ in range(args.K):
        Jacobian = torch.matmul(G_gradient, v_0)
        v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
        v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
        z_list.append(v_0)            
    v_Q = args.eta*(v_temp+torch.sum(torch.stack(z_list), dim=0))

    torch.autograd.grad(Jacobian, params)

    # Gyx_gradient
    output = out_f(data_list[2], params)
    Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams)[0]
    x_update = Gyx_gradient.detach()


    v_0 = torch.unsqueeze(torch.reshape(x_update, [-1]), 1).detach()
    # Gyx_gradient
    output = out_f(data_list[2], params)
    Gx_gradient = gradient_gx(args, labels_list[2], params, data_list[2], hparams, output, reg_f)
    Gx_gradient = torch.reshape(Gx_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gx_gradient, v_0.detach()), params)[0]
    v_0 = torch.unsqueeze(torch.reshape(-Gyx_gradient, [-1]), 1).detach()
   
    v_temp = v_0
    # Hessian
    z_list = []
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy, output, reg_f) 
    G_gradient = torch.reshape(params, [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
    # G_gradient = torch.reshape(params[0], [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
    
    for _ in range(args.hessian_q):
    # for _ in range(args.K):
        Jacobian = torch.matmul(G_gradient, v_0)
        v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
        v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
        z_list.append(v_0)            
    v_Q = args.eta*(v_temp+torch.sum(torch.stack(z_list), dim=0))

    torch.autograd.grad(Jacobian, params)

    temp = v_Q.detach()
    params_new = torch.reshape(params, [-1]) - beta_k*torch.reshape(Gy_gradient.detach(), [-1]) + alpha_k*temp.reshape([-1])
    params_new = torch.reshape(params_new, params.size()).detach()
    params_new.requires_grad_(True)
    H_xy, H_yy = 0, 0
    return params, params_new, -x_update, H_xy, H_yy
