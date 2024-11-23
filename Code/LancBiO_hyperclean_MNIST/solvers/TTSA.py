import torch
from utils.dataset import nositify,build_val_data
from utils.evaluation import loss_train_avg,test_avg,out_f,reg_f,gradient_fy,gradient_gy
import wandb
import time
import torch.nn as nn


def ttsa_solver(args, train_loader, test_loader, val_loader):
    '''
    TTSA:
        M. Hong, H.-T. Wai and Z. Yang. "A Two-Timescale Framework for Bilevel
        Optimization: Complexity Analysis and Application to Actor-Critic". SIAM
        Journal of Optimization. 2023.
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
        inner_lr = args.alpha/torch.pow(torch.tensor(1+epoch),args.a)
        args.outer_lr = args.beta/torch.pow(torch.tensor(1+epoch),args.b)
        train_index_list = torch.randperm(batch_num)
        
        index_rn = train_index_list[index%batch_num]
        images, labels = images_list[index_rn], labels_list[index_rn]
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        labels_cp = labels.to(device)
        weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
        output = out_f(images, parameters)
        inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f,False).detach()
        parameters = parameters - inner_lr*inner_update


        val_index = torch.randperm(args.validation_size//args.batch_size)
        val_data_list, ind_yy, ind_xy = build_val_data(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)
        hparams_yy = lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size]
        hparams_xy = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
        outer_update = ttsa(parameters, hparams_yy,  hparams_xy, val_data_list, args, out_f, reg_f)
        weight = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]            
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



def ttsa(params, hparams_yy, hparams_xy, val_data_list, args, out_f, reg_f):
    '''
    subroutine to approximate the Hessian inverse vector product by Neumann series
    '''
    data_list, labels_list = val_data_list
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()
    
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy, output, reg_f) 
    G_gradient = torch.reshape(params, [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
    
    v_Q = hia(G_gradient,v_0, params,args.eta,args.hessian_q)

    # Gyx_gradient
    output = out_f(data_list[2], params)
    Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams_xy, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams_xy)[0]
    outer_update = -Gyx_gradient 

    return outer_update


def hia(G,v,params,eta,max_iter):
    '''
        Hessian Inverse Approximation via Neumann Series  
        A^{-1}b \approx \eta(I-\eta*A)^i@b, i\in [0,1,...,Q]
    '''
    p = torch.randint(high=max_iter, size=(1,))
    v_0 = v.detach()
    for _ in range(p):
        Jacobian = torch.matmul(G, v_0)
        v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
        v_0 = torch.reshape(v_new, [-1,1]).detach()
    return  max_iter*eta*v_0.detach()
