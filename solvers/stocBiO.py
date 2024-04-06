import torch
from utils.evaluation import *
from utils.dataset import *
import wandb
import time
import torch.nn as nn
from utils.dataset import *
from utils.evaluation import *


def stocbio_solver(args, train_loader, test_loader, val_loader):
    '''
    stocBiO:
        K. Ji, J. Yang and Y. Liang. "Bilevel Optimization: Convergence Analysis
        and Enhanced Design". ICML 2021.
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
        train_index_list = torch.randperm(batch_num)
        for index in range(args.iterations):
            index_rn = train_index_list[index%batch_num]
            images, labels = images_list[index_rn], labels_list[index_rn]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = labels.to(device)
            weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
            output = out_f(images, parameters)
            inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f,False).detach()
            parameters = parameters - args.inner_lr*inner_update

        val_index = torch.randperm(args.validation_size//args.batch_size)
        val_data_list, ind_yy, ind_xy = build_val_data(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)
        hparams_yy = lambda_x[ind_yy*args.batch_size: (ind_yy+1)*args.batch_size]
        hparams_xy = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
        outer_update, res = stocbio(parameters, hparams_yy,  hparams_xy, val_data_list, args, out_f, reg_f,0,epoch)
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

def stocbio(params, hparams_yy, hparams_xy, val_data_list, args, out_f, reg_f,correction=0,epoch=0):
    '''
    subroutine to approximate the Hessian inverse vector product by Neumann series
    '''
    data_list, labels_list = val_data_list
    
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()
    
    if torch.is_tensor(correction):
        correction =  torch.unsqueeze(torch.reshape(correction, [-1]), 1).detach()

    v_0 = v_0 - correction
    v_temp = v_0

    # Hessian vector product
    z_list = []
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy, output, reg_f) 

    G_gradient = torch.reshape(params, [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
    
    for _ in range(args.hessian_q):
        Jacobian = torch.matmul(G_gradient, v_0)
        v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
        v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
        z_list.append(v_0)            
    v_Q = args.eta*(v_temp+torch.sum(torch.stack(z_list), dim=0))


    if  (epoch % args.test_fre == 0):
        v = v_Q.detach()
        F_y = Fy_gradient.detach()
        Gy_gradient = torch.reshape(Gy_gradient, [-1])
        G_y_p_v = torch.matmul(Gy_gradient, v)
        A_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1,1)
        res = torch.norm(A_v-F_y.reshape([-1,1]),2)
    else:
        res = 0
    
    # release the computational graph
    torch.autograd.grad(Jacobian, params)

    output = out_f(data_list[2], params)
    Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams_xy, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams_xy)[0]
    outer_update = -Gyx_gradient 


    return outer_update,res


