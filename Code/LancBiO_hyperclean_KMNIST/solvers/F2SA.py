import torch
from utils.dataset import nositify,build_val_data
from utils.evaluation import loss_train_avg,test_avg,out_f,reg_f,gradient_fy, gradient_gy, gradient_gx
import wandb
import time
import torch.nn as nn


def f2sa_solver(args, train_loader, test_loader, val_loader):
    """
    F2SA:
        J. Kwon, D. Kwon, S. Wright and R. Noewak, "A Fully First-Order Method for
        Stochastic Bilevel Optimization", ICML 2023.
        
    Adapted from https://github.com/benchopt/benchmark_bilevel/blob/main/solvers/f2sa.py
    """

    step_sizes = [args.inner_lr, args.inner_lr, args.outer_lr, args.delta_lambda]
    lr_exp = [0, 0, 1/7, 1]
    penalty_lambda = args.lambda0


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

    # initial variables: x,y,z 
    parameters = torch.randn((args.num_classes, 785), requires_grad=True,device=device)
    parameters_y = torch.randn((args.num_classes, 785), requires_grad=True,device=device)
    parameters = nn.init.kaiming_normal_(parameters, mode='fan_out')
    parameters_y = nn.init.kaiming_normal_(parameters_y, mode='fan_out')
    lambda_x = torch.zeros((args.training_size), requires_grad=True, device=device)
    val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
    test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)

    # wandb log
    if args.track:
        wandb.log({"global_step":0, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": 0, "res":5})

    total_time = 0
    for epoch in range(args.epochs):
        start_time = time.time() 
        cur_lr = update_lr(step_sizes, lr_exp, epoch)
        lr_inner, lr_approx_star, lr_outer, d_lambda = cur_lr
        train_index_list = torch.randperm(batch_num)
        for index in range(args.iterations):
            index_rn = train_index_list[index%batch_num]
            images, labels = images_list[index_rn], labels_list[index_rn]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = labels.to(device)
            weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
            output = out_f(images, parameters)
            inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f,False).detach()
            parameters = parameters - lr_inner*inner_update

            index_rn = train_index_list[index%batch_num]
            images, labels = images_list[index_rn], labels_list[index_rn]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = labels.to(device)
            weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
            # Gy_gradient
            output = out_f(images, parameters_y)
            inner_update = gradient_gy(args, labels_cp, parameters_y, images, weight, output, reg_f,False).detach()

            images, labels = images_val_list[index_rn], labels_val_list[index_rn]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = labels.to(device)
            # Fy_gradient
            output = out_f(images, parameters_y)
            Fy_gradient = gradient_fy(args, labels_cp, parameters_y, images, output).detach()
            parameters_y = parameters_y - lr_approx_star*(Fy_gradient +penalty_lambda*inner_update)

        val_index = torch.randperm(args.validation_size//args.batch_size)
        val_data_list, ind_z, ind_y = build_val_data(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)

        image_samples, label_samples = val_data_list

        hparams_xz = lambda_x[ind_z*args.batch_size: (ind_z+1)*args.batch_size]
        output = out_f(image_samples[1], parameters)
        outer_update_z = gradient_gx(args, label_samples[1], parameters, [], hparams_xz, output, reg_f, create_graph=False)
        weight_z = lambda_x[ind_z*args.batch_size: (ind_z+1)*args.batch_size]         
        outer_update_z = torch.squeeze(outer_update_z)


        hparams_xy = lambda_x[ind_y*args.batch_size: (ind_y+1)*args.batch_size]
        output = out_f(image_samples[2], parameters_y)
        outer_update_y = gradient_gx(args, label_samples[2], parameters_y, [], hparams_xy, output, reg_f, create_graph=False)
        weight_y = lambda_x[ind_y*args.batch_size: (ind_y+1)*args.batch_size]         
        outer_update_y = torch.squeeze(outer_update_y)

        with torch.no_grad():
            weight_z = weight + lr_outer*penalty_lambda*outer_update_z
            lambda_x[ind_z*args.batch_size: (ind_z+1)*args.batch_size] = weight_z

            weight_y = weight - lr_outer*penalty_lambda*outer_update_y
            lambda_x[ind_y*args.batch_size: (ind_y+1)*args.batch_size] = weight_y

            penalty_lambda = penalty_lambda + d_lambda

            end_time = time.time()
            total_time = end_time-start_time + total_time
            
            if epoch % args.test_fre == 0:
                res = 0
                val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
                test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
                # wandb log
                if args.track:
                    wandb.log({"global_step":epoch+1, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": total_time,"res": res})

            

def update_lr(step_sizes, lr_exp, epoch):
    """Update the learning rate according to exponents."""
    lr = [0,0,0,0]
    for i in range(4):
        lr[i] = step_sizes[i] / ((epoch + 1) ** lr_exp[i])
    return lr

