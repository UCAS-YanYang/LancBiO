import torch
from utils.dataset import nositify, build_val_data_lanc
from utils.evaluation import loss_train_avg,test_avg,out_f,reg_f,gradient_fy,gradient_gy
import wandb
import time
import torch.nn as nn

def lancbio_solver(args, train_loader, test_loader, val_loader):
    '''
    LancBiO:
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
    
    # initial variables: x,y; initial subspace W
    parameters = torch.randn((args.num_classes, 785), requires_grad=True,device=device)
    parameters = nn.init.kaiming_normal_(parameters, mode='fan_out')
    lambda_x = torch.zeros((args.training_size), requires_grad=True, device=device)

    W = torch.rand((parameters.numel(),1),device=device)
    F_y = W

    val_loss_avg = loss_train_avg(val_loader, parameters, device, batch_num)
    test_accuracy, test_loss_avg = test_avg(test_loader, parameters, device)
    # wandb log
    if args.track:
        wandb.log({"global_step":0, "accuracy":test_accuracy, "Val_loss": val_loss_avg, "Test_loss": test_loss_avg, "time": 0, "res":5})

    
    total_time = 0

    for epoch in range(args.epochs):
        if epoch < 500:
            args.inner_lr = 0.1
            args.dim_max = 30
        else:
            args.inner_lr  = 0.05
            args.dim_max = 50
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


        if (epoch+1) % args.dim_fre == 0:
            args.m = torch.ceil(torch.tensor(args.m * args.dim_inc)).clone().detach()
            args.m = torch.min(args.m,torch.tensor(args.dim_max)).clone().detach()


      
        # update v, update x
        val_index = torch.randperm(args.validation_size//args.batch_size)
        val_data_list, ind_yy1, ind_yy2, ind_xy = build_val_data_lanc(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device)
        hparams_yy1 = lambda_x[ind_yy1*args.batch_size: (ind_yy1+1)*args.batch_size]
        hparams_yy2 = lambda_x[ind_yy2*args.batch_size: (ind_yy2+1)*args.batch_size]
        hparams_xy = lambda_x[ind_xy*args.batch_size: (ind_xy+1)*args.batch_size]
        
        # restart
        if epoch%args.m == 0:
            W = F_y/torch.norm(F_y,2)
            q_last = 0
            lanc_beta = 0
            tridia_M = torch.empty(0, 0)
        outer_update, W, F_y, q_last, lanc_beta, tridia_M, v, res = lancbio_tridiag(W, parameters, hparams_yy1,  hparams_yy2, hparams_xy, val_data_list, args, out_f, reg_f,q_last,lanc_beta,tridia_M,epoch)
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


def lancbio_tridiag(W,params, hparams_yy1, hparams_yy2, hparams_xy, val_data_list, args, out_f, reg_f,q_last,lanc_beta,tridia_M,epoch):
    '''
    subroutine to approximate the Krylov subspace by constructing a tridiagonal matrix
    '''
    data_list, labels_list = val_data_list
    # Fy_gradient
    output = out_f(data_list[0], params)
    Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
    F_y = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams_yy1, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])


    beta = lanc_beta
    # Lanczos process
    q = W[:,-1].reshape(-1,1)
    G_y_p_q = torch.matmul(Gy_gradient, q)
    A_q = torch.autograd.grad(G_y_p_q, params, retain_graph=True)[0].reshape(-1,1)
    a = q.T@A_q
    q_next = A_q - a*q - lanc_beta*q_last
    lanc_beta = torch.norm(q_next,2)
    q_next = q_next/lanc_beta
    q_last = q
    W = torch.cat((W, q_next), dim=1)

    # Construct tridiagonal projection matrix
    new_matrix = torch.zeros((tridia_M.shape[0]+1,tridia_M.shape[1]+1,), device='cuda')
    new_matrix[:-1,:-1] = tridia_M
    new_matrix[-1,-1] = a
    if new_matrix.shape[0]>1:
        new_matrix[-1,-2]=beta
        new_matrix[-2,-1]=beta
    tridia_M = new_matrix

    if W.shape[1] == args.m+1:
        W = W[:,1:]
        tridia_M = tridia_M[1:,1:]

    WTb = W[:,:-1].T@F_y
    sub_hess_p_Fy = torch.linalg.solve(tridia_M,WTb)
    v = (W[:,:-1].reshape(7850,-1)@sub_hess_p_Fy).detach()

    if (epoch % args.test_fre == 0):
        G_y_p_v = torch.matmul(Gy_gradient, v)
        A_v = torch.autograd.grad(G_y_p_v, params, retain_graph=True)[0].reshape(-1,1)
        res = torch.norm(A_v-F_y.reshape([-1,1]),2)
        # wandb.log({"global_step":epoch+1,"res": res})
    else:
        res = 0
        
    # Gyx_gradient
    output = out_f(data_list[3], params)
    Gy_gradient = gradient_gy(args, labels_list[3], params, data_list[3], hparams_xy, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v), hparams_xy)[0]
    outer_update = -Gyx_gradient 
    
    return outer_update, W, F_y, q_last, lanc_beta, tridia_M, v, res
