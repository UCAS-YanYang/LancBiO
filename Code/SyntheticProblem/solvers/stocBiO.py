import torch
import wandb
import time

def stocbio(args, P):

    n = args.n
    K = args.K
    lam = args.lam
    theta = args.theta

    x = torch.randn((n,1))
    y = torch.randn(n, 1)
    v = torch.randn(n, 1)

    total_time = 0
    for k in range(1,K+1):
        start_time = time.time()

        lam = max(args.min_lam, args.lam/(torch.pow(torch.Tensor([k]),args.b)))
        theta = max(args.min_theta, args.theta/(torch.pow(torch.Tensor([k]),args.a)))


        for _ in range(args.T):
            y = y - theta * P.g_grad_y(x,y)
        b = P.f_grad_y(x,y)

        # Neumann Series: Summed Hessian Inverse Approximation (SHIA)
        v = shia(x,y,b,P,args)
        
        x = x - lam*(P.f_grad_x(x,y)-P.g_jac_xy(x,y,v))

        iter_time = time.time() - start_time
        total_time += iter_time
        if k%args.test_fre == 0:
            val_loss = P.f_value(x,y)
            g_opt = torch.norm(P.g_grad_y(x,y),2)
            res = torch.norm(P.residual(x,y,v),2)
            hyper_esti = torch.norm(P.hyper_grad(x,y,v),2)
            print("At {} epochs f_value: {:.4f}, g_opt: {:.4f}, res: {:.14f}, 'hyper_grad': {:.14f}".format(k, val_loss, g_opt, res, hyper_esti))
            if args.track:
                wandb.log({"global_step":k, "val_loss": val_loss, 'g_opt':g_opt, 'res':res, 'time':total_time, 'hyper_grad':hyper_esti})


def shia(x,y,b,P,args):
    '''
        Summed Hessian Inverse Approximation via Neumann Series  
        A^{-1}b \approx \eta\sum_{i=0}^Q (I-\eta*A)^i@b 
    '''
    eta = args.theta
    tensor_list = [b,]
    v_0 = b
    for i in range(args.hessian_q):
        v_0 = v_0 - eta*P.g_hessian_y(x,y,v_0)
        tensor_list.append(v_0)
    stacked_tensor = torch.stack(tensor_list, dim=0)            # stack these tensors
    summed_tensor = torch.sum(stacked_tensor, dim=0)            # sum these tensors

    return eta*summed_tensor