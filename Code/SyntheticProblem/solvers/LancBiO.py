import torch
import wandb
import time

def lancbio(args,P):
    n = args.n
    K = args.K
    lam = args.lam
    theta = args.theta

    # LancBiO dimension parameters
    m = args.m
    dim_max = args.dim_max
    dim_fre = args.dim_fre
    dim_incre = args.dim_inc


    x = torch.randn((n,1))
    y = torch.randn(n, 1)
    v = torch.randn(n, 1)

    delta_v = 0                     # corection initialization
    h = -1
    
    total_time = 0
    for k in range(1,K+1):
        start_time = time.time()

        lam = max(args.min_lam, args.lam/(torch.pow(torch.Tensor([k]),args.b)))
        theta = max(args.min_theta, args.theta/(torch.pow(torch.Tensor([k]),args.a)))


        for _ in range(args.T):
            y = y - theta * P.g_grad_y(x,y)
        b = P.f_grad_y(x,y)


        if (k) % dim_fre == (dim_fre-1):
            m = torch.ceil(torch.tensor(m * dim_incre)).clone().detach()
            m = torch.min(m,torch.tensor(dim_max)).clone().detach()

        # restart
        if (k%m == 1) or m==1:
            h = h+1
            v_bar_h = v
            w_h = P.g_hessian_y(x,y,v_bar_h)
            q_h = (b - w_h)/torch.norm(b-w_h)
            Q = q_h.clone()
            tridia_T = torch.empty(0, 0)
            q = q_h
            q_last = 0
            beta = 0
        r = b-w_h
        u = P.g_hessian_y(x,y,q)-beta*q_last
        alpha = q.T@u

        # Construct tridiagonal projection matrix
        new_matrix = torch.zeros((tridia_T.shape[0]+1,tridia_T.shape[1]+1))
        new_matrix[:-1,:-1] = tridia_T
        new_matrix[-1,-1] = alpha
        if new_matrix.shape[0]>1:
            new_matrix[-1,-2]=beta
            new_matrix[-2,-1]=beta
        tridia_T = new_matrix


        s = torch.linalg.solve(tridia_T, Q.T@r)
        delta_v = 0
        for i in range(s.size(0)):
            delta_v += Q[:,i]*s[i,0]
        delta_v = delta_v.reshape(-1,1)

        omega = u - alpha*q
        beta = torch.norm(omega,2)
        q_last = q
        q = omega/beta
        Q = torch.cat((Q, q), dim=1)

        v = v_bar_h + delta_v
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