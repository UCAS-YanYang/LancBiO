import torch

class Test_Problem:
    def __init__(self, args):
        self.n = args.n                                                                     # problem dimension

        # upper-level function f config
        self.coef = (torch.rand((self.n,1)) - 0.5) * 10                                     # scale of upper-level cos function
        self.condix = torch.rand((self.n,1)) + 0.1                                          # control upper-level condition 

        # lower-level function g config
        self.gamma = args.gamma                                                             # scale of lower-level sin function
        self.condition = args.condition                                                     # control lower-level condition 
        self.condiy = torch.rand((self.n,1))*self.gamma                                     # scale of lower-level regularization
        self.G = self.generate_spd_matrix_with_condition()                                  # lower-level symmetry matrix
    
        # parameter for testing
        self.eps = args.eps
        self.theta = args.theta

    def generate_spd_matrix_with_condition(self):
        # Step 1: Generate a diagonal matrix D with values from 1 to condition
        # D = torch.diag( torch.rand((self.n))*(self.condition-1)+1 )
        D = torch.diag(torch.linspace(1, self.condition, self.n))

        # Step 2: Generate a random orthogonal matrix Q
        Q, _ = torch.linalg.qr(torch.randn(self.n, self.n))
        
        # Step 3: Construct the SPD matrix G by QDQ^T
        G = Q @ D @ Q.T
        
        return G
    
    # f(x,y) = 0.1*torch.cos(x.T@(self.coef*y) + 0.5*(condi*x-y).T(condi*x-y)
    def f_value(self,x,y):
        return (0.1*torch.cos(x.T@(self.coef*y)) + 0.5*(self.condix*x-y).T@(self.condix*x-y)).item()

    def f_grad_x(self,x,y):
        return  -0.1* torch.sin(x.T@(self.coef*y))*self.coef *y + self.condix*(self.condix*x-y)

    def f_grad_y(self,x,y):
        return - 0.1*torch.sin(x.T@(self.coef*y))*self.coef *x -(self.condix*x-y)
    
    
    # g(x,y) = gamma*(\sum{sin(x_i+y_i)}) + log(\sum{e^{x_iy^i}}) + 0.5*y^T(condiy*y) + 0.5*y^T@G@y
    def g_grad_y(self,x,y):
        exp_xy = torch.exp(x * y)
        sum_exp_xy = torch.sum(exp_xy)
        grad_y = self.gamma * torch.cos(x + y) + (x * exp_xy) / sum_exp_xy + self.condiy*y + self.G@y
        return grad_y

    def g_hessian_y(self,x,y,vec):
        '''
            hessian-vector product
            only self.G@vec  costs O(n^2), other operations cost O(n)
        '''
        first_part = -self.gamma * torch.sin(x + y) * vec

        exp_xy = torch.exp(x * y)
        sum_exp_xy = torch.sum(exp_xy)
        temp = x * x * exp_xy
        g_y = (x * exp_xy) / sum_exp_xy
        second_part = (1/sum_exp_xy) * temp * vec - (g_y.T@vec) * g_y

        return first_part + second_part + self.condiy*vec + self.G@vec 

    def g_jac_xy(self,x,y,vec):
        '''
            Jocabian-vector product
        '''
        first_part = -self.gamma * torch.sin(x + y) * vec

        exp_xy = torch.exp(x * y)
        sum_exp_xy = torch.sum(exp_xy)
        temp = (1 + x*y) * exp_xy
        g_y = (x * exp_xy) / sum_exp_xy
        g_x = (y * exp_xy) / sum_exp_xy
        second_part = (1/sum_exp_xy) * temp * vec - (g_y.T@vec) * g_x

        return first_part + second_part 
    
    def g_jac_yx(self,x,y,vec):
        '''
            Jocabian-vector product, used in STABLE
        '''
        first_part = -self.gamma * torch.sin(x + y) * vec

        exp_xy = torch.exp(x * y)
        sum_exp_xy = torch.sum(exp_xy)
        temp = (1 + x*y) * exp_xy
        g_y = (x * exp_xy) / sum_exp_xy
        g_x = (y * exp_xy) / sum_exp_xy
        second_part = (1/sum_exp_xy) * temp * vec - (g_x.T@vec) * g_y

        return first_part + second_part 

    # test functions
    def residual(self,x,y,v):
        '''
            evaluate residual norm
        '''
        return self.f_grad_y(x,y) - self.g_hessian_y(x,y,v)

    def hyper_grad(self,x,y,v):
        '''
            estimate hyper-gradient
        '''
        for i in range(500):
            d = self.g_grad_y(x,y)
            y = y - self.theta * d 
            if torch.norm(d,2)<self.eps:
                break

        b = self.f_grad_y(x,y)
        A_v = self.g_hessian_y(x,y,v)
        r = A_v - b
        p = -r

        for i in range(50):
            A_p = self.g_hessian_y(x,y,p)   
            alpha = (r.T@r)/(p.T@A_p)

            v = (v + alpha*p)
            r_new = r + alpha*A_p
            beta = (r_new.T@r_new)/(r.T@r)

            p = -r_new + beta*p
            r = r_new
            if torch.norm(r,2)<self.eps:
                break
        
        return self.f_grad_x(x,y) - self.g_jac_xy(x,y,v)