"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # for p in self.params:
        #     if p not in self.u:
        #         self.u[p] = ((1 - self.momentum) * (p.grad + self.weight_decay * p.data)).detach()
        #     else:
        #         self.u[p] = (self.momentum * self.u[p] + (1 - self.momentum) * (p.grad + self.weight_decay * p.data)).detach()
        #     p.data -= self.lr * self.u[p] 
        for p in self.params:
            if p.grad is None: 
                continue
            self.u[p] = (self.momentum * self.u.get(p, 0) + (1 - self.momentum) * (p.grad + self.weight_decay * p.data)).detach()
            p.data -= self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1 
        for p in self.params:
            if p.grad is None:
                continue
            grad = (p.grad + self.weight_decay * p.data).detach()
            self.m[p] = (self.beta1 * self.m.get(p, 0) + (1 - self.beta1) * (grad)).detach()
            self.v[p] = (self.beta2 * self.v.get(p, 0) + (1 - self.beta2) * (grad ** 2)).detach()
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (v_hat ** 0.5 + self.eps))
        
        ### END YOUR SOLUTION
        # STEP-BY-STEP
        # # Increment the timestep
        # self.t += 1
        
        # # Iterate over all parameters
        # for param in self.params:
        #     # Initialize moving averages for the parameter if not already done
        #     if param not in self.m:
        #         self.m[param] = ndl.init.zeros_like(param).detach()
        #         self.v[param] = ndl.init.zeros_like(param).detach()
            
        #     # Get the gradient and detach it to avoid tracking in the computation graph
        #     grad = param.grad.detach()
            
        #     # Apply weight decay (L2 penalty) if specified
        #     if self.weight_decay > 0:
        #         grad = grad + self.weight_decay * param.detach()
            
        #     # Retrieve current moving averages
        #     u = self.m[param]
        #     v = self.v[param]
            
        #     # Update moving averages
        #     u = self.beta1 * u + (1 - self.beta1) * grad
        #     v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
        #     # Store updated moving averages
        #     self.m[param] = u
        #     self.v[param] = v
            
        #     # Apply bias correction
        #     u_hat = u / (1 - self.beta1 ** self.t)
        #     v_hat = v / (1 - self.beta2 ** self.t)
            
        #     # Compute the update
        #     update = u_hat / (v_hat ** 0.5 + self.eps)
            
        #     # Update the parameter data
        #     param.data = param.data - self.lr * update
        
    