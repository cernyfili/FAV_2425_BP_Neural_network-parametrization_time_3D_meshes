import numpy as np
import torch


def mse():

    def mse(x_true, x_pred, encoder):
        return torch.sum(torch.square(x_true - x_pred), dim=1).mean()

    return mse


# modified to work for my project
def mse_area(area_coefficient):

    def mse_area(x_true, x_pred, encoder):
        diff_pos = torch.sum(torch.square(x_true - x_pred), dim=1).mean()

        x_var = torch.autograd.Variable(x_pred, requires_grad=True)
        uv_pred = encoder(x_var)
        u_pred = uv_pred[::, 0]
        v_pred = uv_pred[::, 1]
        g_u = torch.autograd.grad(outputs=u_pred, inputs=x_var, grad_outputs=torch.ones_like(u_pred),
                                  retain_graph=True, allow_unused=True, create_graph=True)[0]
        g_v = torch.autograd.grad(outputs=v_pred, inputs=x_var, grad_outputs=torch.ones_like(v_pred),
                                  retain_graph=True, allow_unused=True, create_graph=True)[0]

        e = torch.sum(g_u * g_u, dim=1)
        f = torch.sum(g_u * g_v, dim=1)
        g = torch.sum(g_v * g_v, dim=1)
        det_i = e * g - f * f
        diff_area = -det_i.mean()  # Use negative mean to maximize det_i

        return diff_pos + area_coefficient * diff_area

    return mse_area


def mse_dirichlet(dirichlet_coefficient):

    def mse_dirichlet(x_true, x_pred, encoder):
        diff_pos = torch.sum(torch.square(x_true - x_pred), dim=1).mean()
        x_var = torch.autograd.Variable(x_pred, requires_grad=True)
        uv_pred = encoder(x_var)
        u_pred = uv_pred[::, 0]
        v_pred = uv_pred[::, 1]
        g_u = torch.autograd.grad(outputs=u_pred, inputs=x_var, grad_outputs=torch.ones_like(u_pred),
                                  retain_graph=True, allow_unused=True, create_graph=True)
        g_v = torch.autograd.grad(outputs=v_pred, inputs=x_var, grad_outputs=torch.ones_like(v_pred),
                                  retain_graph=True, allow_unused=True, create_graph=True)
        gss_u = torch.sum(torch.square(g_u[0]), dim=1)
        gss_v = torch.sum(torch.square(g_v[0]), dim=1)
        # diff_grad = dirichlet_coefficient / (gss_u + gss_v).mean()
        diff_grad = -dirichlet_coefficient * (gss_u + gss_v).mean()
        return diff_pos + diff_grad

    return mse_dirichlet


def mse_variance(variance_coefficient):

    def mse_variance(x_true, x_pred, encoder):
        diff_pos = torch.sum(torch.square(x_true - x_pred), dim=1).mean()
        x_var = torch.autograd.Variable(x_pred, requires_grad=True)
        uv_pred = encoder(x_var)
        u_pred = uv_pred[::, 0]
        v_pred = uv_pred[::, 1]
        u_mean = u_pred.mean()
        v_mean = v_pred.mean()
        u_centered = u_pred - u_mean
        v_centered = v_pred - v_mean
        variance = (torch.square(u_centered) + torch.square(v_centered)).mean()
        diff_var = 1.0 / variance
        return diff_pos + diff_var


def weighted_binary_crossentropy(zero_weight, one_weight):

    bce = torch.nn.BCELoss()

    def weighted_binary_crossentropy(y_pred, y_true):

        b_ce = bce(y_pred, y_true)

        # weighted calc
        weight_vector = y_true * one_weight + (1 - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        return weighted_b_ce.mean()

    return weighted_binary_crossentropy

