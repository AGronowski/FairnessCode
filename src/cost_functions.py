import torch

# y = torch.FloatTensor([1,0])
# yhat = torch.FloatTensor([0.25,0.63])

def cross_entropy(yhat,y):

    cross_entropy = -y*yhat.log() - (1-y)*(1-yhat).log()
    reduced = cross_entropy.sum()
    return reduced

def renyi_cross_entropy(yhat,y,alpha):

    insidelog= y*yhat.pow(alpha-1) +(1-y)*(1-yhat).pow(alpha-1)
    renyi_cross_entropy = insidelog.log()/(1-alpha)
    reduced = renyi_cross_entropy.sum()
    return reduced

def renyi_divergence(mu,logvar,alpha):

    term1 = - 0.5 * logvar
    var = logvar.exp()

    inside_log = alpha + (1-alpha)*var
    term2 = - 0.5 * torch.log(inside_log) /(alpha -1)

    term3_num = 0.5 * alpha * mu.pow(2)
    term3 = term3_num / inside_log

    total = term1 + term2 + term3
    return torch.sum(total)



def get_IB_or_Skoglund_loss(yhat, y, mu, logvar, beta):
    # I(Z;X)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # sums in both dimensions. columns are equal to latent_dim, rows to batch size
    divergence = KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    # divergence = renyi_divergence(mu,logvar,0.2)

    # hopefully this prevents the RuntimeError: reduce failed to synchronize: device-side assert triggered error

    # output.view(-1)[output.view(-1) > 1.0] = 1.0
    # output.view(-1)[output.view(-1) < 0] = 0

    # I(Z;Y)
    # reduction has to be sum. Cross entropies for each example are summed together

    yhat = yhat.view(-1)
    #torch.where: (condition, condition true, condition false)
    yhat = torch.where(torch.isnan(yhat), torch.zeros_like(yhat), yhat)
    yhat = torch.where(torch.isinf(yhat), torch.zeros_like(yhat), yhat)

    cross_entropy = torch.nn.functional.binary_cross_entropy(yhat, y,
                                                   reduction='sum')  # l(x,y)

    # rce = renyi_cross_entropy(output.view(-1), y, alpha=6)

    loss = divergence + beta * cross_entropy
    return loss

def get_combined_loss(yhat, yhat_fair, y, mu, logvar, beta):
    # I(Z;X)
    divergence = KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    IB_cross_entropy = torch.nn.functional.binary_cross_entropy(yhat.view(-1), y,
                                                   reduction='sum')
    Skoglund_cross_entropy = torch.nn.functional.binary_cross_entropy(yhat_fair.view(-1), y,
                                                   reduction='sum')

    loss = divergence + IB_cross_entropy + beta * Skoglund_cross_entropy

    return loss