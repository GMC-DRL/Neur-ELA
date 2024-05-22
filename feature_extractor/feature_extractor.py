import torch
from feature_extractor.attention_block import MultiHeadEncoder, EmbeddingNet, PositionalEncoding
from torch import nn
import numpy as  np


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# neural feature extractor
class Feature_Extractor(nn.Module):
    def __init__(self,
                 node_dim=2,    # todo: node_dim should be win * 2
                 hidden_dim=16,
                 n_heads=1,
                 ffh=16,
                 n_layers=1,
                 use_positional_encoding = True,
                 is_mlp = False,
                 ):
        super(Feature_Extractor, self).__init__()
        # bs * dim * pop_size * 2  ->  bs * dim * pop_size * hidden_dim
        self.embedder = EmbeddingNet(node_dim=node_dim,embedding_dim=hidden_dim)
        # positional_encoding, we only add PE at before the dimension encoder
        # since each dimension should be regarded as different parts, their orders matter.
        self.is_mlp = is_mlp
        if not self.is_mlp:
            self.use_PE = use_positional_encoding
            if self.use_PE:
                self.position_encoder = PositionalEncoding(hidden_dim,512)

            # bs * dim * pop_size * hidden_dim -> bs * dim * pop_size * hidden_dim
            self.dimension_encoder = mySequential(*(MultiHeadEncoder(n_heads=n_heads,
                                                    embed_dim=hidden_dim,
                                                    feed_forward_hidden=ffh,
                                                    normalization='n2')
                                                    for _ in range(n_layers)))
            # the gradients are predicted by attn each dimension of a single individual.
            # bs * pop_size * dim * 128 -> bs * pop_size * dim * 128
            self.individual_encoder = mySequential(*(MultiHeadEncoder(n_heads=n_heads,
                                                    embed_dim=hidden_dim,
                                                    feed_forward_hidden=ffh,
                                                    normalization='n1')
                                                    for _ in range(n_layers)))
        else:
            self.mlp = nn.Linear(hidden_dim, hidden_dim)
            self.acti = nn.ReLU()
        # print('------------------------------------------------------------------')
        # print('The feature extractor has been successfully initialized...')
        # print(self.get_parameter_number())
        # print('------------------------------------------------------------------')
        self.is_train = False

    def set_on_train(self):
        self.is_train = True
    
    def set_off_train(self):
        self.is_train = False

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # return 'Total {} parameters, of which {} are trainable.'.format(total_num, trainable_num)
        return total_num

    # todo: add a dimension representing the window
    # xs(bs * pop_size * dim) are the candidates,
    # ys(bs * pop_size) are the corresponding obj values, they are np.array
    def forward(self, xs, ys):
        if self.is_train:
            return self._run(xs, ys)
        else:
            with torch.no_grad():
                return self._run(xs, ys)
    
    def _run(self, xs, ys):
        _,d = xs.shape
        xs = xs[None,:,:] # 1 * n * d
        ys = ys[None,:]
        y_ = (ys - ys.min(-1)[:, None]) / (ys.max(-1)[:, None] - ys.min(-1)[:, None] + 1e-12)
        ys = y_[:, :, None]

        # pre-processing data as the form of per_dimension_feature bs * d * n * 2
        a_x = xs[:, :, :, None]
        a_y = np.repeat(ys, d, -1)[:, :, :, None]
        raw_feature = np.concatenate([a_x, a_y], axis=-1).transpose((0, 2, 1, 3)) # bs * d * n * 2
        h_ij = self.embedder(torch.tensor(raw_feature,dtype=torch.float32)) # bs * dim * pop_size * 2  ->  bs * dim * pop_size * hd
        bs, dim, pop_size, node_dim = h_ij.shape
        # resize h_ij as (bs*dim) * pop_size * hd
        h_ij = h_ij.view(-1,pop_size,node_dim)
        if not self.is_mlp:
            o_ij = self.dimension_encoder(h_ij).view(bs,dim,pop_size,node_dim) # bs * dim * pop_size * 128 -> bs * dim * pop_size * hd
            # resize o_ij, to make each dimension of the single individual into as a group
            o_i = o_ij.permute(0,2,1,3).contiguous().view(-1,dim,node_dim)
            if self.use_PE:
                o_i = o_i + self.position_encoder.get_PE(dim) * 0.5
            out = self.individual_encoder(o_i).view(bs,pop_size,dim,node_dim) # (bs * pop_size) * dim * 128 -> bs * pop_size * dim * hidden_dim
            # bs * pop_size * hidden_dim
            out = torch.mean(out,-2)
            return out
        else:
            out = self.mlp(self.acti(h_ij)).view(bs,dim,pop_size,node_dim)
            out = torch.mean(out, -3)
            return out

if __name__ == '__main__':

    h_choices = [16, 64, 128]
    n_choices = [1, 3, 5]

    for h in h_choices:
        for n in n_choices:
            fe = Feature_Extractor(hidden_dim=h, n_layers=n, is_mlp=False)
            num_para = fe.get_parameter_number()
            print(f'h: {h}, n: {n}, num_para: {num_para}')

    for h in h_choices:
        fe = Feature_Extractor(hidden_dim=h, is_mlp=True)
        num_para = fe.get_parameter_number()
        print(f'mlp, h: {h}, num_para: {num_para}')
    

