import torch
from torch import nn as nn
from torch.nn import functional as F
import random

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage # if LQ_stage is True, it means the indices of input has a groundtruth to learn from.
        self.beta = beta 
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    
    def dist(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + \
                    torch.sum(y**2, dim=1) - 2 * \
                    torch.matmul(x, y.t())
    
    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h*w, c)
        y = y.reshape(b, h*w, c)

        gmx = x.transpose(1, 2) @ x / (h*w)
        gmy = y.transpose(1, 2) @ y / (h*w)
    
        return (gmx - gmy).square().mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        d = self.dist(z_flattened, codebook)
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)

            gt_min_indices = gt_indices.reshape_as(min_encoding_indices)
            gt_min_onehot = torch.zeros(gt_min_indices.shape[0], codebook.shape[0]).to(z)
            gt_min_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_min_onehot, codebook)
            z_q_gt = z_q_gt.view(z.shape)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)

        e_latent_loss = torch.mean((z_q.detach() - z)**2)  #只用了前面这一个损失
        q_latent_loss = torch.mean((z_q - z.detach())**2)

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * ((z_q_gt.detach() - z) ** 2).mean() 
            texture_loss = self.gram_loss(z, z_q_gt.detach()) 
            codebook_loss = codebook_loss + texture_loss 
        else:
            codebook_loss = q_latent_loss + e_latent_loss * self.beta

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    
    # def get_codebook_entry(self, indices):
    #     b, _, h, w = indices.shape

    #     indices = indices.flatten().to(self.embedding.weight.device)
    #     min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
    #     min_encodings.scatter_(1, indices[:,None], 1)

    #     # get quantized latent vectors
    #     z_q = torch.matmul(min_encodings.float(), self.embedding.weight)        
    #     z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
    #     return z_q
    


class CombineQuantBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out
    



class Q_tools(nn.Module):
    def __init__(self):
        super().__init__()  

        self.Num_codebook = 3
        self.quant_seperation_layer = nn.Conv2d(180, 180, 1)

        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(self.Num_codebook):
            quantize = VectorQuantizer(
                256,
                180,
                LQ_stage=False,
            )
            self.quantize_group.append(quantize)
            self.before_quant_group.append(nn.Conv2d(180, 180, 1))
            self.after_quant_group.append(CombineQuantBlock(180, 0, 180))


        self.Main_in_conv = nn.Conv2d(180, 180, 1)
        self.Main_out_conv = CombineQuantBlock(180, 0, 180)

        self.Main_quantize = VectorQuantizer(
            256,
            180,
            LQ_stage=False,
        )


    def Shuffer_index(self,input_index, num_range, Ratio):   #用来对输入序列进行随机排序

        Shuff_index = torch.ones_like(input_index) * 1000
        
        for i in range(len(input_index)):
            
            Random_ratio = random.random()
            if Random_ratio <= Ratio:
                i_r = random.randint(0, num_range-1)
                Shuff_index[i] = i_r
            else:
                i_v = input_index[i]
                Shuff_index[i] = i_v

        return Shuff_index



    def forward_train(self, input, x_size, Class_gt = None):


        quant_high_list = [] 
        codebook_loss_list = []
        sort_index = []

        loss_count = 0

        Random_ratio = 0.0
        if self.training:
            Return_index = self.Shuffer_index(Class_gt, self.Num_codebook, Random_ratio)
        else:
            Return_index = Class_gt

      
        b,hw,c = input.shape
        input = input.transpose(2,1).contiguous()
        input = input.reshape(b,c,x_size[0],x_size[1]).contiguous()

        low_feature = self.quant_seperation_layer(input)
        high_feature = input - low_feature



        for i_class in range(self.Num_codebook):

            index_feature = torch.where(Return_index == i_class)


            if len(index_feature[0]) == 0:
                continue
            
            loss_count += 1
            Split_high_feature = high_feature[index_feature]

            Split_high_feature = self.before_quant_group[i_class](Split_high_feature)   
            z_quant_high, codebook_loss, indices = self.quantize_group[i_class](Split_high_feature)
            z_quant_high = self.after_quant_group[i_class](z_quant_high, None)  


            quant_high_list.append(z_quant_high)
            codebook_loss_list.append(codebook_loss)
            sort_index.append(index_feature[0])


        high_quant_feature = torch.cat(quant_high_list, dim=0)
        sort_index = torch.cat(sort_index)
        sort_index = torch.argsort(sort_index)
        high_quant_feature = torch.index_select(input=high_quant_feature, dim=0, index=sort_index)


        low_feature = self.Main_in_conv(low_feature)
        z_quant_low, codebook_loss_low, indices_low = self.Main_quantize(low_feature)
        low_quant_feature = self.Main_out_conv(z_quant_low, None)

        codebook_loss_list.append(codebook_loss_low)
        loss_count += 1

        out = low_quant_feature + high_quant_feature


        out = out.reshape(b,c,-1).contiguous()
        out = out.transpose(2,1).contiguous()

        loss_all = sum(codebook_loss_list)/loss_count

        return out, loss_all
    


    def forward_test(self, input, x_size):


        b,hw,c = input.shape
        input = input.transpose(2,1).contiguous()
        input = input.reshape(b,c,x_size[0],x_size[1]).contiguous()

        low_feature = self.quant_seperation_layer(input)
        high_feature = input - low_feature


        low_feature = self.Main_in_conv(low_feature)
        z_quant_low, codebook_loss_low, indices_low = self.Main_quantize(low_feature)
        low_quant_feature = self.Main_out_conv(z_quant_low, None)
        
        out = low_quant_feature + high_feature



        out = out.reshape(b,c,-1).contiguous()
        out = out.transpose(2,1).contiguous()



        return out 