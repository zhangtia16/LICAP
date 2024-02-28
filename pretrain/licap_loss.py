import torch
import torch.nn as nn
import torch.nn.functional as F


class Top_Bin_Loss(nn.Module):
    def __init__(self, temp):
        super(Top_Bin_Loss, self).__init__()
        self.temp = temp

    def forward(self, embed_important, embed_normal):
        
        
        embed_important_anchor = torch.mean(embed_important, dim=0)

        # normalize
        embed_important = F.normalize(embed_important)  
        embed_normal = F.normalize(embed_normal)
        embed_important_anchor = F.normalize(embed_important_anchor,dim=0)

        anchors_bot = torch.cat((embed_important_anchor.unsqueeze(0), embed_normal), dim=0) # [1+normal_num, dim]

        embed_important_anchor = embed_important_anchor.expand(embed_important.shape[0],embed_important_anchor.shape[0])


        pos_score = torch.mul(embed_important, embed_important_anchor).sum(dim=1)  # [imp_num, 1]
        pos_score = torch.exp(pos_score/ self.temp)
        

        ttl_score = torch.matmul(embed_important, anchors_bot.transpose(0, 1)) # [imp_num, dim]*[dim, 1+normal_num]=[imp_num, 1+normal_num]
        ttl_score = torch.exp(ttl_score / self.temp).sum(dim=1) # [imp_num, 1]

        loss = -torch.log(pos_score / ttl_score).mean()
        return loss

class Finer_Bin_Loss(nn.Module):
    def __init__(self, temp):
        super(Finer_Bin_Loss, self).__init__()
        self.temp = temp

    def forward(self, imp_node_embed, imp_bin_centroids, imp_node_centroid, imp_node_coeff):
        
        # normalize
        imp_node_embed = F.normalize(imp_node_embed)  # [imp_num, dim]
        imp_node_centroid = F.normalize(imp_node_centroid) # [imp_num, dim]
        imp_bin_centroids = F.normalize(imp_bin_centroids) # [bin_num, dim]
        
        
       
        pos_logits = torch.mul(imp_node_embed, imp_node_centroid).sum(dim=1)  # [imp_num, dim] -> [imp_num, 1]
        pos_logits = torch.exp(pos_logits/ self.temp) # [imp_num, 1]
        

        ttl_logits = torch.matmul(imp_node_embed, imp_bin_centroids.transpose(0, 1)) # [imp_num, dim]*[dim, bin_num]=[imp_num, bin_num]

        ttl_logits = torch.mul(ttl_logits, imp_node_coeff) # [imp_num, bin_num]
        ttl_logits = torch.exp(ttl_logits / self.temp).sum(dim=1) # [imp_num, bin_num] -> [imp_num, 1]
        

        loss = -torch.log(pos_logits / ttl_logits).mean() # [imp_num, 1] -> [1]
        return loss


