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


# class ScaleGNN(nn.Module):
#     def __init__(self, g, loss_function):
#         super(ScaleGNN, self).__init__()
#         self.scaling_model = nn.Linear(64,1)
#         self.loss_fn = loss_function

#     def forward(self, embed, logits, labels=None, idx=None):
#         t = self.scaling_model(embed)
#         t = F.relu(t)
#         t = torch.log(torch.exp(t) + torch.tensor(1.1))
#         output = logits * t

#         loss = self.loss_fn(output[idx], labels[idx].unsqueeze(-1))
#         return output, loss



# class ClusterLoss_0(nn.Module):
#     def __init__(self):
#         super(ClusterLoss_0, self).__init__()

#     def forward(self, embed_important, embed_normal):
#         embed_important_anchor = torch.mean(embed_important, dim=0)
#         embed_normal_anchor = torch.mean(embed_normal, dim=0)   
#         loss = torch.abs(F.cosine_similarity(embed_important_anchor,embed_normal_anchor, dim=0))
#         return loss



# class ClusterLoss_1(nn.Module):
#     def __init__(self, dis_norm):
#         super(ClusterLoss, self).__init__()
#         self.dis_norm = dis_norm

#     def forward(self, embed_important, embed_normal):
#         embed_important_anchor = torch.mean(embed_important, dim=0)

#         #embed_normal_anchor = torch.mean(embed_normal, dim=0)

#         if self.dis_norm == 'L2':
#             loss = torch.mean(torch.pow((embed_important - embed_important_anchor), 2)) #+ torch.mean(torch.pow((embed_normal - embed_normal_anchor), 2))
#         elif self.dis_norm == 'L1':
#             loss = torch.mean(torch.abs(embed_important - embed_important_anchor))
#         else:
#             print('dis_norm wrong type')
#         return loss



# class ClusterLoss_3(nn.Module):
#     def __init__(self, dis_norm, temp):
#         super(ClusterLoss, self).__init__()
#         self.dis_norm = dis_norm
#         self.temp = temp

#     def forward(self, embed_important, bin_centroids):
        
#         # embed_important_anchor = torch.mean(embed_important, dim=0)

#         # normalize
#         embed_important = F.normalize(embed_important)  # [imp_num, dim]
#         bin_centroids = F.normalize(bin_centroids) # [bin_num, dim]

#         important_centroid = bin_centroids[-1] # [dim]
#         important_centroid = important_centroid.expand(embed_important.shape[0],important_centroid.shape[0]) # [imp_num, dim]
       
#         pos_logits = torch.mul(embed_important, important_centroid).sum(dim=1)  # [imp_num, dim] -> [imp_num, 1]
        
#         pos_logits = torch.exp(pos_logits/ self.temp) # [imp_num, 1]
        

#         ttl_logits = torch.matmul(embed_important, bin_centroids.transpose(0, 1)) # [imp_num, dim]*[dim, bin_num]=[imp_num, bin_num]
        
#         ttl_logits = torch.exp(ttl_logits / self.temp).sum(dim=1) # [imp_num, dim] -> [imp_num, 1]
        

#         loss = -torch.log(pos_logits / ttl_logits).mean() # [imp_num, 1] -> [1]
        
#         # print(pos_score)
#         # print(ttl_score)
#         # print('-----')
#         return loss

# class Finer_Bin_Loss_0(nn.Module):
#     def __init__(self, temp):
#         super(Finer_Bin_Loss, self).__init__()
#         self.temp = temp

#     def forward(self, train_node_embed, centroids, train_node_centroid, alpha_coeff):
        
#         # embed_important_anchor = torch.mean(embed_important, dim=0)

#         # normalize
#         train_node_embed = F.normalize(train_node_embed)  # [train_num, dim]
#         train_node_centroid = F.normalize(train_node_centroid) # [train_num, dim]
#         centroids = F.normalize(centroids) # [bin_num, dim]
        
       
#         pos_logits = torch.mul(train_node_embed, train_node_centroid).sum(dim=1)  # [train_num, dim] -> [train_num, 1]
#         pos_logits = torch.exp(pos_logits/ self.temp) # [train_num, 1]
        

#         ttl_logits = torch.matmul(train_node_embed, centroids.transpose(0, 1)) # [train_num, dim]*[dim, bin_num]=[train_num, bin_num]

#         ttl_logits = torch.mul(ttl_logits, alpha_coeff) # [train_num, bin_num]
#         ttl_logits = torch.exp(ttl_logits / self.temp).sum(dim=1) # [train_num, bin_num] -> [train_num, 1]
        

#         loss = -torch.log(pos_logits / ttl_logits).mean() # [train_num, 1] -> [1]
        
#         # print(pos_logits)
#         # print(ttl_logits)
#         # print('-----')
#         return loss