import torch 
import torch.nn as nn
### Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self,  **kwargs):
        super(BAP, self).__init__()
    def forward(self,feature_maps,attention_maps):
        feature_shape = feature_maps.size() ## 12*768*26*26*
        attention_shape = attention_maps.size() ## 12*num_parts*26*26
        # print(feature_shape,attention_shape)
        phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps)) ## 12*32*768
        phi_I = torch.div(phi_I, float(attention_shape[2] * attention_shape[3]))
        phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 1e-12))
        phi_I = phi_I.view(feature_shape[0],-1)
        raw_features = torch.nn.functional.normalize(phi_I, dim=-1) ##12*(32*768)
        pooling_features = raw_features*100
        # print(pooling_features.shape)
        return raw_features,pooling_features
class ResizeCat(nn.Module):
    def __init__(self,  **kwargs):
        super(ResizeCat, self).__init__()
    
    def forward(self,at1,at3,at5):
        N,C,H,W = at1.size()
        resized_at3 = nn.functional.interpolate(at3,(H,W))
        resized_at5 = nn.functional.interpolate(at5,(H,W))
        cat_at = torch.cat((at1,resized_at3,resized_at5),dim=1)
        return cat_at

if __name__ == '__main__':
    # a = BAP()
    a = ResizeCat()
    a1 = torch.Tensor(4,3,14,14)
    a3 = torch.Tensor(4,5,12,12)
    a5 = torch.Tensor(4,9,9,9)
    ret = a(a1,a3,a5)
    print(ret.size())