import torch
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union, Dict, Callable, Type
# torch.set_printoptions(threshold=torch.inf)
def conweight(net, key2, out_c, in_c, head, dim, k, offset):
    res = torch.tensor(torch.zeros(in_c, in_c, k, k), device='cuda')
    for i in range(head):
        res[dim*i:dim*(i+1)] = net['state_dict'][key2][offset+48*i:offset+48*i+dim].clone()
        print(f"res[{dim*i}:{dim*(i+1)}]=net['state_dict']['{key2}'][{offset+48*i}:{offset+48*i+dim}]")
    return res

def conweight1(net, key2, out_c, in_c, head, dim, k, offset):
    res = torch.tensor(torch.zeros(in_c, 1, k, k), device='cuda')
    for i in range(head):
        res[dim*i:dim*(i+1)] = net['state_dict'][key2][offset+48*i:offset+48*i+dim].clone()
        print(f"res[{dim*i}:{dim*(i+1)}]=net['state_dict']['{key2}'][{offset+48*i}:{offset+48*i+dim}]")
    return res

def conweight2(net, key2, out_c, in_c, head, dim, k, offset):
    res = torch.tensor(torch.zeros(in_c, 16, k, k), device='cuda')
    for i in range(head):
        res[dim*i:dim*(i+1)] = net['state_dict'][key2][offset+48*i:offset+48*i+dim].clone()
        print(f"res[{dim*i}:{dim*(i+1)}]=net['state_dict']['{key2}'][{offset+48*i}:{offset+48*i+dim}]")
    return res

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def has_close_enough_values(x, y, threshold=0.000001, percentage=0.999999):
    rel_err = torch.abs(x - y) / torch.abs(y)
    close_count = torch.sum(rel_err < threshold).item()
    total_count = x.numel()
    return close_count / total_count >= percentage
        
net = torch.load('../checkpoints/effvit/quan_b1_full.pth' ,map_location=torch.device('cuda'))
for key, value in list(net['state_dict'].items()):
    if 'qkv' in key or 'aggre' in key: print(key, value.size(), sep=" ")


def all_in(net, a, b, out_c, in_c, head, dim, k):
    res_q_1 = conweight(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight',out_c,in_c,head,16,1,0)
    res_k_1 = conweight(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight',out_c,in_c,head,16,1,16)
    res_v_1 = conweight(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight',out_c,in_c,head,16,1,32)
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.q.conv.weight'] = res_q_1
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.k.conv.weight'] = res_k_1
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.v.conv.weight'] = res_v_1
    res1 = torch.cat([res_q_1,res_k_1,res_v_1])
    w1 = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight']
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight'] = res1
    w1_t = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight']
    
    res_q_2 = conweight1(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight',out_c,in_c,head,16,5,0)
    res_k_2 = conweight1(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight',out_c,in_c,head,16,5,16)
    res_v_2 = conweight1(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight',out_c,in_c,head,16,5,32)
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg_q.0.0.weight'] = res_q_2
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg_k.0.0.weight'] = res_k_2
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg_v.0.0.weight'] = res_v_2
    res2 = torch.cat([res_q_2,res_k_2,res_v_2])
    w2 = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight']
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight'] = res2
    w2_t = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight']

    res_q_3 = conweight2(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight',out_c,in_c,head,16,1,0)
    res_k_3 = conweight2(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight',out_c,in_c,head,16,1,16)
    res_v_3 = conweight2(net, f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight',out_c,in_c,head,16,1,32)
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg_q.0.1.weight'] = res_q_3
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg_k.0.1.weight'] = res_k_3
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg_v.0.1.weight'] = res_v_3
    res3 = torch.cat([res_q_3,res_k_3,res_v_3])
    # print(res3.shape)
    w3 = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight']
    net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight'] = res3
    w3_t = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight']
    H = 60
    W = 120
    test = torch.randn(1,in_c,H,W, device='cuda') # BCHW
    real_1 = F.conv2d(test, w1, bias=None, groups=1, padding=get_same_padding(1))
    real_2 = F.conv2d(real_1, w2, bias=None, groups=in_c*3, padding=get_same_padding(5))
    real_3 = F.conv2d(real_2, w3, bias=None, groups=head*3, padding=get_same_padding(1))
    real = torch.cat([real_1, real_3],dim=1)
    real = torch.reshape(
        real,
        (
            1,
            -1,
            3 * dim,
            H * W,
        ),
    )
    real = torch.transpose(real, -1, -2)
    q, k, v = (
            real[..., 0 : dim],
            real[..., dim : 2 * dim],
            real[..., 2 * dim :],
        )
    m_q_1 = F.conv2d(test,  res_q_1, bias=None, groups=1, padding=get_same_padding(1))
    m_q_2 = F.conv2d(m_q_1, res_q_2, bias=None, groups=in_c, padding=get_same_padding(5))
    m_q_3 = F.conv2d(m_q_2, res_q_3, bias=None, groups=head, padding=get_same_padding(1))
    m_q = torch.cat([m_q_1,   m_q_3], dim=1)
    
    m_k_1 = F.conv2d(test,  res_k_1, bias=None, groups=1, padding=get_same_padding(1))
    m_k_2 = F.conv2d(m_k_1, res_k_2, bias=None, groups=in_c, padding=get_same_padding(5))
    m_k_3 = F.conv2d(m_k_2, res_k_3, bias=None, groups=head, padding=get_same_padding(1))
    m_k = torch.cat([m_k_1,   m_k_3], dim=1)
    
    m_v_1 = F.conv2d(test,  res_v_1, bias=None, groups=1, padding=get_same_padding(1))
    m_v_2 = F.conv2d(m_v_1, res_v_2, bias=None, groups=in_c, padding=get_same_padding(5))
    m_v_3 = F.conv2d(m_v_2, res_v_3, bias=None, groups=head, padding=get_same_padding(1))
    m_v = torch.cat([m_v_1,   m_v_3], dim=1)
    
    m_1 = F.conv2d(test, res1, bias=None, groups=1, padding=get_same_padding(1))
    m_2 = F.conv2d(m_1, res2, bias=None, groups=in_c*3, padding=get_same_padding(5))
    m_3 = F.conv2d(m_2, res3, bias=None, groups=head*3, padding=get_same_padding(1))
    m_q = torch.cat([m_1[:,0:in_c,:,:], m_3[:,0:in_c,:,:]], dim=1)
    m_k = torch.cat([m_1[:,in_c:in_c*2,:,:], m_3[:,in_c:in_c*2,:,:]], dim=1)
    m_v = torch.cat([m_1[:,in_c*2:,:,:], m_3[:,in_c*2:,:,:]], dim=1) # 1, 128, 1, 2 -> 1, 8, 16, 2
    m_q = torch.reshape(m_q, (1,-1,dim,H*W,),)
    m_q = torch.transpose(m_q, -1, -2)
    m_k = torch.reshape(m_k, (1,-1,dim,H*W,),)
    m_k = torch.transpose(m_k, -1, -2)
    m_v = torch.reshape(m_v, (1,-1,dim,H*W,),)
    m_v = torch.transpose(m_v, -1, -2)
    # torch.set_printoptions(threshold=torch.inf)
    
    m_1_t = F.conv2d(test, w1_t, bias=None, groups=1, padding=get_same_padding(1))
    m_2_t = F.conv2d(m_1_t, w2_t, bias=None, groups=in_c*3, padding=get_same_padding(5))
    m_3_t = F.conv2d(m_2_t, w3_t, bias=None, groups=head*3, padding=get_same_padding(1))
    m_q_t = torch.cat([m_1_t[:,0:in_c,:,:], m_3_t[:,0:in_c,:,:]], dim=1)
    m_k_t = torch.cat([m_1_t[:,in_c:in_c*2,:,:], m_3_t[:,in_c:in_c*2,:,:]], dim=1)
    m_v_t = torch.cat([m_1_t[:,in_c*2:,:,:], m_3_t[:,in_c*2:,:,:]], dim=1) # 1, 128, 1, 2 -> 1, 8, 16, 2
    m_q_t = torch.reshape(m_q_t, (1,-1,dim,H*W,),)
    m_q_t = torch.transpose(m_q_t, -1, -2)
    m_k_t = torch.reshape(m_k_t, (1,-1,dim,H*W,),)
    m_k_t = torch.transpose(m_k_t, -1, -2)
    m_v_t = torch.reshape(m_v_t, (1,-1,dim,H*W,),)
    m_v_t = torch.transpose(m_v_t, -1, -2)

    print(torch.all(q==m_q).item() and torch.all(q==m_q_t).item())
    print(torch.all(k==m_k).item() and torch.all(k==m_k_t).item())
    print(torch.all(v==m_v).item() and torch.all(v==m_v_t).item())
    print(torch.all(w1==w1_t))
    return test, m_q_t, m_k_t, m_v_t

# all_in(net,2,1,192,64,4,16,5)
# all_in(net,2,2,192,64,4,16,5)
# all_in(net,3,1,384,128,8,16,5)
all_in(net,2,1,384,128,8,16,5)
all_in(net,2,2,384,128,8,16,5)
all_in(net,2,3,384,128,8,16,5)
all_in(net,3,1,768,256,16,16,5)
all_in(net,3,2,768,256,16,16,5)
all_in(net,3,3,768,256,16,16,5)
test, m_q_t_real, m_k_t_real, m_v_t_real = all_in(net,3,4,768,256,16,16,5)
torch.save(net, '../checkpoints/effvit/quan_b1_full_qkv.pth')
# #验证修改是否成功
net = torch.load('../checkpoints/effvit/quan_b1_full_qkv.pth' ,map_location=torch.device('cuda'))
for key, value in list(net['state_dict'].items()):
    print(key, value.size(), sep=" ")

in_c = 256
head = 16
H = 60
W = 120
dim = 16
a = 3
b = 4
w1_t = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.qkv.conv.weight']
w2_t = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.0.weight']
w3_t = net['state_dict'][f'backbone.stages.{a}.op_list.{b}.context_module.main.aggreg.0.1.weight']
m_1_t = F.conv2d(test, w1_t, bias=None, groups=1, padding=get_same_padding(1))
m_2_t = F.conv2d(m_1_t, w2_t, bias=None, groups=in_c*3, padding=get_same_padding(5))
m_3_t = F.conv2d(m_2_t, w3_t, bias=None, groups=head*3, padding=get_same_padding(1))
m_q_t = torch.cat([m_1_t[:,0:in_c,:,:], m_3_t[:,0:in_c,:,:]], dim=1)
m_k_t = torch.cat([m_1_t[:,in_c:in_c*2,:,:], m_3_t[:,in_c:in_c*2,:,:]], dim=1)
m_v_t = torch.cat([m_1_t[:,in_c*2:,:,:], m_3_t[:,in_c*2:,:,:]], dim=1) # 1, 128, 1, 2 -> 1, 8, 16, 2
m_q_t = torch.reshape(m_q_t, (1,-1,dim,H*W,),)
m_q_t = torch.transpose(m_q_t, -1, -2)
m_k_t = torch.reshape(m_k_t, (1,-1,dim,H*W,),)
m_k_t = torch.transpose(m_k_t, -1, -2)
m_v_t = torch.reshape(m_v_t, (1,-1,dim,H*W,),)
m_v_t = torch.transpose(m_v_t, -1, -2)
print(m_q_t)
print(m_q_t_real)
print(torch.all(m_q_t_real==m_q_t).item())
print(torch.all(m_k_t_real==m_k_t).item())
print(torch.all(m_v_t_real==m_v_t).item())