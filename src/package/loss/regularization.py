import torch
import torch.nn as nn

class _Regularization(nn.Module):
    def __init__(self,model,weight_decay,logger,p=2):
        '''
        :param model
        :param weight_decay
        :param p: p=0 -> L2, p=1 -> L1
        '''
        super(_Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.logger=logger
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        only need to penalize the weight term install of the bias term
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        :param weight_list:
        :param p:
        :param weight_decay:
        :return: regularization loss
        '''
        reg_loss=0
        for _, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        :param weight_list:
        '''
        self.logger.info("---------------regularization weight---------------")
        for name, _ in weight_list:
            self.logger.info("\t{}".format(name))