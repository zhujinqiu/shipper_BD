import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, BertModel, AutoModel
from torch import nn
from torch.autograd import Variable
import torch
from torch.nn import functional as F

class Bert(nn.Module):
    def __init__(self, num_labels):
        super(Bert, self).__init__()

        # Load Model with given checkpoint and extract its body
        self.endode = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext", output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext") #预训练模型选取www.huggingface.co,BERT模型里一致
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.endode(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))
        return logits

    ####################储存与读取模块###### ************用于储存微调后的模型
    def _save(self, save_path='./'):
        encoder_path = os.path.join(save_path, "encoder")
        dense_path = os.path.join(save_path, "project_layers")
        if not os.path.exists(encoder_path):
            os.mkdir(encoder_path)
        if not os.path.exists(dense_path):
            os.mkdir(dense_path)
        # save BERT weight,config and tokenizer
        model_to_save = (self.endode.module if hasattr(self.endode, "module") else self.endode)
        model_to_save.save_pretrained(encoder_path)
        self.tokenizer.save_pretrained(encoder_path)

    def _load(self, save_path='./') -> bool:
        encoder_path = os.path.join(save_path, "encoder")
        dense_path = os.path.join(save_path, "project_layers")
        if not os.path.exists(encoder_path):
            return False
        if not os.path.exists(dense_path):
            return False
        try:
            # load BERT weight,config and tokenizer
            self.endode = AutoModel.from_pretrained(encoder_path, output_hidden_states=True)
            self.endode.cuda()
            # load project_layers
        except FileNotFoundError:
            print("model checkpoints file missing: %s" % save_path)
            return False
        return True

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss