'''
class loss는 데이터에따라  bce를 쓸지 안쓸지 정함
'''

import random
import torch
import torch.nn as nn

from util import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()  # sigmoid + CrossEntropy  -> multi label classification
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()


        # Constants signifying how much to pay for each respectivve part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):  # prediction:(N, 3, 13, 13, 17), target:(n,3,13,13,6)
        # Check where obj and noobj (we ignore if target == -1)
        # 6 : (object_prob, x, y, w, h, class)
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0


        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])  # target 이미지에 object가 없는 위치를 indexing
        )


        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)  # w와 h를 가진 3개의 anchor가 모든 셀에서 계산하기위해 broad casting을 사용
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]),torch.exp(predictions[..., 3:5])*anchors], dim=-1)
        # 내 생각: sigmoid(tx) + Cx 가 아닌 sigmoid(tx)만 있는 이유는 차원이 (N, 3, 13, 13, 17) 에서 13x13으로 나누어져 있기 때문
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()  # 실제 박스와 예측이 얼마나 겹쳤는가
        # detach: gradient가 전파되지 않는 텐서생성
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])
        # 내 생각 : 1을 예측하는 것이 아닌 true와 predict coordinate이 겹친정도(confidence)를 예측
        # 즉 object prob * IOU = confidence 가 0.8이고 에측값의 object prob가 0.4이면 0.4가 아닌 0.8이 되도록 학습을 진행시킴


        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates  Cx, Cy가 없는 이유는 위와 동일
        target[..., 3:5] = torch.log(  # tw = log(Bw/Pw)
            (1e-16 + target[..., 3:5] / anchors)  # 분자가 0이 됨을 막기 위함
        )  # width, height coordinates  # target을 bw,bh에서 tw, th상태로 만들어줌
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )


        #    print("__________________________________")
        #    print(self.lambda_box * box_loss)
        #    print(self.lambda_obj * object_loss)
        #    print(self.lambda_noobj * no_object_loss)
        #    print(self.lambda_class * class_loss)
        #    print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )










