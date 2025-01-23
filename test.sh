# for any model，you should run all attack
# all model from the RobustBench https://robustbench.github.io/ except ResNet18
# cifar10 model list[ResNet18, Debenedetti2022Light_XCiT-M12,Debenedetti2022Light_XCiT-S12,Pang2022Robustness_WRN28_10,Peng2023Robust]
# cifar100 model list[Debenedetti2022Light_XCiT-M12,Cui2023Decoupled_WRN-28-10]
# ImageNet model list[Debenedetti2022Light_XCiT-S12,Liu2023Comprehensive_ConvNeXt-B]

python ./main.py \
    --data 'cifar10'\
    --model-name ResNet18 \
    --attack linf-pgd \
    --attack-iter 40 \
    --attack-loss ce \
    --attack-eps  8/255  \
    --attack-step 2/255 \
    --s-loss-name ce \
    --eps 8/255 \
    --batch-size 64  \
    --topk 5

python ./main.py \
    --data 'cifar10'\
    --model-name ResNet18 \
    --attack linf-pgd \
    --attack-iter 40 \
    --attack-loss cw \
    --attack-eps  8/255  \
    --attack-step 2/255 \
    --s-loss-name ce \
    --eps 8/255 \
    --batch-size 64  \
    --topk 5

python ./main.py \
    --data 'cifar10'\
    --model-name ResNet18 \
    --attack linf-apgd \
    --attack-iter 100 \
    --attack-loss ce \
    --attack-eps  8/255  \
    --attack-step 2/255 \
    --s-loss-name ce \
    --eps 8/255 \
    --batch-size 64  \
    --topk 5

python ./main.py \
    --data 'cifar10'\
    --model-name ResNet18 \
    --attack linf-apgd \
    --attack-iter 100 \
    --attack-loss dlr \
    --attack-eps  8/255  \
    --attack-step 2/255 \
    --s-loss-name ce \
    --eps 8/255 \
    --batch-size 64  \
    --topk 5

python ./main.py \
    --data 'cifar10'\
    --model-name ResNet18 \
    --attack linf-fab \
    --attack-iter 100 \
    --attack-loss ce \
    --attack-eps  8/255  \
    --attack-step 2/255 \
    --s-loss-name ce \
    --eps 8/255 \
    --batch-size 64  \
    --topk 5

python ./main.py \
    --data 'cifar10'\
    --model-name ResNet18 \
    --attack linf-square \
    --attack-iter 100 \
    --attack-loss ce \
    --attack-eps  8/255  \
    --attack-step 2/255 \
    --s-loss-name ce \
    --eps 8/255 \
    --batch-size 64  \
    --topk 5