# Examples on CIAFR-10
# User can change the dataset, backbone, eps, strength, and evluation algorithm for different experiments 

# Generation
# AUE attack
python aue.py --dataset cifar10 --backbone resnet18 --eps 8 --strength 0.6 --gpu-id 0
# Targeted AAP attack
python aap.py --dataset cifar10 --backbone resnet18 --eps 8 --ref-strength 0.4 --gen-strength 0.4 --gpu-id 0
# Untargeted AAP attack
python aap.py --dataset cifar10 --backbone resnet18 --eps 8 --ref-strength 0.4 --gen-strength 0.4 --gpu-id 0 --untargeted

# Evaluation
# Clean SL
python evaluation_sl.py --experiment clean --dataset cifar10 --backbone resnet18 --gpu-id 0
# Clean SimCLR
python evaluation_cl.py --experiment clean --method SimCLR --dataset cifar10 --backbone resnet18 --gpu-id 0
# SL for AUE attack
python evaluation_sl.py --experiment aue --dataset cifar10 --backbone resnet18 --gpu-id 0 --poison-path results/aue/cifar100/resnet18/eps-8/constant-0.6/seed-1/poison.pt
# SimCLR for AUE attack
python evaluation_cl.py --experiment aue --method SimCLR --dataset cifar10 --backbone resnet18 --gpu-id 0 --poison-path results/aue/cifar100/resnet18/eps-8/constant-0.6/seed-1/poison.pt



# Examples on CIAFR-100
# User can change the dataset, backbone, eps, strength, and evluation algorithm for different experiments 

# Generation
# AUE attack
python aue.py --dataset cifar100 --backbone resnet18 --eps 8 --strength 1 --gpu-id 0
# Targeted AAP attack
python aap.py --dataset cifar100 --backbone resnet18 --eps 8 --ref-strength 0.8 --gen-strength 0.8 --gpu-id 0

# Evaluation
# Clean SL
python evaluation_sl.py --experiment clean --dataset cifar100 --backbone resnet18 --gpu-id 0
# Clean SimCLR
python evaluation_cl.py --experiment clean --method SimCLR --dataset cifar100 --backbone resnet18 --gpu-id 0
# SL for AUE attack
python evaluation_sl.py --experiment aue --dataset cifar100 --backbone resnet18 --gpu-id 0 --poison-path results/aue/cifar100/resnet18/eps-8/constant-1.0/seed-1/poison.pt
# SimCLR for AUE attack
python evaluation_cl.py --experiment aue --method SimCLR --dataset cifar100 --backbone resnet18 --gpu-id 0 --poison-path results/aue/cifar100/resnet18/eps-8/constant-1.0/seed-1/poison.pt

