# Mobilenet-v2-compression-for-CIFAR-10
to train the base model 
python train_base.py

to prune the model (ensure that base_model.pth is present after training the baseline
python iterative_pruning.py 

to quantize 
python quantize.py model_path --weight-bits 8 --act-bits 8

to evaluate the model 
for non quantized models 
python test.py model_path  

for quantized models (also input the number of bits quntization 
python test.py model_path --bits 8 


python iterative_pruning.py


