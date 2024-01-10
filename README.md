# Runing a Model  
To run the model, execute the following command:  
'''python  
 CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k-237 --num_iterations 500 --batch_size 256
                                       --lr 0.003 --dr 1.0 --edim 400 --rdim 400 --input_dropout 0.3 
                                       --hidden_dropout0 0.4 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1  
# Requirements  
The codebase is implemented in Python 3.6.6. Required packages are:  
'''python  
numpy      1.15.1
pytorch    1.0.1
