# Runing a Model  
To run the model, execute the following command:               
 <span style="color:#333333">`  </span>            

 CUDA_VISIBLE_DEVICES=0 python main.py --dataset FB15k-237 --num_iterations 500 --batch_size 256--lr 0.003 --dr 1.0 --edim 400 --rdim 400 --input_dropout 0          
                                       
                                       --hidden_dropout0 0.4 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1` </span>  
# Requirements  
The codebase is implemented in Python 3.6.6. Required packages are:  
 

<span style="color:#333333">`numpy 1.15.1          

pytorch    1.0.1` </span>          
