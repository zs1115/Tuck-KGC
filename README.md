### Running a model
## To run the model, execute the following command
 CUDA_VISIBLE_DEVICES=0 python main.py --dataset dia --num_iterations 500 --batch_size 256
                                       --lr 0.0003 --dr 1.0 --edim 400 --rdim 400  --label_smoothing 0.1

### Requirements
## The codebase is implemented in Python 3.6.6. Required packages are:
numpy      1.15.1
pytorch    1.0.1
