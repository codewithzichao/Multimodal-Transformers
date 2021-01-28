# Meme Classification for Tamil Language at EACL2021 Workshop

Our source code for EACL2021 workshop: Meme Classification for Tamil Language.

**Updated:** Source code is released!🤩
> I will release the code very soon.

## Repository structure
```shell
├── MyLoss.py                          # Impelmentation of some loss function 
├── README.md                   
├── __init__.py
├── args.py                            # declare some argument
├── ckpt
│   └── README.md
├── data                               # store data
│   └── README.md       
├── gen_data.py                        # generate Dataset
├── install_cli.sh                     # install required package
├── logfile                            # store logfile during training
├── main.py                            # train model         
├── model.py                           # define model
├── multimodal_attention.py            # Implentation of multimodal attention layer
├── pred_data
│   └── README.md
├── preprocessing.py                   # preprocess the data
├── pretrained_weights                 # store pretrained weights of resnet and xlm-roberta
│   └── README.md
├── run.sh                             # run model
└── train.py                           # define training and validation loop             

```

## Installation
Use the following command so that you can install all of required packages:
```shell
sh install_cli.sh
```

## Preprocessing
The first step is to preprocess the data. Just use the following command:
```shell
python3 -u preprocessing.py
```

## Training
The second step is to train our model. Use the following command:
```shell
nohup sh run.sh > run_log.log 2>&1 &
```

## Inference
The final step is inference after training. Use the following command:
```shell
nohup python3 -u inference.py > inference.log 2>&1 &
```
Congralutions! You have got the final results!🤩


> If you use our code, please indicate the source.