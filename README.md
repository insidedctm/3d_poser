## 3d_poser
Code repository for MSc Computational Statistics and Machine Learning thesis


### Training 

Data for training needs to be placed in the data/train folder, with validation data in data/val

To train the batch_normalisation model with strong supervision:

  ```python bn_main.py --name=bn_strong --learning_rate=0.0001  --batch_size=32 --gf_dim=32 --is_sup_train=True --max_iter=10001```

To train the dropout model with strong supervision (using the batch normalisation model as a starting point):

  ```
  python dropout_main.py --name=dropout_strong --learning_rate=0.001 --batch_size=32 --gf_dim=32 --is_sup_train=True 
        --model_dir=checkpoint/bn_strong --keep_prob=0.8 --max_iter=10001
  ```

To fine-tune the batch_normalisation models with weak supervision:

  ```
  python bn_main.py --name=bn_strong_finetune --learning_rate=0.000001 --batch_size=2 --model_dir=checkpoint/bn_strong
  --gf_dim=32 --is_sup_train=False --key_loss=True --max_iter=1001
  ```

### Evaluation 

Models are saved to the checkout folder

To evaluate on SURREAL data using Batch Normalisation model
 
  ```python model_evaluate.py --name=bn_strong --batch_size=25 --data_name=SURREAL --max_iter=12528```

To evaluate on Human3.6M data using Batch Normalisation model
  
  ```python model_evaluate.py --name=bn_strong --batch_size=25 --data_name=H36M --max_iter=100000```

To evaluate on SURREAL data using Dropout (for standard dropout use keep_prob=1)

  ```python model_evaluate.py --name=dropout_strong --batch_size=25 --data_name=SURREAL --max_iter=12528 --model_type=Dropout --keep_prob=1.0```

To generate one set of samples on SURREAL data using Monte Carlo (MC) dropout:

  ```python model_evaluate.py --name=dropout_strong --batch_size=25 --data_name=SURREAL --max_iter=12528 --model_type=Dropout --keep_prob=0.8```
