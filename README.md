# Python script to train a huggingface Model


Example usage:

`python train_model.py vilsonrodrigues/falcon-7b-instruct-sharded Amod/mental_health_counseling_conversations finetuned-model akumar23/mental-falcon-7b`

OR if you don't want to push the model to huggingface

`python train_model.py vilsonrodrigues/falcon-7b-instruct-sharded Amod/mental_health_counseling_conversations finetuned-model`

To use with Transformers and Tokenizer like the parent falcon model can be you would need to 
generate the config.json file using `config_gen.py` and then push that config.json to the 
huggingface repo.

Example usage for that script:

```
python config_gen.py [huggingface-repo]

python config_gen.py akumar23/mental-falcon-7b
```