# tinyrwkv7

Based on: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo_rnn.py

download model: https://huggingface.co/BlinkDL/rwkv-7-pile/tree/main

Run the code:
```
python3 -m venv venv
source .venv/bin/activate
pip install -r requirements.txt 

python3 rwkv7.py 
```

Current output:
``` 
People from France speak French,
People from France speak French, and
People from France speak French, and the
People from France speak French, and the rest
People from France speak French, and the rest of
People from France speak French, and the rest of the
People from France speak French, and the rest of the world
People from France speak French, and the rest of the world speaks
People from France speak French, and the rest of the world speaks English
People from France speak French, and the rest of the world speaks English.
```
