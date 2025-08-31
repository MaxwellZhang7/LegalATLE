# This is a pytorch implementation of the paper[LegalATLE: An Active Transfer Learning Framework for Legal Triple Extraction]

## Environment
  - Pytorch 1.7.1
  - Python 3.6.13

## How to run the code?

1. Download pretrained models to folder `PLM`.

  * [bert-wwm-cn:](https://github.com/yyang1201/chinese-bert-wwm)

  the PLM folder should consist of the following files:
    ```
    |-- PLM
        |-- bert-cn-wwm
            |-- config.json
            |-- pytorch_model.bin
            |-- vocab.txt
    ```
2. Modify the parameters in `main.py`

3. Run the  `main.py`
