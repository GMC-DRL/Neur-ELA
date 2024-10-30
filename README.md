# Neural Exploratory Landscape Analysis


## Requirements
You can install all of dependencies of NeurELA via the command below.
```bash
pip install -r requirements.txt
```

After installing these dependecies, **please** refer to [instruction_to_modify_pypop7](instruction_to_modify_pypop7.md) for help to modify the source code of PyPop7 to meet our requirements.

## Train
The training process can be easily activated via the command below.
```bash
python run.py
```
For more adjustable settings, please refer to `run.py` and `config.py` for details.

Recording results: Log files and `NeurELA` model checkpoints will be saved to `./records`, the file structure is as follow.
```
records
|--run_name
   |--log_file
      |--logging files
      |--...
   |--saved_model
      |--model checkpoint
      |--...
```

## Zero-shot

Once the NeurELA checkpoint saved, you can validate its zero-shot performance via running the commmand below. Note that you should provide `load_path` correctly 

```bash
python zero-shot.py
```

You can modify `testsuits` or `MetaBBO algorithms` in `zero-shot.py` for specific requirements.

## Fine-tune

You can activate the fine-tuning process by running the command below,

```bash
python transfer.py
```

Similar to zero-shot process, you can modify `testsuits` or `MetaBBO algorithms` in `transfer.py` for specific requirements.

## Citing NeurELA

The PDF version of the paper is available [here](https://arxiv.org/pdf/2404.08239). If you find our NeurELA useful, please cite it in your publications or projects.

```latex
@article{ma2024neural,
  title={Neural Exploratory Landscape Analysis},
  author={Ma, Zeyuan and Chen, Jiacheng and Guo, Hongshu and Gong, Yue-Jiao},
  journal={arXiv preprint arXiv:2408.10672},
  year={2024}
}
```
