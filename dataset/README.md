# Dataset

## CALVIN

To download the CALVIN task_D_D dataset:
```bash
cd $LUMOS_ROOT/dataset
sh download_data.sh calvin
```

To have a faster dataloading, first preprocess the dataset to extract the essential info:
```bash
cd $LUMOS_ROOT/scripts
python preprocess_calvin_dataset.py
```


## LUMOS Real-World Play Data

To download the LUMOS real-world play dataset:
```bash
cd $LUMOS_ROOT/dataset
sh download_data.sh lumos_dataset
```
