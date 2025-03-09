The main requirements are [pytorch](http://pytorch.org/) (`v0.4.0`) and python `2.7`. Some dependencies that may not be installed in your machine are [tabulate](https://pypi.org/project/tabulate/) and [h5py](https://github.com/h5py/h5py). Please install other missing dependencies.

## Get started
1. Download preprocessed datasets
```bash
wget http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz
tar -xvzf datasets.tar.gz
```

**Updates**: The QMUL server is inaccessible. Download the datasets from this [google drive link](https://drive.google.com/open?id=1Bf0beMN_ieiM3JpprghaoOwQe9QJIyAN).

## How to train
2. Train the model 
```bash
python enhancedmain3.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/splits_90_10.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --verbose
```

## How to test
3. Test the model
```bash
python enhancedmain3.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/splits_90_10.json -m summe --gpu 0 --save-dir log/summe-split0 --split-id 0 --evaluate --resume log/summe-split0\best_model.pth.tar --verbose --save-results
```
If argument `--save-results` is enabled, output results will be saved to `results.h5` under the same folder specified by `--save-dir`. 

4. To visualize the score-vs-gtscore, simple do
```bash
python visualize_results.py -p path_to/result.h5
```

## Plot
5. We provide codes to plot the rewards obtained at each epoch. Use `parse_log.py` to plot the average rewards
```bash
python parse_log.py -p path_to/log_train.txt
```
The plotted image would look like
<div align="center">
  <img src="imgs/overall_reward.png" alt="overall_reward" width="50%">
</div>

If you wanna plot the epoch-reward curve for some specific videos, do
```bash
python parse_json.py -p path_to/rewards.json -i 0
```

You will obtain images like
<div align="center">
  <img src="imgs/epoch_reward_0.png" alt="epoch_reward" width="30%">
  <img src="imgs/epoch_reward_13.png" alt="epoch_reward" width="30%">
  <img src="imgs/epoch_reward_15.png" alt="epoch_reward" width="30%">
</div>

If you prefer to visualize the epoch-reward curve for all training videos, try `parse_json.sh`. Modify the code according to your purpose.

## Visualize summary
You can use `summary2video.py` to transform the binary `machine_summary` to real summary video. You need to have a directory containing video frames. The code will automatically write summary frames to a video where the frame rate can be controlled. Use the following command to generate a `.mp4` video
```bash
python summary2video.py -p log/summe-split0/result.h5 -d frames -i 0 --fps 30 --save-dir log --save-name summary.mp4
```





