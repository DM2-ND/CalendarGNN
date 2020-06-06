# Calendar Graph Neural Networks for Modeling Time Structures in Spatiotemporal User Behaviors
**Description: This repository contains the reference implementation of the *CalendarGNN* proposed in the paper *Calendar Graph Neural Networks for Modeling Time Structures in Spatiotemporal User Behaviors* submitted to KDD 2020.**

## Model
![The architecture of *CalendarGNN*](https://github.com/kdd2020calendargnn/CalendarGNN/blob/master/fig/CalendarGNN.png? "*CalendarGNN*")
The architecture of *CalendarGNN* incorporates two networked structures. One is a tripartite network of items, sessions, and locations.
The other is a hierarchical calendar network of hour, week, and weekday nodes.
It first aggregates embeddings of location and items into session embeddings via the tripartite network, and then generates user embeddings from the session embeddings via the calendar structure. The user embeddings preserve spatial patterns and temporal patterns of a variety of periodicity (e.g., hourly, weekly, and weekday patterns).
It adopts the attention mechanism to model complex interactions among the multiple patterns in user behaviors.

## Usage
### 1. Dependencies
This code package was developed and tested with Python 3.7 and [PyTorch 1.2.0](https://pytorch.org/).
Make sure all dependencies specified in the `./requirements.txt` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```
Other environment management tool such as [Conda](https://www.anaconda.com/) can also be used.

### 2. Data
**Note: Due to size limits, this repository does not contain the datasets of spatiotemporal user behaviors. Please download one or more datasets following the links below.**

Real spatiotemporal datasets can be found at:
+ B(w1): <https://bit.ly/2SpocOD>
+ B(w2): <https://bit.ly/2SM1T4T>

After downloading, please decompress it into the `./data/` folder. Predefined paths for locating necessary data files can be found in the `./config.py` file.

### 3. Run
To train the model, run
```
python main.py --model calendargnn --label gender --num_epochs 10
```
List of arguments:
+ `--model`: The model to use. Valid choices include `calendargnn` and `calendargnnattn`. Default is `calendargnn`
+ `--label`: The user label for prediction. Valid choices include `gender`, `income` and `age`. Default is `gender`
+ `--num_epochs`: Num of epochs for training. Default is `10`
+ `--rand_seed`: Random seed. Default is `999`
+ `--hidden_dim`: Dimension of spatial/temporal unit embeddings. Default is `256`
+ `--pattern_dim`: Dimension of spatial/temporal patterns. Default is `128`
+ `--best_model_file`: Checkpoint file for storing the best model during training. Default is `./best_model.pt`
+ `--cuda`: Use GPU for training the model

During training, the script keeps writing to the checkpoint file specified by the `--best_model_file` argument. And, it should be explictly given when running batch jobs to avoid lossing training progress. We _highly_ recommend adding `--cuda` argument for training if possible.

## Examples
Other examples are provided in the `./demo.sh` file.

## Miscellaneous
We thank all anonymous reviewers for their valuable feedbacks. More experimental settings and details can be found in the main body and the supplementary materials of our original paper *Calendar Graph Neural Networks for Modeling Time Structures in Spatiotemporal User Behaviors*.
