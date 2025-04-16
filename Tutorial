-This project is a modified version of the paper "TUTR: Trajectory Unified Transformer for Pedestrian Trajectory Prediction":
https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf

-The original code can be accessed here:
https://github.com/lssiair/TUTR?tab=readme-ov-file#tutr-trajectory-unified-transformer-for-pedestrian-trajectory-prediction

-We have change their original design and we are going to use it on Nuscenes prediction challenge dataset. The dataset has been processed for you and included in the project folder. Each sample of the dataset includes the states of the target vehicle and the states of up to 10 agents in its surrounding for 6 seconds (13 time indices) plus a binary mask to for masking the agents that are far away from the target vehicle or in the case that there were less than 10 neighboring agents. We would like to predict the position of the target agents for 4 seconds (for 8 time indices) using the 2 seconds (5 time indices) states of the target vehicle and neighboring agents.   

-Before you start, please note that you need a gpu with cuda compatibility for this assignment. Download and install anaconda3 using the following link:
https://docs.anaconda.com/free/anaconda/install/linux/
-Install CUDA, you can use the following tutorial if you have Ubuntu:
https://www.cherryservers.com/blog/install-cuda-ubuntu

-Download and unzip the folder

-open a terminal in the project folder path

-create a conda evironment with python 3.9:
conda create -n courseproject python=3.9

-activate the environment you made:
conda activate courseproject

-install the requirements in the requirements.txt file:
pip install -r requirements.txt

now you should be able to run the train_eval.py file:
python3 train_eval.py



Part 1) Run the code and submit the best (lowest) statistics of the ADE and FDE error on test set

Part2) change line 53 in train_eval.py to "parser.add_argument('--minADEloss', type=bool, default=True)", complete the MinADE_loss class in (train_eval.py lines 90 to 118) and run the code again. 
You should see a drastic change in the best (lowest) statistics of the ADE and FDE error on test set. Report the statistics
Note that, in this setup, during training the output of the model is num_k(=10) trajectories, contrary to part 1 that the model output was only 1 trajectory with the highest probability during training. Therefore, the loss should be the minimum average difference between these 10 trajectories and the ground truth. Please see the comments in the code (lines 91 to 118)
 
-Example of a custom MSE loss function for pytorch to help you with part 2:

class Custom_MSE(nn.Module):
  def __init__(self):
    super(Custom_MSE, self).__init__();

  
   def __call__(self, predictions, target):
     square_difference = torch.square(y_predictions - target)
     loss_value = torch.mean(square_difference)
     return loss_value


