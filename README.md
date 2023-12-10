# CS577-Project
Contribution:
Yue Zhao responsible for writing papers and collecting data sources，Chengcan Gao is 
responsible for running Visual Prompting via Image Inpainting code and  experimental results，
Weiye Gao  take the responsibility for presentation.

# Important:
Because the output model and the dataset for trainning is too big, so we delete the trainning dataset in figuares_dataset and result model in output_dir , otherwise, it can't been upload to github.

And the experiment result is to big, we only keep 100 photos and log.txt for each downstream tasks, it kept in evaluate/result directory. 

If the program can run, it should have last.ckpt in this project, but it is so big, so we don't upload it.

# The main Contribution:
The main contributions of the project can be summarized as follows:

Environment Setup: The project recommends using a stable version of Python 3.11 for running the code.

Code Instructions:

Training: The training process involves running the main_pretrain.py file. The dataset is located in the figures_dataset folder. To modify the training data, you can change the path in line 69 of main_pretrain.py. The CIFAR dataset used is CIFAR-10.

Evaluation: There are four evaluation scripts available on GitHub. The evaluation results can be found in the evaluate/result directory, and the checkpoints are saved in the output_dir folder.

Code Modifications:

In util/misc.py, change 'from torch._six import inf' to 'from torch import inf': torch._six is an old usage, and the new version requires importing inf directly from the torch module.

In util/pos_embed.py, in line 56, change 'dtype=np.float' to 'dtype=np.float64': There is a difference in usage between versions, and the new version requires np.float64 as the data type.

In evaluate/reasoning_dataloader.py, in line 59, change 'res_img=torch.tensor(output)[nn_indices]' to 'res_img = outputs.clone().detach()[nn_indices]': This modification creates a completely independent tensor by using .clone().detach() to avoid sharing the data and computation history of the original tensor.

In evaluate/evaluate_reasoning.py, in line 96, the code limits color to be a 2x3 matrix, representing white and target colors. However, after examining the code details:
np.reshape(target,(-1,3)) reshapes the input image target from an MxNx3 image matrix to a Kx3 matrix, where K = M*N, and each row represents the RGB values of a pixel.
np.unique(... , axis=0) finds the unique rows in this Kx3 matrix, meaning if there are duplicate colors, only one will be kept. Therefore, the final output size of color should be Xx3, where X depends on the number of different colors in the target image.
As a result, we did the modification as below:

Directly modify line 96 as follows:

Change 'assert colors.shape[0] == 2, colors' to 'assert colors.shape[0] == 3, colors', and the program will run successfully.

If not modified, an error will be thrown: AssertionError: 

[[  0   0 255]

 [  0 255   0]
 
 [255 255 255]] 

Additional Information:

The prep_cifar.sh script contains a program for processing the CIFAR dataset, converting it into image format for further use.

To separate the results and facilitate storage, a new evaluate_Foregroundsegm.py file was created for foreground segmentation, while object detection uses the same evaluation script.