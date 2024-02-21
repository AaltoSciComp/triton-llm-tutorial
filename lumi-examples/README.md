## DPO example on LUMI
### Create environment
This environment is based on a container provided by LUMI. The container contains pytorch,torchvision, torchaudio.

Run the following script to obtain the container image and install additional requirements by using the requirements.txt file in the folder. 

```
sbatch buildEnv.sh
```
### Before you run this, modify the buildEnv.sh file to use your project id.


Then run the following script to start the model training.
```
sbatch run_dpo.sh
```
### Before you run this, modify the run_dpo.sh file to use your project id and set up environment variables properly.
