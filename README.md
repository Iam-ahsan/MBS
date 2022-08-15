
# MBS
MBS: Optimizing Inference Serving on Serverless Platforms

**prerequisite**

- [AWS  Cli](https://aws.amazon.com/cli/)
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---
**Deployment:**

To deploy the Lambda serverless function follow the instruction in the [How to deploy deep learning models with aws lambda and tensorflow](https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/). However, this tutorial do not support batching.


Our updated [model and lambda package](https://drive.google.com/drive/folders/1R5eJ-dQZDmTU45-YBj1CJyiYWsExTWvN?usp=sharing) supports batching. The batching enabled lambda packge can be deployed by following the same steps mentioned in the demo and just by replacing the model file and packge file. We use Tensorflow 1.8 in our packge. At the time of our experiments this was the most latest Tensorflow version which could be zipped to 50 MB (Lambda limitation). 

**Find optimal configuration**
- To find the optimal configuration of the serverless environment, run the ./model/MBS_solver.py python script.
- MBS_solver.py_ must be run with python3 and requires the following modules:
   1. argparse
   2. numpy
   3. matplotlib
   4. scipy
   5. ortools
- To print the help, try: _python MBS_solver.py --help_
- To run the solver, try: _python MBS_solver.py --model TF-inceptionV4 --percentile 0.95 --slo 0.00003 --constraint cost --trace Twitter --start 1 --end 1_
---
