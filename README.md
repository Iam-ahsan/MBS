
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
- To run the solver, try: _python MBS_solver.py  --percentile 0.95 --slo 0.00003 --constraint cost --trace Twitter --start 1 --end 1_
---

**Run Experiments**
Usage:
python Clustering_BATCH2_Client.py
- Run experiment with default  Exponential arrival python Clustering_BATCH2_Client.py. This file takes the following argunments as command line argunments.
    1.num_classes # number of buffers
    2. num_requests # Intreger value that depicts the number of request types in the workload
    3. class_id # ID of the buffer a request is assigned to
    4. batch_size # optimal batch size calculated through MBS_soler.py
    5. t_out # optimal timeout calculated through MBS_soler.py
    6. interval # interval from the twitter traces ( integer between 0 and 23)
    7 allocated_memory # optimal memory calculated through MBS_soler.py
    
    **Run model**
- Model takes the arrival process i.e. inter-arrival time to calculate the efficient memory size, batch size, batch timeout and optimal number of buffers. 
-----
**Collect logs**
- Once the experiments are done three log files are generated for each buffer for within experiment.
  1. Lambda logs: These logs contains all the information regarding each lambda invocation i.e. print out values in the lambda function, init time, execution time, billing time, memory utilization, exceptions if any and error if any.
  2. Lambda per buffer per batch logs: These logs contains information regardin batch i.e. Batch starting time, Batch ending time, Batch size and Batch serivce time for a particular buffer.
  3. Lmabda per buffer pre request logs: This file contains all the information of each request i.e. arrival time, departure time, latency, and size of batch it was served in for a particular bufffer.
  
  
  **All log are collected through cloudwatch default configurations**
