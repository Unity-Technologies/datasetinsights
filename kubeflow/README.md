Train and Evaluate Model using Kubeflow
=======================================


## Train a model on Kubeflow for SynthDet Project
Go to this pipeline [url](https://datasetinsights.endpoints.unity-ai-thea-test.cloud.goog/_/pipeline/#/pipelines/details/93dfcb42-5846-4a31-b74b-f25efd846c65)
press the button `Create Run` at the top of the screen (It has a solid blue fill and white letters)

0. set `docker_image` to be gcr.io/unity-ai-thea-test/datasetinsights:<git-comit-sha> Where <git-comit-sha> is the sha from the latest version of master in the thea repo. It should be the latest commit in the history: [link](https://gitlab.internal.unity3d.com/machine-learning/thea/commits/master).
1. (optional) rename the Run name (3rd from top under Run details) to something which describes the job, if not the job will be called Run of train_synthetic_validate_synthetic with an appended hash
2. update `logdir` to be the current date and time e.g. gs://thea-dev/runs/20200413-1015
3. update `auth_token` to be a non-expired auth token (if you don't have usim installed follow these instructions https://github.cds.internal.unity3d.com/unity/google-dr-paper#usim-run-instructions) to do so
4. Once you have USim Installed, Download Usim [boundle](https://github.com/Unity-Technologies/Unity-Simulation-Docs/releases)
5. Run `usim refresh auth` to refresh your authorization token
6. Run `usim inspect auth` to view your authentication, your access token is the string on the first row after access token: Bearer
7. update the run execution id to be the one from the synthetic dataset you would like to train and validate on https://docs.google.com/spreadsheets/d/18DOyjqjXnor2nlemhB7QwLO0ZVoEQvX8TB4yAkS2das/edit#gid=1096748050
8. Press start
9. to check the progress of your model run `docker run -p 6006:6006 -v $HOME/.config:/root/.config:ro tensorflow/tensorflow tensorboard --host=0.0.0.0 --logdir gs://your/log/directory`. Open `http://localhost:6006` to view the dashboard. This command assumes you have run `gcloud auth login` command and the local credential is stored in `$HOME/.config` which is mounted to the home directory inside docker. It must have read permission to `gs://your/log/directory`
10. If the mAP and mAR for validation are leveling off then you can terminate the run early; it's unlikely the model's performance will improve.
11. The model will save checkpoints after every epoch to the logdir with the format gs://logdir/ep#.estimator e.g.
gs://thea-dev/runs/20200328_221415/FasterRCNN.ep24.estimator

### Understanding the tensorboard output
Loss is the combined average loss of faster rcnn's four loss functions.
mAP and mAR are the mean Average Precision and mean Average Recall across all classes see  [this blog](https://www.google.com/search?q=mAP+and+mAR+object+detection&oq=mAP+and+mAR+object+detection&aqs=chrome..69i57j69i64l4.7407j0j7&sourceid=chrome&ie=UTF-8)
for more information on AP.

## Evaluate/Test your model on Kubeflow for SynthDet Project

1. Go to this pipeline [url](https://datasetinsights.endpoints.unity-ai-thea-test.cloud.goog/_/pipeline/#/pipelines/details/2c1e3378-7426-4559-8fe7-87b66a5797b0) press the button Create Run at the top of the screen
 (It has a solid blue fill and white letters).
2. Set `test_split`. The default value is "test", which means you can test your model on the whole testing dataset. User can also specify some subsets of the testing dataset. Available splits are: "test_high_ratio", "test_low_ratio", "test_low_contrast", "test_high_contrast", which stands for high foreground/background ratio data, low foreground/background ratio data, low contrast data, and high contrast data.
3. Set `docker_image` to be gcr.io/unity-ai-thea-test/datasetinsights:<git-comit-sha> Where <git-comit-sha> is the sha from the latest version of master in the thea repo. It should be the latest commit in the history: [link](https://gitlab.internal.unity3d.com/machine-learning/thea/commits/master).
4. (optional) rename the Run name (3rd from top under Run details) to something which describes the job, if not the job will be called Run of test_groceries_real with an appended hash
5. update the logdir to be the current date and time e.g. gs://thea-dev/runs/20200413-1015
6. update the checkpoint file to be the checkpoint from the model you would like to evaluate. The format for checkpoint files is logdir/FasterRCNN.ep#.estimator For instance if you wanted to load the checkpoint for model after its 25th epoch which had the logdir gs://thea-dev/runs/20200328_221415 the file would be gs://thea-dev/runs/20200328_221415/FasterRCNN.ep24.estimator
7. Press Start
8. To view the progress of your evaluation click on the node `evaluate` go to logs. If you get the error Warning: failed to retrieve pod logs. Possible reasons include cluster autoscaling or pod preemption, click the link  Stackdriver Kubernetes Monitoring.
9. Once your run has completed you can find the metrics in the logs, to find mAR search for the log `AR has mean result:` for mAP search for `AP has mean result:` and to find validation loss look for `validation loss is` alternately you can look at the tensorboard for this run (there will only be one datapoint per graph)

## Create New Pipeline

The development virtual environment have [kfp](https://pypi.org/project/kfp/) installed.
This is currently not required by datasetinsignts framework unless we want to directly use CLI to create/submit new pipelines.

Compile Kubeflow pipeline

```
dsl-compile --py=pipelines.py --function=<function> --output=compiled/<function>.yaml
```

Replace `<function>` to the pipeline function you want to compile.
This will create a file `compiled/<function>.yaml` which can be uploaded to kubeflow pipeline for executions. Next, go to kubeflow dashboard, upload and create new pipeline using the above pipeline. You should be able to create a new parameterize experiment to run kubeflow pipeline following this [tutorial](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart).
