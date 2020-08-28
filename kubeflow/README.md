# Compile Kubeflow Pipelines

The development virtual environment have [kfp](https://pypi.org/project/kfp/) installed.
This is currently not required by datasetinsignts package unless you want to directly use CLI to create/submit new pipelines.

To compile Kubeflow pipeline, run:

```bash
dsl-compile --py=pipelines.py --function=<function> --output=compiled/<function>.yaml
```

Replace `<function>` to the pipeline function you want to compile.
This will create a file `compiled/<function>.yaml` which can be uploaded to kubeflow pipeline for executions. Next, go to kubeflow dashboard, upload and create new pipeline using the above pipeline. You should be able to create a new parameterize experiment to run kubeflow pipeline following this [tutorial](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart).
