FUNCTIONS=(
    "train_on_synthdet_sample"
    "train_on_real_world_dataset"
    "evaluate_the_model"
)

for function in "${FUNCTIONS[@]}"
do
    echo "Complining $function ..."
    dsl-compile --py=pipelines.py --function=$function --output="compiled/$function.yaml"
done
