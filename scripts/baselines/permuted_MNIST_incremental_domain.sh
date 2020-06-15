OUTDIR=outputs/permuted_MNIST_incremental_domain/
REPEAT=10
mkdir -p ${OUTDIR}Offline/
mkdir -p ${OUTDIR}Adam/
mkdir -p ${OUTDIR}SGD/
mkdir -p ${OUTDIR}Adagrad/
python -u iBatchLearn.py --gpuid 0 --repeat $REPEAT --optimizer Adam    --n_permutation 20 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.0001  --offline_training  --output_dir ${OUTDIR}Offline/  | tee ${OUTDIR}Offline/experiment.log    &   
python -u iBatchLearn.py --gpuid 1 --repeat $REPEAT --optimizer Adam    --n_permutation 20 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.0001                      --output_dir ${OUTDIR}Adam/     | tee ${OUTDIR}Adam/experiment.log       &
python -u iBatchLearn.py --gpuid 2 --repeat $REPEAT --optimizer SGD     --n_permutation 20 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.001                       --output_dir ${OUTDIR}SGD/      | tee ${OUTDIR}SGD/experiment.log        &
python -u iBatchLearn.py --gpuid 3 --repeat $REPEAT --optimizer Adagrad --n_permutation 20 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP1000                                                     --lr 0.001                       --output_dir ${OUTDIR}Adagrad/  | tee ${OUTDIR}Adagrad/experiment.log    &