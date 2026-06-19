./run_until_it_ends.sh experiments/run_gridsearch.py \ 
                                        --input_folder="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
                                        --output_folder="test_none" \
                                        --model="v1" \
                                        --model_runner="v1" \
                                        --params_file="test_none.yaml"



./run_until_it_ends.sh experiments/run_keras_tuner.py \ 
                                        --input_folder="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example/juxtaposed_scalograms_L" \
                                        --output_folder="test_juxtaposed" \
                                        --model="v1" \
                                        --model_runner="v1" \
                                        --max_trials=5 \
                                        --params_file="test_juxtaposed.yaml"
                                                    

./run_until_it_ends.sh experiments/run_cross_validation_loso.py \  
                                        --input_folder="data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
                                        --output_folder="useless" \
                                        --model="v1" \
                                        --params_file="test_rpca.yaml"



./run_until_it_ends.sh experiments/apply_rpca_simple.py \
                                        --image_path "data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example/img_1234.png" \
                                        --output_folder "experimento_customizado" \
                                        --lambdas 0.1 0.15 0.25 \
                                        --cmap "gray"


./run_until_it_ends.sh experiments/apply_rpca_juxtaposed.py \
                                        --input_folder "data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
                                        --output_folder "resultado_juxtaposto_teste" \
                                        --lamb 0.15 \
                                        --mu 0.5 \
                                        --tolerance 1e-6 \
                                        --max_iteration 2000



./run_until_it_ends.sh experiments/apply_rpca_isolated.py \
                                        --input_folder "data/generated_scalograms_ALL_gray_overlap0.733_extra_input_example" \
                                        --output_folder "resultado_isolado_teste" \
                                        --cmap "viridis" \
                                        --lamb 0.2