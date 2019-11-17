model=model001
gpu=0
fold=4
conf=./conf/${model}.py

python3 -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
