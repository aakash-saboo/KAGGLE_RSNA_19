
model=model001
gpu=0
fold=1
ep=1
tta=5
clip=1e-6
conf=./conf/${model}.py

snapshot=./model/${model}/fold${fold}_ep${ep}.pt
valid=./model/${model}/fold${fold}_ep${ep}_valid_tta${tta}.pkl
test=./model/${model}/fold${fold}_ep${ep}_test_tta${tta}.pkl
sub=./data/submission/${model}_fold${fold}_ep${ep}_test_tta${tta}.csv

python3 -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
python3 -m src.postprocess.make_submission --input ${test} --output ${sub} --clip ${clip}
#kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "" -f ./data/submission/${sub}



