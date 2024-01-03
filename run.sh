
#for dataset in cifar10 svhn tinyimagenet
#do
#    for s in a b c d e
#    do
#          python3 main.py --gpu 0 --ds ./exps/$dataset/spl_$s.json --config ./configs/pcssr/$dataset.json --save $dataset/$s'_m5' --method csgrl --test_interval 1
#    done
#done

for dataset in ood
do
  python3 main_ood.py --gpu 0 --ds ./exps/cifar10/$dataset"_spl".json --config ./configs/pcssr/cifar10.json --save '_m5' --method csgrl --test_interval 1
done
