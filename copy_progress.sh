
for model in model:{lublin_256,lublin_1024,PIK-IPLEX-2009-1,CTC-SP2-1996-3.1-cln}:{homo,hetero}:BSLD; do 
   cp data/logs/$model/${model}_s2406/progress.txt ../${model}_s2406_progress.txt
done
for model in model:cl{4,8,16}:KIT-FH2-2016-1:{homo,hetero}_x8:BSLD; do 
   cp data/logs/$model/${model}_s2406/progress.txt ../${model}_s2406_progress.txt
done
cp data/logs/compare_models:*{lub,PIK,CTC,KIT}*:{hetero,homo}*.dat ..
