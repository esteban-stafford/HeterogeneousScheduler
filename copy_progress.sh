
for model in model:{lublin_256,lublin_1024,PIK-IPLEX-2009-1,CTC-SP2-1996-3.1-cln}:{homo,hetero}:BSLD; do 
   cp data/logs/$model/${model}_s2406/progress.txt ../${model}_s2406_progress.txt
done
cp data/logs/compare_models:{lub,PIK,CTC}*:{hetero,homo}*.dat ..
