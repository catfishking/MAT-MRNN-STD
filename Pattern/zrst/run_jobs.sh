for i in $(seq $1 $2)
do
(
echo $i
python run_script_35.py $i 
sleep 0.5
)&
done 
