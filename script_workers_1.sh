for try in 1 2 3
do
	servers=1
	workers=6
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"




done

