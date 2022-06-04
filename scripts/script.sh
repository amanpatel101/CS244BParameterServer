for try in 1 2 3
do
	servers=1
	workers=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"



	servers=2
	workers=1
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"


	servers=4
	workers=1
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=6
	workers=1
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=8
	workers=1
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=10
	workers=1
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

done
