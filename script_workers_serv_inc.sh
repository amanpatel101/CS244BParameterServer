for try in 1 2 3
do
	servers=12
	workers=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"


	echo $try
	servers=6
	workers=2
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"


	servers=4
	workers=3
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=3
	workers=4
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=2
	workers=6
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

	servers=1
	workers=12
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"


	servers=4
	workers=2
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"


	servers=2
	workers=4
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
	--output_path "results/servers_"$servers"_workerss_"$workers"_try_"$try".json"

done
