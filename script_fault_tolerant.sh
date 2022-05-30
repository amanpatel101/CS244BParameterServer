for try in 1 2 3
do
	servers=5
	workers=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
    --do_failure_test 1 \
    --iteration_to_fail 30 \
	--output_path "results/fault_tolerance/fault_tolerant_failure_{$iteration_to_fail}_try_${try}.json"

	servers=5
	workers=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
    --do_failure_test 1 \
    --iteration_to_fail 130 \
	--output_path "results/fault_tolerance/fault_tolerant_failure_{$iteration_to_fail}_try_${try}.json"

	servers=5
	workers=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
    --do_failure_test 1 \
    --iteration_to_fail 230 \
	--output_path "results/fault_tolerance/fault_tolerant_failure_{$iteration_to_fail}_try_${try}.json"

	servers=5
	workers=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 1 \
    --do_failure_test 1 \
    --iteration_to_fail 330 \
	--output_path "results/fault_tolerance/fault_tolerant_failure_{$iteration_to_fail}_try_${try}.json"




done
