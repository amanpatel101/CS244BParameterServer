for try in 1 2 3
do
	servers=5
	workers=1
    iteration_to_fail=50
    eval_interval=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 5 \
    --do_failure_test 1 \
    --iteration_to_fail $iteration_to_fail \
    --eval_interval $eval_interval\
	--output_path "results/fault_tolerance/fault_tolerant_failure_${iteration_to_fail}_try_${try}.json"

	servers=5
	workers=1
    iteration_to_fail=150
    eval_interval=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 5 \
    --do_failure_test 1 \
    --iteration_to_fail $iteration_to_fail \
    --eval_interval $eval_interval\
	--output_path "results/fault_tolerance/fault_tolerant_failure_${iteration_to_fail}_try_${try}.json"

	servers=5
	workers=1
    iteration_to_fail=250
    eval_interval=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 5 \
    --do_failure_test 1 \
    --iteration_to_fail $iteration_to_fail \
    --eval_interval $eval_interval\
	--output_path "results/fault_tolerance/fault_tolerant_failure_${iteration_to_fail}_try_${try}.json"

	servers=5
	workers=1
    iteration_to_fail=350
    eval_interval=1
	echo $try
	python src/main.py \
	--num_servers $servers \
	--num_workers $workers \
	--num_iterations 500 \
	--checkpoint 5 \
    --do_failure_test 1 \
    --iteration_to_fail $iteration_to_fail \
    --eval_interval $eval_interval\
	--output_path "results/fault_tolerance/fault_tolerant_failure_${iteration_to_fail}_try_${try}.json"




done
