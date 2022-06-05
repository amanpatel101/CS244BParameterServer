servers=5
workers=1
iteration_to_fail=40
eval_interval=1
python src/main.py \
--num_servers $servers \
--num_workers $workers \
--num_iterations 100 \
--checkpoint 5 \
--do_failure_test 1 \
--iteration_to_fail $iteration_to_fail \
--eval_interval $eval_interval \
--output_path "live_demo/training_run.json"

echo "Saved training output to live_demo/training_run.json"

python live_demo/plot_training_results.py live_demo/training_run.json

echo "Saved plot to live_demo/training_plot.png"
