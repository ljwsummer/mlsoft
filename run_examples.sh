
export PYTHONPATH=./
mkdir examples_output
python examples/linear_reg_example.py -c examples/model.cfg -m examples_output/model.linear_reg
python examples/logistic_reg_example.py -c examples/model.cfg -m examples_output/model.logistic_reg

