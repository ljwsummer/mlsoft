
work=./

if ! echo $PYTHONPATH | egrep -q "(^|:)$work($|:)" 
then
    export PYTHONPATH=$PYTHONPATH:$work
fi

nosetests unit_tests/common
nosetests unit_tests/linear_model

