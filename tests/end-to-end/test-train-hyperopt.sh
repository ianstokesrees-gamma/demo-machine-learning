TEST=train-hyperopt
CMD='python -m acmemodel.main --train --hyperopt'
DIVIDER='=============================================='

echo "Starting test $TEST: $CMD"
echo $DIVIDER
$CMD
EXIT=$?
echo $DIVIDER

if (( $EXIT == 0 )); then
    echo "Test $TEST PASSED"
else
    echo "Test $TEST FAILED with exit code $EXIT"
fi
