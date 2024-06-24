DIVIDER='=============================================='
PATHS=( ".env" "data/australian_open.csv" )

echo "Checking for file paths:"
for p in $PATHS; do
    echo -n "\t$p: "
    if [ ! -f "$p" ]; then
        echo "exists"
    else
        echo "does not exist"
        exit -1
    fi
done

echo "Starting test $TEST"
echo $DIVIDER

python -m acmemodel.main --train | tee logs/train.out
ID=`cat logs/train.out | tail -1`
echo "ID: $ID"
python -m acmemodel.main --deploy-model --model-id $ID
python -m acmemodel.main --features
echo $DIVIDER
