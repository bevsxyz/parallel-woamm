default: sequential cuda

cuda:
    # Switch to the desired directory
    cd ./cuda &&  ./run.sh

sequential:
    cd ./sequential && ./run.sh
