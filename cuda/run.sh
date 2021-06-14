#!/bin/bash

cd ./build/

cmake ..

cmake --build .

./whale_test

> ./out/memory_alloc_dealloc_time.csv

for i in {1..5}
do
 ./whale_test
done

cd ..
