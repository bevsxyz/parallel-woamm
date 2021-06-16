#!/bin/bash

rm ./out
mkdir ./out

cd ./build

mkdir ./out

cmake .

make clean

cmake --build .

for iterations in 30 100 300
do

mkdir ../out/iteration-${iterations}

./whale ${iterations}

mv ./out/* ../out/iteration-${iterations}/

done