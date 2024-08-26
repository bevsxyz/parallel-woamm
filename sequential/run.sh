#!/bin/bash

rm -r ./out
mkdir ./out
mkdir -p build

cd ./build

rm -r ./*

mkdir ./out

cmake ..


cmake --build .

for iterations in 30 100 300
do

mkdir ../out/iteration-${iterations}

./whale ${iterations}

mv ./out/* ../out/iteration-${iterations}/

done
