#!/bin/bash

rm -r ./out
mkdir ./out

cd ./build

rm -r ./*

mkdir ./out

cmake3 ..


cmake3 --build .

for iterations in 30 100 300
do

mkdir ../out/iteration-${iterations}

./whale ${iterations}

mv ./out/* ../out/iteration-${iterations}/

done
