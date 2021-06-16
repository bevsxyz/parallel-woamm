#!/bin/bash

make clean

make

rm -r ./out

mkdir ./out
for rng in philox4 MTGP32 MRG32k3a
do
    mkdir ./out/${rng}
    for iterations in 30 100 300
    do
        mkdir ./out/${rng}/iterations-${iterations}
        for block in 1 2 4 6
        do
            mkdir ./out/${rng}/iterations-${iterations}/blocks-${block}
            
            ./${rng} ${block} ${iterations}
        done
    done
done

