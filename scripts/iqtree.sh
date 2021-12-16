#!/bin/bash

i=1
while [ $i -le 8 ]
do
    cd ./analysis/DS$i
    mkdir iqtree
    iqtree -s data/DS.nex -m JC --prefix ./iqtree/DS --nt AUTO
    cd ../..
    ((i++))
done