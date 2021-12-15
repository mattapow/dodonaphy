#!/bin/bash

cd ./analysis
i=1
while [ $i -le 2 ]
do
    cd ./DS$i
    mkdir RAxML
    raxmlHPC -m GTRGAMMA --JC69 -s data/DS.fasta -n jc69 -T 2 -p $RANDOM 
    mv *.jc69 RAxML
    cd ..
    ((i++))
done