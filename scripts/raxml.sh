#!/bin/bash

cd ./analysis
path_raxml=RAxML
i=1
while [ $i -le 1 ]
do
    cd ./DS$i    
    mkdir $path_raxml
    raxmlHPC -m GTRGAMMA --JC69 -s data/DS.fasta -n jc69 -T 2 -p $RANDOM 
    mv *.jc69 $path_raxml
    cd ..
    ((i++))
done