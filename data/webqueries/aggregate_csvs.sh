#!/bin/bash

for i in $( ls ); do
    grep "" $i/*.csv >> train.csv
done

sed -i 's/:/,/' train.csv
sed -i 's/.csv//' train.csv
sed -i -E 's/^query_[0-9]+\///' train.csv
