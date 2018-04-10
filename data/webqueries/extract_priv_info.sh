#!/bin/bash

for file in ./*.xml
do
    sed -i -n 8,11p $file
    sed -i -E 's/<\/?[a-z]+>//g' $file
done
