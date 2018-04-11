#!/bin/bash

# This bash script iterates through all .xml files in the metadata/ directory,
# strips all text except the contents of the <before>, <after>, <ptitle> and
# <alt> tags, using sed.

for file in ./*.xml
do
    sed -i -n 8,11p $file
    sed -i -E 's/<\/?[a-z]+>//g' $file
done
