#!/usr/bin/env bash

clear
make clean
make -j 4

echo "Running Result:"
./crf train

./crf test


#make clean