#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: ./inference.sh <word1> <word2> ..."
  exit 1
fi

echo " Translating: $@"
python3 inference.py "$@"