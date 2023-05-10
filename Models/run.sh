#!/bin/bash

python BERT.py | tee a.log

python distilbert.py | tee b.log

python xlm-roberta.py | tee c.log

python Roberta.py | tee d.log