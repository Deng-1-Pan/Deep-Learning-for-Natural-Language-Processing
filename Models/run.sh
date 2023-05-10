#!/bin/bash

python Models/BERT.py | tee log/a.log

python Models/distilbert.py | tee log/b.log

python Models/xlm-roberta.py | tee log/c.log

python Models/Roberta.py | tee log/d.log