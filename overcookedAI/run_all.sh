#!/bin/bash
cd pestel
make
cd ..
python3 run.py 0
python3 run.py 1
python3 run.py 2
python3 run.py 3
python3 run.py 4
python3 run.py 6
python3 run.py t0
python3 run.py t1
python3 genPlot.py
