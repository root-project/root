#!/bin/bash
echo " ***** Running python ***** "
python train.py


echo "***** Running check_preds.py *****"
python check_preds.py

echo "***** Benchmarking *****"
make -f makefile_bench.make
./mybenchmark.exe --benchmark_repetitions=1 \
                  --benchmark_filter=Bdt \
                  --benchmark_format=console \
                  --benchmark_report_aggregates_only=true
