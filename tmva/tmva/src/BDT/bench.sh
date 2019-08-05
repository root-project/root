#!/bin/bash
make -f makefile_bench.make
./mybenchmark.exe --benchmark_repetitions=5 \
                  --benchmark_format=console \
                  --benchmark_out=benchs/a.txt \
                  --benchmark_filter=Bdt \
                  --benchmark_report_aggregates_only=true \
                  --benchmark_counters_tabular=true
