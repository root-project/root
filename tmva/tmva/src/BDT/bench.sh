#!/bin/bash
rm mybenchmark.exe
make -f makefile_bench.make
./mybenchmark.exe --benchmark_repetitions=10 \
                  --benchmark_format=console \
                  --benchmark_out=data/a.txt \
                  --benchmark_filter=UniqueBdt \
                  --benchmark_report_aggregates_only=true \
                  --benchmark_counters_tabular=true
