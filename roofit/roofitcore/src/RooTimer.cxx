#include "RooTimer.h"

double RooTimer::timing_s() {
  return _timing_s;
}

void RooTimer::set_timing_s(double timing_s) {
  _timing_s = timing_s;
}


RooInstantTimer::RooInstantTimer() {
  _timing_begin = std::chrono::high_resolution_clock::now();
}

void RooInstantTimer::stop() {
  _timing_end = std::chrono::high_resolution_clock::now();
  set_timing_s(std::chrono::duration_cast<std::chrono::nanoseconds>(_timing_end - _timing_begin).count() / 1.e9);
}


RooInstantCPUTimer::RooInstantCPUTimer() {
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_timing_begin);
}

void RooInstantCPUTimer::stop() {
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_timing_end);
  set_timing_s((_timing_end.tv_nsec - _timing_begin.tv_nsec) / 1.e9);
}
