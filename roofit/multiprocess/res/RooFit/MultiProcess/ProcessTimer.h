/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, Nikhef, zefwolffs@gmail.com
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_MultiProcess_ProcessTimer
#define ROOT_ROOFIT_MultiProcess_ProcessTimer

#include <chrono>
#include <string>
#include <sys/types.h> // pid_t
#include <map>
#include <list>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace std;

namespace RooFit {
namespace MultiProcess {

class ProcessTimer {
public:
   // setup processtimer, initialize some variables
   static void setup(pid_t proc, bool set_begin = true)
   {
      ProcessTimer::process = proc;
      if (set_begin)
         ProcessTimer::begin = chrono::steady_clock::now();
      ProcessTimer::previous_write = ProcessTimer::begin;
   };

   static pid_t get_process() { return ProcessTimer::process; };
   static void set_process(pid_t proc) { ProcessTimer::process = proc; };

   static list<chrono::time_point<chrono::steady_clock>> get_durations(string section_name);

   static void start_timer(string section_name);
   static void end_timer(string section_name);

   static void print_durations(string to_print = "all");
   static void print_timestamps();

   static void write_file();

   static void add_metadata(json data);

   static void set_write_interval(int write_interval);

private:
   // Map of a list of timepoints, even timepoints are start times, uneven timepoints are end times
   using duration_map_t = map<string, list<chrono::time_point<chrono::steady_clock>>>;

   static duration_map_t durations;
   static chrono::time_point<chrono::steady_clock> begin;
   static chrono::time_point<chrono::steady_clock> previous_write;
   static pid_t process;
   static json metadata;
   static int write_interval;
   static int times_written;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_ProcessTimer