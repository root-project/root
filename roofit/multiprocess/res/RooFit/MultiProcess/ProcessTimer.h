#ifndef ROOT_ROOFIT_MultiProcess_ProcessTimer
#define ROOT_ROOFIT_MultiProcess_ProcessTimer

#include <chrono>
#include <string>
#include <sys/types.h> // pid_t
#include <map>
#include <list>
#include <tuple>
#include <nlohmann/json.hpp>

using namespace std;

class ProcessTimer {

   // Map of a list of timepoints, even timepoints are start times, uneven timepoints are end times
   using duration_map_t = map<string, list<chrono::time_point<chrono::steady_clock>>>;
   static duration_map_t durations;
   static chrono::time_point<chrono::steady_clock> begin;
   static chrono::time_point<chrono::steady_clock> previous_write;
   static pid_t process;
   static nlohmann::json metadata;
   static int write_interval;
   static int times_written;

public:
   static void setup(pid_t proc, bool set_begin = true)
   {
      ProcessTimer::process = proc;
      if (set_begin)
         ProcessTimer::begin = chrono::steady_clock::now();
      ProcessTimer::previous_write = ProcessTimer::begin;
   };

   static pid_t get_process() { return ProcessTimer::process; };

   static void set_process(pid_t proc) { ProcessTimer::process = proc; };

   static void start_timer(string section_name);

   static void end_timer(string section_name);

   static void print_durations(string to_print = "all");

   static void print_timestamps();

   static void write_file();

   static void add_metadata(nlohmann::json data);

   static void set_write_interval(int write_interval);
};

#endif // ROOT_ROOFIT_MultiProcess_ProcessTimer