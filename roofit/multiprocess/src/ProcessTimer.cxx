#include "RooFit/MultiProcess/ProcessTimer.h"

#include <thread>
#include <iostream>
#include <thread>
#include <fstream>
#include <iomanip> // setw

void ProcessTimer::start_timer(string section_name)
{
   auto it = ProcessTimer::durations.find(section_name);
   if (it == ProcessTimer::durations.end()) {
      // Key does not exist in map yet, start the first timer of this section
      ProcessTimer::durations.insert({section_name, {chrono::steady_clock::now()}});
   } else if (it->second.size() % 2 != 0) {
      // All even indices contain start times, if size of list is currently even we can not start a new timer
      throw ::invalid_argument("Section name " + section_name +
                               " timer has already started, and was not stopped before calling `start_timer`");
   } else {
      // Add start time to list
      it->second.push_back(chrono::steady_clock::now());
   }
}

void ProcessTimer::end_timer(string section_name)
{
   auto it = ProcessTimer::durations.find(section_name);
   if (it == ProcessTimer::durations.end()) {
      // Key does not exist in map yet
      throw ::invalid_argument("Section name " + section_name + " timer was never started!");
   } else if (it->second.size() % 2 == 0) {
      // All odd indices contain end times, if size of list is currently odd we can not start a new timer
      throw ::invalid_argument("Section name " + section_name +
                               " timer does exist, but was not started before calling `end_timer`");
   } else {
      // Add end time to list
      it->second.push_back(chrono::steady_clock::now());
   }

   // Write to file intermittently if interval is reached and write_interval is set
   if (write_interval && (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - previous_write).count() >
                          write_interval)) {
      previous_write = chrono::steady_clock::now();
      ProcessTimer::write_file();
      times_written++;
   }
}

void ProcessTimer::print_durations(string to_print)
{
   cout << "On PID: " << ProcessTimer::process << endl << "====================" << endl << endl;
   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = std::move(durations_element);
      if (to_print != "all" && sec_name != to_print)
         continue; // continue if only asked for specific section

      int i = 0;
      long total_duration = 0;
      cout << "Section name " << sec_name << ":" << endl;
      for (auto it = duration_list.begin(); it != duration_list.end(); ++it) {
         long duration = chrono::duration_cast<chrono::milliseconds>(*++it - *it).count();
         cout << "Duration " << i << ": " << duration << "ms +" << endl;
         total_duration += duration;
         i++;
      }
      cout << "--------------------" << endl << "Total: " << total_duration << "ms" << endl << endl;
   }
}

void ProcessTimer::print_timestamps()
{
   cout << "On PID: " << ProcessTimer::process << endl;
   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = std::move(durations_element);
      int i = 0;
      cout << "Section name " << sec_name << ":" << endl;
      for (auto it = duration_list.begin(); it != duration_list.end(); ++it) {
         long duration_since_begin_start =
            chrono::duration_cast<chrono::milliseconds>(*it - ProcessTimer::begin).count();

         long duration_since_begin_end =
            chrono::duration_cast<chrono::milliseconds>(*++it - ProcessTimer::begin).count();

         cout << "Duration " << i << ": " << duration_since_begin_start << "ms-->" << duration_since_begin_end << "ms"
              << endl;
         i++;
      }
   }
}

void ProcessTimer::write_file()
{
   nlohmann::json j;
   j["metadata"] = metadata;
   std::ofstream file("p_" + to_string((long)ProcessTimer::get_process()) + ".json." + to_string(times_written),
                      ios::app);
   list<long> durations_since_begin;

   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = std::move(durations_element);
      durations_since_begin.clear();
      for (auto const &timestamp : duration_list) {
         durations_since_begin.push_back(
            chrono::duration_cast<chrono::microseconds>(timestamp - ProcessTimer::begin).count());
      }
      j[sec_name] = durations_since_begin;
   }
   file << std::setw(4) << j;
   file.close();
}

void ProcessTimer::add_metadata(nlohmann::json data)
{
   if (write_interval) {
      nlohmann::json j, meta;
      meta.push_back(std::move(data));
      j["metadata"] = meta;
      std::ofstream file("p_" + to_string((long)ProcessTimer::get_process()) + ".json", ios::app);
      file << std::setw(4) << j;
   } else {
      metadata.push_back(std::move(data));
   }
}

void ProcessTimer::set_write_interval(int write_int)
{
   write_interval = write_int;
   if (write_interval) {
      nlohmann::json j, meta;
      meta["write_interval"] = true;
      j["metadata"] = meta;
      std::ofstream file("p_" + to_string((long)ProcessTimer::get_process()) + ".json", ios::app);
      file << std::setw(4) << j;
   }
}

// Initialize static members
ProcessTimer::duration_map_t ProcessTimer::durations;
chrono::time_point<chrono::steady_clock> ProcessTimer::begin = chrono::steady_clock::now();
chrono::time_point<chrono::steady_clock> ProcessTimer::previous_write = chrono::steady_clock::now();
pid_t ProcessTimer::process = 0;
nlohmann::json ProcessTimer::metadata;
int ProcessTimer::write_interval = 0;
int ProcessTimer::times_written = 0;