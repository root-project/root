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

#include "RooFit/MultiProcess/ProcessTimer.h"

#include <iostream>
#include <fstream>
#include <iomanip> // setw

using std::list, std::string, std::invalid_argument, std::to_string, std::ios;
namespace chrono = std::chrono; // alias

namespace RooFit {
namespace MultiProcess {

/** \class ProcessTimer
 *
 * \brief Can be used to generate timings of multiple processes simultaneously and output logs
 *
 * This static class records timings of multiple processes simultaneously and allows for these
 * timings to be written out in json format, one file for each process. Multiple overlapping
 * sections can be timed independently on the same process. It also allows for the timings
 * to be written out to json logfiles in a specified interval, for example every half hour.
 *
 * Note that this class logs timings in milliseconds.
 */

list<chrono::time_point<chrono::steady_clock>> ProcessTimer::get_durations(string to_return)
{
   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = durations_element;
      if (sec_name != to_return) {
         continue;
      } else {
         return duration_list;
      }
   }
   throw ::invalid_argument("section name " + to_return +
                            " not found in timer map, so it cannot"
                            " be retrieved");
}

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
   std::cout << "On PID: " << ProcessTimer::process << std::endl << "====================" << std::endl << std::endl;
   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = durations_element;
      if (to_print != "all" && sec_name != to_print)
         continue; // continue if only asked for specific section

      int i = 0;
      long total_duration = 0;
      std::cout << "Section name " << sec_name << ":" << std::endl;
      for (auto it = duration_list.begin(); it != duration_list.end(); ++it) {
         long duration = chrono::duration_cast<chrono::milliseconds>(*std::next(it) - *it).count();
         std::cout << "Duration " << i << ": " << duration << "ms +" << std::endl;
         total_duration += duration;
         i++;
      }
      std::cout << "--------------------" << std::endl << "Total: " << total_duration << "ms" << std::endl << std::endl;
   }
}

void ProcessTimer::print_timestamps()
{
   std::cout << "On PID: " << ProcessTimer::process << std::endl;
   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = durations_element;
      int i = 0;
      std::cout << "Section name " << sec_name << ":" << std::endl;
      for (auto it = duration_list.begin(); it != duration_list.end(); ++it) {
         long duration_since_begin_start =
            chrono::duration_cast<chrono::milliseconds>(*it - ProcessTimer::begin).count();

         long duration_since_begin_end =
            chrono::duration_cast<chrono::milliseconds>(*std::next(it) - ProcessTimer::begin).count();

         std::cout << "Duration " << i << ": " << duration_since_begin_start << "ms-->" << duration_since_begin_end << "ms"
              << std::endl;
         i++;
      }
   }
}

void ProcessTimer::write_file()
{
   json j;
   j["metadata"] = metadata;
   std::ofstream file("p_" + to_string((long)ProcessTimer::get_process()) + ".json." + to_string(times_written),
                      ios::app);
   list<long> durations_since_begin;

   ProcessTimer::duration_map_t::key_type sec_name;
   ProcessTimer::duration_map_t::mapped_type duration_list;
   for (auto const &durations_element : ProcessTimer::durations) {
      std::tie(sec_name, duration_list) = durations_element;
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

void ProcessTimer::add_metadata(json data)
{
   if (write_interval) {
      json j;
      json meta;
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
      json j;
      json meta;
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
json ProcessTimer::metadata;
int ProcessTimer::write_interval = 0;
int ProcessTimer::times_written = 0;

} // namespace MultiProcess
} // namespace RooFit
