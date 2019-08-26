#ifndef __BDT_HELPERS_H_
#define __BDT_HELPERS_H_

#include <numeric>
#include <iostream>
#include <string>
#include <streambuf>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <ios>
#include <array>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>

//
/// sums internal values of vector
template <class T>
inline T vec_sum(std::vector<T> vec)
{
   return std::accumulate(vec.begin(), vec.end(), 0.0);
}

///////////////////////////////////////////////////////
/// Objective functions
/// logistic function
template <class T>
inline T logistic_function(T value)
{
   return 1. / (1. + (1. / std::exp(value)));
}
/// binary logistic
template <class T>
inline bool binary_logistic(T value)
{
   return (logistic_function(value) > 0.5);
}

std::function<bool(float)> get_classification_function(std::string &s_class_func)
{
   std::function<bool(float)> classification_function;
   if (!s_class_func.compare("logistic")) {
      std::cerr << "Not implemented yet" << std::endl;
   } else {
      classification_function = binary_logistic<float>;
   }
   return classification_function;
}
/// END objective functions
/////////////////////////////////////////////////////77//

// --------------- READING FILES ----------------------------
std::string read_file_string(const std::string &filename)
{
   std::ifstream     t(filename);
   std::stringstream buffer;
   buffer << t.rdbuf();
   return buffer.str();
}

/// get linux time
float get_time()
{
   auto unix_timestamp        = std::chrono::seconds(std::time(NULL));
   int  unix_timestamp_x_1000 = std::chrono::milliseconds(unix_timestamp).count();
   return static_cast<float>(unix_timestamp_x_1000);
}

std::string get_time_string()
{
   std::string s_time = std::to_string(get_time());
   s_time.erase(std::remove(s_time.begin(), s_time.end(), '.'), s_time.end());
   return s_time;
}

//////////////////////////////////////////////////////
/// CSV helpers

/// read a line of a "csv file" format
template <class T>
std::vector<T> _read_csv_line(std::string &s_line)
{
   std::vector<T>    vector_line;
   std::stringstream sstream_line(s_line);
   std::string       cell;
   while (getline(sstream_line, cell, ',')) {
      if (!cell.empty()) {
         vector_line.push_back(std::stod(cell)); // Warning: using stod
      } else {
         vector_line.push_back(666);
         std::cerr << "Cell is empty" << std::endl;
      }
   }
   return vector_line;
}

/// reads csv file contaning floats into vector<vector<float>>
template <class T>
std::vector<std::vector<T>> read_csv(std::string &filename)
{
   // std::ifstream fin;
   std::ifstream               file(filename);
   std::vector<std::vector<T>> out;

   std::string cell, line;
   while (file.good()) {
      getline(file, line);
      if (!line.empty()) out.push_back(_read_csv_line<T>(line));
   }
   return out;
}

/// write vector of vectors to csv file
template <class T>
void write_csv(std::string &filename, std::vector<std::vector<T>> values_vec)
{
   std::ofstream fout;
   // opens an existing csv file or creates a new file.
   fout.open(filename, std::ios::out); // | std::ios::app if you want to append"reportcard.csv"
   // Read the input
   // for (auto it = begin (values_vec); it != end (values_vec); ++it) {
   fout.precision(10); // set precision for floats
   int counter = 0;
   for (auto &line : values_vec) {
      counter = 0;
      for (auto &&pred : line) {
         if (counter == 0) {
            fout << pred;
         } else {
            fout << ", " << pred;
         }
         counter++;
      }
      fout << "\n";
   }
   fout.close();
}

/// write vector to csv file
template <class T>
void write_csv(std::string &filename, std::vector<T> values_vec)
{
   std::ofstream fout;
   // opens an existing csv file or creates a new file.
   fout.open(filename, std::ios::out); // | std::ios::app if you want to append"reportcard.csv"
   // Read the input
   for (auto line : values_vec) {
      fout << line;
      fout << "\n";
   }
   fout.close();
}

template <typename T>
std::vector<T> convert_VecMatrix2Vec(std::vector<std::vector<T>> vec_vec)
{
   std::vector<T> out;
   int            rows = vec_vec.size();
   int            cols = vec_vec[0].size();
   out.reserve(rows * cols);
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         out.push_back(vec_vec.at(i).at(j));
      }
   }
   return out;
}
// converts vector<vector> into vector <...>

#endif
// end
