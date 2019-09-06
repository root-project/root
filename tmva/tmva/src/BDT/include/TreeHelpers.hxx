/**********************************************************************************
 * Project: ROOT - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Author: Luca Zampieri (luca.zampieri@alumni.epfl.ch)  01/09/2019               *
 *                                                                                *
 * Copyright (c) 2019:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef __TREE_HELPERS_H_
#define __TREE_HELPERS_H_

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

////////////////////////////////////////////////////////////////////////////////
/// logistic function
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] value score to be transformed in probability
/// \param[out] logistic function of the input
template <typename T>
inline T logistic_function(T value)
{
   return 1. / (1. + (1. / std::exp(value)));
}

////////////////////////////////////////////////////////////////////////////////
/// binary logistic function
///
/// \tparam T type, usually floating point type (float, double, long double)
template <typename T>
inline bool binary_logistic(T value)
{
   return (logistic_function(value) > 0.5);
}

////////////////////////////////////////////////////////////////////////////////
/// Identity function
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] value score to be processed
/// \param[out] input
template <typename T>
inline T identity_function(T value)
{
   return value;
}

template <typename T>
std::function<T(T)> get_classification_function(const std::string &s_class_func)
{
   std::function<T(T)> classification_function;

   if (std::string("logistic").compare(s_class_func) == 0) {
      classification_function = binary_logistic<T>;
   } else if (std::string("identity").compare(s_class_func) == 0) {
      classification_function = identity_function<T>;
   } else {
      std::cerr << "Function " << s_class_func << "is not implemented yet" << std::endl;
      classification_function = binary_logistic<T>;
   }
   return classification_function;
}

////////////////////////////////////////////////////////////////////////////////
/// Reads file into string
///
/// \param[in] filename
/// \param[out] string containing the content of the file
std::string read_file_string(const std::string &filename)
{
   std::ifstream     t(filename);
   std::stringstream buffer;
   buffer << t.rdbuf();
   return buffer.str();
}

////////////////////////////////////////////////////////////////////////////////
/// param[out] linux time in milliseconds
double get_time()
{
   auto unix_timestamp        = std::chrono::seconds(std::time(NULL));
   int  unix_timestamp_x_1000 = std::chrono::milliseconds(unix_timestamp).count();
   return static_cast<double>(unix_timestamp_x_1000);
}

////////////////////////////////////////////////////////////////////////////////
/// \param[out] time processed and ready to be used as a namespace
std::string get_time_string()
{
   std::string s_time = std::to_string(get_time());
   s_time.erase(std::remove(s_time.begin(), s_time.end(), '.'), s_time.end());
   return s_time;
}

////////////////////////////////////////////////////////////////////////////////
/// read a line of a csv file format
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] s_line string contain a csv line
/// \param[out] line written as a vector
template <typename T>
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

////////////////////////////////////////////////////////////////////////////////
/// reads csv file into vector<vector<T>>
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] filename of the csv file
/// \param[out] content written in a vector of vectors
template <typename T>
std::vector<std::vector<T>> read_csv(const std::string &filename)
{
   std::ifstream               file(filename);
   std::vector<std::vector<T>> out;
   std::string                 cell, line;
   while (file.good()) {
      getline(file, line);
      if (!line.empty()) out.push_back(_read_csv_line<T>(line));
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// write vector of vectors to csv file
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] filename where to write
/// \param[in] data to write to file
template <typename T>
void write_csv(const std::string &filename, std::vector<std::vector<T>> values_vec)
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

////////////////////////////////////////////////////////////////////////////////
/// write vector to csv file
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] filename where to write
/// \param[in] data to write to file
template <typename T>
void write_csv(const std::string &filename, std::vector<T> values_vec)
{
   std::ofstream fout;
   // opens an existing csv file or creates a new file.
   fout.open(filename, std::ios::out); // | std::ios::app if you want to append"reportcard.csv"
   fout.precision(10);                 // set precision for floats
   // Read the input
   for (auto line : values_vec) {
      fout << line;
      // std::cout << line << std::endl;
      fout << "\n";
   }
   fout.close();
}

////////////////////////////////////////////////////////////////////////////////
/// convert vector of vectors into a single vector
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] vec_vec data to convert
/// \param[in] converted vector
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

////////////////////////////////////////////////////////////////////////////////
/// Data structure for tests and benchmarking
///
/// \tparam T type, usually floating point type (float, double, long double)
///
/// Contains all data to make different kind of predictions
template <typename T>
struct DataStruct {
   const std::vector<std::vector<T>>    events_vec_vec;           ///< events
   const std::vector<T>                 events_vector;            ///< events
   const T *                            events_pointer = nullptr; ///< events
   const std::vector<std::vector<bool>> groundtruth;              ///< "real" predictions
   std::vector<T>                       scores;
   std::vector<bool>                    preds;
   const int                            rows, cols;

   DataStruct(const std::string &events_file, const std::string &preds_file)
      : events_vec_vec(read_csv<T>(events_file)), events_vector(convert_VecMatrix2Vec<T>(events_vec_vec)),
        events_pointer(events_vector.data()), groundtruth(read_csv<bool>(preds_file)), rows(events_vec_vec.size()),
        cols(events_vec_vec[0].size())
   {
      if (rows < 1) {
         std::cerr << "No events in event vector!!! (usually bad) \n";
      }
      preds.resize(rows);
      scores.resize(rows);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Data structure for tests and benchmarking Regression Forests
///
/// \tparam T type, usually floating point type (float, double, long double)
///
/// Contains all data to make different kind of predictions
template <typename T>
struct DataStructRegression {
   const std::vector<std::vector<T>> events_vec_vec;           ///< events
   const std::vector<T>              events_vector;            ///< events
   const T *                         events_pointer = nullptr; ///< events
   const std::vector<std::vector<T>> python_preds;             ///< "real" predictions
   const std::vector<std::vector<T>> python_scores;            ///< "real" predictions
   std::vector<T>                    scores;
   const int                         rows, cols;

   DataStructRegression(const std::string &events_file, const std::string &preds_file, const std::string &scores_file)
      : events_vec_vec(read_csv<T>(events_file)), events_vector(convert_VecMatrix2Vec<T>(events_vec_vec)),
        events_pointer(events_vector.data()), python_preds(read_csv<T>(preds_file)),
        python_scores(read_csv<T>(scores_file)), rows(events_vec_vec.size()), cols(events_vec_vec[0].size())
   {
      if (rows < 1) {
         std::cerr << "No events in event vector!!! (usually bad) \n";
      }
      if (python_scores.size() < 1) {
         std::cerr << "No events in python_scores!!! (usually bad) \n";
      }
      if (python_preds.size() < 1) {
         std::cerr << "No events in python_preds!!! (usually bad) \n";
      }
      // preds.resize(rows);
      scores.resize(rows);
   }
};

#endif
// end
