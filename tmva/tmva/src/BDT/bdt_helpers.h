#include <numeric>
# include <iostream>
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

//#include "TInterpreter.h" // for gInterpreter

//
/// sums internal values of vector
template <class T>
T vec_sum(std::vector<T> vec){
  return std::accumulate(vec.begin(), vec.end(), 0.0);
}

/// logistic function
template <class T>
T logistic_function(T value){
  return 1. / (1. + (1./std::exp(value)));
}

/// binary logistic
template <class T>
bool binary_logistic(T value){
  return (logistic_function(value)>0.5)? 1: 0;
}


std::function<bool (float)> get_classification_function(std::string & s_class_func){
  std::function<bool(float)> classification_function;
   if (!s_class_func.compare("logistic")){
     std::cerr << "Not implemented yet" << std::endl;
   }
   else {
     classification_function = binary_logistic<float>;
   }
   return classification_function;
}


// --------------- READING FILES ----------------------------
std::string read_file_string(const std::string &filename){
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

/// read a line of a "csv file" format
std::vector<float> read_csv_line(std::string & s_line){
  std::vector<float> vector_line;
  std::stringstream sstream_line(s_line);
  std::string cell;
  while (getline(sstream_line, cell, ',')) {
    if (!cell.empty()){
      vector_line.push_back(std::stod(cell));
    }
    else{
      vector_line.push_back(666);
      std::cerr << "Cell is empty"<< std::endl;
    }
  }
  return vector_line;
}

/// reads csv file contaning floats into vector<vector<float>>
std::vector<std::vector<float>> read_csv(std::string & filename) {
  //std::ifstream fin;
  std::ifstream       file(filename);
  std::vector<std::vector<float>> out;
  std::string cell, line;
  //std::stringstream sstream_line;
  while(file.good()){
    getline(file, line);
    if (!line.empty())
      out.push_back(read_csv_line(line));
  }
  return out;
}

template <class T>
void write_csv(std::string &filename, std::vector<std::vector<T>> values_vec)
{
  std::ofstream fout;
  // opens an existing csv file or creates a new file.
  fout.open(filename, std::ios::out); // | std::ios::app if you want to append"reportcard.csv"
  // Read the input
  //for (auto it = begin (values_vec); it != end (values_vec); ++it) {
  int counter = 0;
  for (auto & line : values_vec) {
    counter = 0;
    for (auto &&pred : line){
      if (counter == 0){
        fout << pred;
      }
      else{
        fout << ", " << pred;
      }
      counter++;
    }
    fout << "\n";
  }
  fout.close();
}
