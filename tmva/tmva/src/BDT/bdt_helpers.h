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

double average(std::vector<double> vec){
  return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

std::string read_file_string(const std::string &filename){
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

size_t count_columns_in_line(const std::string &line){
  size_t number_of_columns = std::count(line.begin(), line.end(), ',');
  return number_of_columns;
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ','))
    {
        result.push_back(cell);
        std::cout << cell;
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}

/// read a line of a "csv file" format
std::vector<double> read_csv_line(std::string & s_line){
  std::vector<double> vector_line;
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

/// reads csv file contaning doubles into vector<vector<double>>
std::vector<std::vector<double>> read_csv(std::string & filename) {
  //std::ifstream fin;
  std::ifstream       file(filename);
  std::vector<std::vector<double>> out;
  std::string cell, line;
  //std::stringstream sstream_line;
  while(file.good()){
    getline(file, line);
    if (!line.empty())
      out.push_back(read_csv_line(line));
  }
  return out;
}

std::vector<std::vector<double>> read_csv_old(std::string & filename) {
  std::fstream fin;
  fin.open(filename, std::ios::in); //"reportcard.csv"
  std::vector<std::vector<double>> out;
  std::string line, word, temp;

  //std::cout << read_file_string(filename) << std::endl;

  int number_of_features=0;
  int count = 0;
  while (fin >> temp) {
      std::getline(fin, line); // read one line and put it in a string
      std::stringstream s(line);
      std::cout << line << std::endl;
      if (count == 0){number_of_features=count_columns_in_line(line);}
      std::cout << number_of_features << std::endl;
      std::vector<double> event;
      while (std::getline(s, word, ',')) { //decompose line in words
          //event[counter] = std::stod(word);
          std::cout << word << std::endl;
          event.push_back(std::stod(word));
      }
      if (event.size() != number_of_features){
        std::cerr << "#columns of csv file not constant!\n";
      }
      out.push_back(event);
  }
  fin.close();
  return out;
}





void write_csv(std::string &filename, std::vector<std::vector<double>> values_vec)
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
