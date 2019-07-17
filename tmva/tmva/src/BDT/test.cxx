
# include <iostream>
#include <string>
#include <streambuf>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>


#include <map>
#include <vector>
#include <functional>
#include "bdt_helpers.h"
#include "TInterpreter.h" // for gInterpreter
//# include "generated_code.h"


using namespace std;



double testJitting(std::vector<double> event, int tree_index){
   std::string filename = "generated_files/generated_tree_" + std::to_string(tree_index) + ".h";
   string tojit = read_file_string(filename);
   gInterpreter->Declare(tojit.c_str());

   std::string func_ref_name = "&generated_tree_"+std::to_string(tree_index);

   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   double (*func)(std::vector<double>) = reinterpret_cast<double(*)(std::vector<double>)>(ptr);
   return func(event);
}

std::function<double (std::vector<double>)> jit_function_reader(std::vector<double> event, int tree_index){
   std::string filename = "generated_files/generated_tree_" + std::to_string(tree_index) + ".h";
   string tojit = read_file_string(filename);
   gInterpreter->Declare(tojit.c_str());

   std::string func_ref_name = "&generated_tree_"+std::to_string(tree_index);

   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   double (*func)(std::vector<double>) = reinterpret_cast<double(*)(std::vector<double>)>(ptr);
   std::function<double (std::vector<double>)> fWrapped{func};
   return fWrapped;
}







int main() {
  std::string data_folder = "./data_files/";
  //double event[4] = {1.,115.,70.,30.};
  std::vector<double> event{1.,115.,70.,30.};
  std::string events_file = data_folder+"events.csv";
  std::vector<std::vector<double>> events_vector = read_csv(events_file);
  std::string test_file = data_folder+"test.csv";
  write_csv(test_file, events_vector);
  int trees_number = 4;
  //std::cout << "Generated prediction: \n";
  // Directly read the function from the file
  //std::cout << generate_tree_1(event) << std::endl;
  // call while jitting
  std::cout << "Jitted: \n";
  //std::cout << events_vector[0].size() << std::endl;
  std::string write_file = data_folder+"cpp_scores.csv";
  //for (auto & event : events_vector)
  //for (auto it = begin (vector); it != end (vector); ++it) {
  //  it->doSomething ();}

  std::vector<std::vector<double>> out;

  std::vector<double> out_tmp;
  double prediction = 0;
  std::function<double (std::vector<double>)> func;
  std::vector<std::function<double (std::vector<double>)>> function_vector;
  for (int i=0; i<trees_number; i++){
    //func = jit_function_reader(event, i);
    function_vector.push_back(jit_function_reader(event, i));
  }
  //for (auto it = events_vector.begin() ; it != events_vector.end(); ++it){
  int counter = 0;
  for (auto &event : events_vector){
    //for (auto it2 = event.begin() ; it2 != event.end(); ++it2){
    //for (auto &
    //for (auto it = function_vector.begin() ; it != function_vector.end(); ++it){
    out_tmp.clear();
    for (auto &func : function_vector){
      prediction = func(event);
      //std::cout << func(event) << std::endl;
      out_tmp.push_back(prediction);
    }
    out.push_back(out_tmp);
      //std::cout << " " << *it2;
    //for (auto it2 = event.begin() ; it2 != event.end(); ++it2){
    //  std::cout << *it2 << " ";
    //}
    //std::cout << counter << std::endl;
    counter++;
    //}
  }
  /*
  for (auto &event1 : events_vector){
    std::cout << event1[0];
    for (auto it = begin(event1); it != event1(vector); ++it) {
      //prediction =function_vector[i](event1);
      //std::cout << prediction << std::endl;
    out_tmp.push_back(prediction);
  }*/
  //out_tmp.push_back(prediction);
    //std::cout << generated_tree_1(event);
/*
  out.push_back(out_tmp);
  out.push_back(out_tmp);
  out.push_back(out_tmp);
  out.push_back(out_tmp);
  out.push_back(out_tmp);
*/
  write_csv(write_file, out);

  return 0;
}
