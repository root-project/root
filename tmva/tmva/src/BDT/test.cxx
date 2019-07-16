
# include <iostream>
#include <string>
#include <streambuf>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <map>
#include <vector>
#include "TInterpreter.h" // for gInterpreter
//# include "generated_code.h"


using namespace std;

std::string read_file_string(const std::string &filename){
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

double testJitting(double event[], int tree_index){
   std::string filename = "generated_files/generated_tree_" + std::to_string(tree_index) + ".h";
   string tojit = read_file_string(filename);
   gInterpreter->Declare(tojit.c_str());

   std::string func_ref_name = "&generated_tree_"+std::to_string(tree_index);

   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   double (*func)(double*) = reinterpret_cast<double(*)(double*)>(ptr);
   return func(event);
}




int main() {
  double event[4] = {1.,115.,70.,30.};
  int trees_number = 4;
  //std::cout << "Generated prediction: \n";
  // Directly read the function from the file
  //std::cout << generate_tree_1(event) << std::endl;
  // call while jitting
  std::cout << "Jitted: \n";

  for (int i=0; i<trees_number; i++){
    std::cout << testJitting(event, i) << std::endl;
  }
  return 0;
}
