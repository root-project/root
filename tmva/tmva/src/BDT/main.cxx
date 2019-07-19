#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>
#include <array>
#include <chrono>
#include <ctime> // for date
#include <functional> // for std::fucntion

#include "bdt_helpers.h"


#define BDT_KIND 2

#include "unique_bdt.h"
#include "array_bdt.h"
#if BDT_KIND == 1
  #include "bdt.h"
  using namespace shared;
#elif BDT_KIND == 2
  //using namespace unique_bdt;
#elif BDT_KIND == 3
  //using namespace array_bdt;
#endif

//#include "array_bdt.h"
#include "bdt_generator.h"


using json = nlohmann::json;
// ------------------- helper functions -------------------
json read_file(const std::string &filename) {
  std::ifstream i(filename);
  json j;
  i >> j;
  return j;
}


void print(const std::string message){
  std::cout << message << std::endl;
}

void print_json_type(json j){
  std::string my_type = "unknown";
  if (j.type() == json::value_t::null){my_type = "null";}
  else if (j.type() == json::value_t::boolean){my_type = "boolean";}
  else if (j.type() == json::value_t::number_integer){my_type = "number_integer";}
  else if (j.type() == json::value_t::number_unsigned){my_type = "number_unsigned";}
  else if (j.type() == json::value_t::number_float){my_type = "number_float";}
  else if (j.type() == json::value_t::object){my_type = "object";}
  else if (j.type() == json::value_t::array){my_type = "array";}
  else if (j.type() == json::value_t::string){my_type = "string";}
  std::cout << "Type: " << my_type << '\n';
}

/* // params of the children
std::string params[7] = {"depth",
                          "no",
                          "split_condition",
                          "nodeid",
                          "split",
                          "yes",
                          "children"};
*/
void check_json(json &jTree){
  bool node_has_children = (jTree.find("children") != jTree.end());
  if (!node_has_children){
    std::cerr << "Warning: node has no childrens!!!\n";
  }
  if (jTree["yes"] != jTree["children"][0]["nodeid"]){
    std::cerr << "Implementation error for reading nodes.";
  }
}

// ---------------------------  read json  ------------------------------
// TODO
int main() {
  std::cout << "\n\n\n ########## READING MAIN.CXX ##########\n\n";

  std::cout << "\n ***** READ JSON *****\n";
  //json config = read_file("model.json"); // get the model as a json object
  std::string my_config = read_file_string("model.json"); // get model as string
  // Parse the string with all the model
  auto json_model = json::parse(my_config);
  std::cout << "Json read, there are " << json_model.size()
            << " trees in the forest.\n";
  int number_of_trees = json_model.size();
  std::vector<float> event_sample{1., 115., 70., 30.}; // event to test trees



  std::cout << "\n\n ***** Create unique_ptr representation ***** \n";
  unique_bdt::Tree trees[number_of_trees];

  for (int i=0; i<number_of_trees; i++){
    unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
  }
  for (auto& tree : trees){
    std::cout << "unique_ptr pred: " << tree.inference(event_sample) << std::endl;
  }


  std::cout << "\n\n ***** Create array representation ***** \n";
  array_bdt::Tree trees_array[number_of_trees];
  for (int i=0; i<number_of_trees; i++){
    array_bdt::read_nodes_from_tree(json_model[i], trees_array[i]);
  }
  for (auto& tree : trees_array){
    std::cout << "array pred: " << tree.inference(event_sample) << std::endl;
  }

  std::cout << "\n\n ***** Generating text representation of trees ***** \n";
  time_t my_time = time(0);
  std::cout << "current time used as namespace: "<< my_time << std::endl;
  std::string s_trees[number_of_trees];

  for (int i = 0; i<number_of_trees; i++){
    std::filebuf fb;
    std::string filename = "./generated_files/generated_tree_"
                          +std::to_string(i)+".h";
    fb.open (filename, std::ios::out);
    std::ostream os(&fb);
    generate_code_bdt(os, trees[i], i);
    //generate_code_bdt(os, trees[i], i, s_namespace_name);
    std::stringstream ss;
    ss << os.rdbuf();
    s_trees[i] = ss.str();
    fb.close();
  }


  std::cout << "\n ########## END MAIN.CXX ##########\n\n\n";
  return 0;
} // End main
