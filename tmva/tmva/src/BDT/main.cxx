#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>

//#define __MY_UNIQUE__

#ifndef __MY_UNIQUE__
  #include "bdt.h"
  using namespace shared;
#else
  #include "unique_bdt.h"
  using namespace unique;
#endif



using json = nlohmann::json;
// ------------------- helper functions -------------------
json read_file(const std::string &filename) {
  std::ifstream i(filename);
  json j;
  i >> j;
  //std::cout << "Read file: " << j.type();
  return j;
}

std::string read_file_string(const std::string &filename){
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
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

int main() {
  json config = read_file("model.json"); // get the model as a json object
  std::string my_config = read_file_string("model.json"); // get model as string
  //std::cout << "String: " << my_config << std::endl;

  // Parse the string with all the model
  auto json_model = json::parse(my_config);

  std::cout << "\n *************************** \n\n";
  std::cout << "Create " << json_model.size() << " trees\n";
  int number_of_trees = json_model.size();
  Tree trees[number_of_trees];
  for (int i=0; i<number_of_trees; i++){
    //trees[i].nodes =
    read_nodes_from_tree(json_model[i], trees[i]);
    std::cout << "Number of nodes : "
              //<< trees[i].nodes.size()
              << std::endl;
  }

  //double event[4] = {6.,148.,72.,35.};
  double event[4] = {1.,115.,70.,30.};

  for (auto& tree : trees){
    //std::cout  << "There are: " << tree.nodes.size() << " nodes\n";
    std::cout << "Prediction: " << tree.inference(event) << std::endl;
  }

  std::cout << "Count: "
            //<< trees[0].nodes.back()->count
            << std::endl;

  std::cout << "\n***** END *****" << std::endl;
} // End main
