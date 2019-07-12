#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>

#include "bdt.h"

using json = nlohmann::json;

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



void check_params(json j, int max_counter, int counter=0){

  std::string params[7] = {"depth",
                            "no",
                            "split_condition",
                            "nodeid",
                            "split",
                            "yes",
                            "children"};

  for (size_t i = 0; i<j.size(); i++){
    for (auto &mess : params){
      if ((mess == "children") && (counter<max_counter)){
        counter++;
        check_params(j[i][mess],max_counter, counter);
      }
      else{
        std::cout << mess <<": " <<j[0]["children"][i][mess] << std::endl;
      }
    }
  }
}

void check_params2(json j, int max_counter, int counter=0){

  std::string params[7] = {"depth",
                            "no",
                            "split_condition",
                            "nodeid",
                            "split",
                            "yes",
                            "children"};

  for (size_t i = 0; i<j.size(); i++){
    for (auto &mess : params){
    //for (auto &mess : j[i]){
      if (j[i].count("leaf") > 0){
        std::cout<< "FOUND\n";
      }
      else if ((mess == "children") && (counter<max_counter)){
        counter++;
        check_params2(j[i][mess],max_counter, counter);
      }
      else{
        std::cout << mess <<": " <<j[i]["children"][i][mess] << std::endl;
      }
    }
  }
}
// -----------------------------------------------------------------------------

void write_node_members(json &jTree, Node* tmp_node){
  tmp_node->split_value = jTree["split_condition"];
  tmp_node->node_id = jTree["nodeid"];
  tmp_node->child_id_true = jTree["yes"];
  tmp_node->child_id_false = jTree["no"];

  std::string tmp_str = jTree["split"].get<std::string>();
  tmp_str.erase(tmp_str.begin(), tmp_str.begin()+1); // remove initial "f"
  tmp_node->split_variable = std::stoi(tmp_str);
}
/// Need a nlohmann::json object from an xgboost saved format
//AbstractNode* _read_nodes(json jTree, std::vector<AbstractNode> *nodes_vector){
void check_json(json &jTree){
  bool node_has_children = (jTree.find("children") != jTree.end());
  if (!node_has_children){
    std::cerr << "Warning: node has no childrens!!!\n";
  }
  if (jTree["yes"] != jTree["children"][0]["nodeid"]){
    std::cerr << "Implementation error for reading nodes.";
  }
}


Node* _read_nodes(json jTree, Tree &tree){
  std::cout << "create\n";
  bool is_leaf_node = (
    (jTree["children"][0].find("leaf") != jTree["children"][0].end())
    && (jTree["children"][0].find("nodeid") != jTree["children"][0].end())
  );

  Node* tmp_node;
  tmp_node = new Node();
  write_node_members(jTree, tmp_node);
  if (is_leaf_node){
    tmp_node->leaf_true = jTree["children"][0]["leaf"];
    tmp_node->leaf_false = jTree["children"][1]["leaf"];
    tmp_node->is_leaf_node=1;
    tree.nodes.push_back(tmp_node);
    //return tmp_node;
  }
  else {
    tmp_node->child_true = _read_nodes(jTree["children"][0], tree);
    tmp_node->child_false = _read_nodes(jTree["children"][1], tree);
    tree.nodes.push_back(tmp_node);
  }
  return tmp_node;

}

//std::vector<AbstractNode>
void read_nodes_from_tree(json jTree,Tree &tree){
  //std::vector<AbstractNode> nodes;
  _read_nodes(jTree, tree);
  //return nodes;
}

// */


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
              << trees[i].nodes.size()
              << std::endl;
  }

  double event[4] = {0,1,2,3};

  std::cout << trees[0].inference(event) << std::endl;
  std::cout << trees[0].nodes.size() << std::endl;
  /*
  for (auto &node : test.nodes.leaf_nodes){
    std::cout << node.split_value << std::endl;
    std::cout << node.inference(event) << std::endl;
  }*/
/*
  for (auto &node : test.nodes.normal_nodes){
    std::cout << node.split_value << std::endl;
    std::cout << node.inference(event) << std::endl;
  }
*/


/*
  Forest my_forest;
  for (auto &tree : trees){
    my_forest.trees.push_back(tree);
  }
*/
  //my_forest.trees = trees;
  //std::vector<AbstractNode> nodes = read_nodes_from_tree(json_model[0]);
  //std::cout << "Number of nodes : " << nodes.size() << std::endl;
  //tree_1.nodes = nodes;

  // template faillure
  //Node2<double> leaf;




  std::cout << "\n***** END *****" << std::endl;
} // End main
