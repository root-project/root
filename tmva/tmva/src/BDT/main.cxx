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

  for (int i = 0; i<j.size(); i++){
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

  for (int i = 0; i<j.size(); i++){
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



AbstractNode* _read_nodes(json jTree, std::vector<AbstractNode> *nodes_vector){
  bool node_has_children = (jTree.find("children") != jTree.end());
  bool is_leaf_node = (
    (jTree["children"][0].find("leaf") != jTree["children"][0].end())
    && (jTree["children"][0].find("nodeid") != jTree["children"][0].end())
  );

  if (node_has_children){
    if (is_leaf_node){
      LeafNode tmp_leaf_node;
      tmp_leaf_node.threshold = jTree["split_condition"];
      tmp_leaf_node.split_variable = jTree["split"];
      tmp_leaf_node.node_id = jTree["nodeid"];
      if (jTree["yes"] == jTree["children"][0]["nodeid"]){
        tmp_leaf_node.leaf_true = jTree["children"][0]["leaf"];
        tmp_leaf_node.leaf_false = jTree["children"][1]["leaf"];
      }
      else {
        std::cerr << "Implementation error for reading leaf nodes.";
      }
      nodes_vector->push_back(tmp_leaf_node);
    }
    else {
      Node tmp_normal_node;
      tmp_normal_node.threshold = jTree["split_condition"];
      tmp_normal_node.split_variable = jTree["split"];
      tmp_normal_node.node_id = jTree["nodeid"];
      tmp_normal_node.child_id_true = jTree["yes"];
      tmp_normal_node.child_id_false = jTree["no"];
      // Read childs
      if (jTree["yes"] == jTree["children"][0]["nodeid"]){
        tmp_normal_node.child_true = _read_nodes(jTree["children"][0], nodes_vector);
        tmp_normal_node.child_false = _read_nodes(jTree["children"][1], nodes_vector);
      }
      else {
        std::cerr << "Implementation error for reading normal nodes.";
      }
      nodes_vector->push_back(tmp_normal_node);

    }
  }
  else {
    std::cerr << "Warning: node has no childrens!!!\n";
  }
}

std::vector<AbstractNode> read_nodes_from_tree(json jTree){
  std::vector<AbstractNode> nodes;
  _read_nodes(jTree, &nodes);
  return nodes;
}

// */


int main() {
  json config = read_file("model.json"); // get the model as a json object
  std::string my_config = read_file_string("model.json"); // get model as string
  //std::cout << "String: " << my_config << std::endl;

  // Parse the string with all the model
  auto j3 = json::parse(my_config);
  std::cout << "json stringed" << j3.dump() << std::endl;
  std::cout << "Size: " <<  j3.size() << std::endl;

  auto my_type = j3.type();
  //std::cout << my_type.get<std::string>();

  int count = 0;
  for (auto &tree : j3){
    std::cout << tree << "\n\n";
    count++;
  }
  std::cout << "\n *************************** \n\n";

  //print(j3[0].size().dump());
  std::cout << j3[0].size();
  std::cout << j3[0]["children"];

  print_json_type(j3);
  print_json_type(j3[0]);
  print_json_type(j3[0]["children"]);
  std::cout << j3[0]["children"].size();

  std::cout << "\n" << j3[1] << std::endl;

  int base_count = 0;
  for (auto &tree : j3){
    base_count++;
    std::cout << base_count << "[";
    int sub_count=0;
    for (auto &objs : tree){
      sub_count++;
      std::cout << sub_count << "{";
      int sscount = 0;
      for (auto &stree : objs){
        sscount++;
        std::cout << sscount << ",";
      }
      std::cout << "}\n";
    }
    std::cout << "]\n\n";
  }


  //Node n2;  Node n3;  Node n4;  Node n5;  Node n6;  Node n7;

  std::vector<AbstractNode*> nodes;
  for (auto &node : normalNodes){
    nodes.push_back(&node);
  }

  for (auto &node : leafNodes){
    nodes.push_back(&node);
  }

  for (auto node : nodes){
    std::cout << node->kind << std::endl;
  }


  Tree my_tree;
  /*
  std::map<unsigned int, AbstractNode> level_1;
  std::map<unsigned int, AbstractNode> level_2;
  */
  check_params2(j3, 4);
  std::cout << "\n***** pause *****" << std::endl;

  for (auto &mess: j3[0]){
    std::cout << "***  " << mess << "  ***\n";
  }

  for (json::iterator it = j3[0].begin(); it != j3[0].end(); ++it) {
    std::cout << it.key() << " : " << it.value() << "\n";
  }

  int fob_present = j3[0]["children"][0]["children"][0]["children"][0].count("leaf");
  std::cout << fob_present;

  std::vector<AbstractNode> my_tree2 = read_nodes_from_tree(j3[0]);
  std::cout << "Values: " << my_tree2.size() << std::endl;
  //my_tree.nodes[]
  std::cout << "\n***** END *****" << std::endl;
} // End main
