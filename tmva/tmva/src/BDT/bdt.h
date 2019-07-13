#include <string>
#include <map>
#include <iostream>

#include "json.hpp"
#include <fstream>
#include <sstream>
#include <streambuf>
#include <vector>

class Node{
public:
  static int count;
  bool is_leaf_node=0;
  double split_value = 0;
  int split_variable;
  int node_id;
  Node* child_true = nullptr;
  Node* child_false = nullptr;
  double leaf_true, leaf_false;
  int depth;
  int missing; // what is missing?
  int child_id_true;
  int child_id_false;

  double inference(double event[]){
    if (this->is_leaf_node){
      return ((event[split_variable] < split_value) ? leaf_true : leaf_false);
    }
    else{
      return ((event[split_variable] < split_value) ?
              child_true->inference(event) : child_false->inference(event));
    }
  }
  Node();
  Node(Node const & node);
  Node(Node&& other);
  ~Node();

  friend void swap(Node &first, Node& second);

  Node& operator= (const Node& other);
  Node& operator=(Node other);
  Node& operator=(Node&& other);
  Node& operator=(const Node&& other);

};








class Tree{
public:
  std::vector<Node*> nodes;
  double inference(double event[]){
    return nodes.back()->inference(event);
  }
// /*
  ~Tree(){
    int i = 0;
    for (auto node : nodes){
      std::cout << i << std::endl;
      i++;
      delete node;
    }
  }
// */
};



// end
