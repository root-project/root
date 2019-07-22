#ifndef __UNIQUE_BDT_H_
#define __UNIQUE_BDT_H_


#include <string>
#include <map>
#include <iostream>

#include "json.hpp"
#include <fstream>
#include <sstream>
#include <streambuf>
#include <vector>
#include <memory>

using json = nlohmann::json;

namespace unique_bdt{
  class Node{
  private:
    //int depth;
    //int missing; // what is missing?
    //int child_id_true;
    //int child_id_false;
    //int node_id;

  public:
    bool is_leaf_node=0;
    float split_threshold;
    int split_variable;
    static int count;
    void set_split_variable(int split_variable){
      this->split_variable=split_variable;
    }
    void set_split_theshold(float split_threshold){
      this->split_threshold=split_threshold;
    }
    void set_is_leaf_node(bool is_leaf_node){
      this->is_leaf_node=is_leaf_node;
    }

    std::unique_ptr<Node> child_true;
    std::unique_ptr<Node> child_false;
    float leaf_true, leaf_false;




    float inference(std::vector<float> event){
      if (this->is_leaf_node){
        return ((event[split_variable] < split_threshold) ? leaf_true : leaf_false);
      }
      else{
        return ((event[split_variable] < split_threshold) ?
                child_true->inference(event) : child_false->inference(event));
      }
    }
    Node();
    Node(Node const & node);
    Node(Node&& other);
    ~Node();

    //friend void swap(Node &first, Node& second);
    //Node& operator= (const Node& other);
    //Node& operator=(Node other);
    //Node& operator=(Node&& other);
    //Node& operator=(const Node&& other);

  };

  class Tree{
  public:
    //std::vector<std::unique_ptr<Node>> nodes;
    std::unique_ptr<Node> nodes;
    float inference(std::vector<float> event){
      return nodes->inference(event);
    }

  };

  // Reading functions
  void write_node_members(json const &jTree, std::unique_ptr<Node> &tmp_node);
  std::unique_ptr<Node> _read_nodes(json const &jTree, Tree &tree);
  void read_nodes_from_tree(json const &jTree,Tree &tree);
}

#endif
// end
