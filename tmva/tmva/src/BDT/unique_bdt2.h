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

namespace unique_bdt {

class Node {
public: // TODO: switch to private
   bool  is_leaf_node = 0;
   float split_threshold;
   int   split_variable;
   // float leaf_true, leaf_false;
   float                 leaf;
   std::unique_ptr<Node> child_true  = nullptr;
   std::unique_ptr<Node> child_false = nullptr;

public:
   static int count;
   // void set_split_variable(int split_variable) { this->split_variable = split_variable; }
   // void set_split_theshold(float split_threshold) { this->split_threshold = split_threshold; }
   // void set_is_leaf_node(bool is_leaf_node) { this->is_leaf_node = is_leaf_node; }

   float inference(const std::vector<float> &event)
   ///*
   {
      if (this->is_leaf_node) {
         return this->leaf;
      } else {
         return ((event[split_variable] < split_threshold) ? child_true->inference(event)
                                                           : child_false->inference(event));
      }
   }
   //*/
   /*
   float inference_old(const std::vector<float> &event)
   {
      if (this->is_leaf_node) {
         return ((event[split_variable] < split_threshold) ? leaf_true : leaf_false);
      } else {
         return ((event[split_variable] < split_threshold) ? child_true->inference(event)
                                                           : child_false->inference(event));
      }
   }
   // */
};

class Tree {
public:
   std::unique_ptr<Node> nodes;
   float                 inference(const std::vector<float> &event) { return nodes->inference(event); }
};

// Reading functions
void write_node_members(json const &jTree, std::unique_ptr<Node> &tmp_node);
void read_nodes_from_tree(json const &jTree, Tree &tree);

std::unique_ptr<Node> _read_nodes(json const &jTree, Tree &tree);

} // namespace unique_bdt

#endif
// end
