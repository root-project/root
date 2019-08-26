#ifndef __UNIQUE_BDT_HXX_
#define __UNIQUE_BDT_HXX_

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

template <typename T>
class Node {
public: // TODO: switch to private
   bool is_leaf_node = 0;
   T    split_threshold;
   int  split_variable;
   T    leaf_true, leaf_false;
   // int  depth = 0;

   std::unique_ptr<Node<T>> child_true  = nullptr;
   std::unique_ptr<Node<T>> child_false = nullptr;

public:
   // void set_split_variable(int split_variable) { this->split_variable = split_variable; }
   // void set_split_theshold(float split_threshold) { this->split_threshold = split_threshold; }
   // void set_is_leaf_node(bool is_leaf_node) { this->is_leaf_node = is_leaf_node; }

   inline T inference(const std::vector<T> &event);
   inline T inference(const T *event);
   ///*
};

template <typename T>
class Tree {
public:
   std::unique_ptr<Node<T>> nodes;

   inline T inference(const std::vector<T> &event) { return nodes->inference(event); }
   inline T inference(const T *event) { return nodes->inference(event); }
};

////////////////////////////////////////////////////////////////////////////////
///// Functions definitions /////
template <typename T>
inline T Node<T>::inference(const std::vector<T> &event)
{
   {
      if (event[split_variable] <= this->split_threshold) {
         if (child_true)
            child_true->inference(event);
         else
            return this->leaf_true;

      } else {
         if (child_false)
            child_false->inference(event);
         else
            return this->leaf_false;
      }
   }
}

template <typename T>
inline T Node<T>::inference(const T *event)
{
   {
      if (event[split_variable] <= this->split_threshold) {
         if (child_true)
            child_true->inference(event);
         else
            return this->leaf_true;

      } else {
         if (child_false)
            child_false->inference(event);
         else
            return this->leaf_false;
      }
   }
}

/// Reading members of the json
template <typename T>
void write_node_members(json const &jTree, std::unique_ptr<Node<T>> &tmp_node)
{
   tmp_node->split_threshold = jTree.at("split_condition");
   std::string tmp_str       = jTree.at("split").get<std::string>();
   tmp_str.erase(tmp_str.begin(), tmp_str.begin() + 1); // remove initial "f"
   tmp_node->split_variable = std::stoi(tmp_str);
}

template <typename T>
std::unique_ptr<Node<T>> _read_nodes_from_json(json const &jTree, Tree<T> &tree)
{
   std::unique_ptr<Node<T>> tmp_node(new Node<T>);
   write_node_members<T>(jTree, tmp_node);
   if (jTree.at("children").at(0).find("leaf") != jTree.at("children").at(0).end()) {
      tmp_node->leaf_true    = jTree.at("children").at(0).at("leaf");
      tmp_node->is_leaf_node = 1;
   } else {
      tmp_node->child_true = _read_nodes_from_json<T>(jTree.at("children").at(0), tree);
   }
   if (jTree.at("children").at(1).find("leaf") != jTree.at("children").at(1).end()) {
      tmp_node->leaf_false   = jTree.at("children").at(1).at("leaf");
      tmp_node->is_leaf_node = 1;
   } else {
      tmp_node->child_false = _read_nodes_from_json<T>(jTree.at("children").at(1), tree);
   }
   return std::move(tmp_node);
}

/// read tree structure from xgboost json file
template <typename T>
void read_nodes_from_tree(json const &jTree, Tree<T> &tree)
{
   tree.nodes = _read_nodes_from_json<T>(jTree, tree);
}

/// Comparison functions between trees
template <typename T>
bool cmp(const Tree<T> &a, const Tree<T> &b)
{
   if (a.nodes->split_variable == b.nodes->split_variable) {
      return a.nodes->split_threshold < b.nodes->split_threshold;
   } else {
      return a.nodes->split_variable < b.nodes->split_variable;
   }
}

} // namespace unique_bdt

#endif
// end
