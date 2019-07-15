#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>


namespace array_bdt{
  // Create one for thresholds, one for features number
  double * tree_creator(int depth){
    int node_number = std::pow(2, depth);
    double tree[node_number] = {0};
    return tree;
  }
  /// Need a nlohmann::json object from an xgboost saved format
  void _read_nodes(json &jTree,
                    double[] &tree_features,
                    double[] &tree_values,
                    int depth,
                    int index
                  ){
    bool is_leaf_node = (
      (jTree["children"][0].find("leaf") != jTree["children"][0].end())
      && (jTree["children"][0].find("nodeid") != jTree["children"][0].end())
    );

    if (is_leaf_node){
      // std::pow(2,depth)
      tree_features[index]=-1;
      tree_values[index]=
      tmp_node->leaf_true = jTree["children"][0]["leaf"];
      tmp_node->leaf_false = jTree["children"][1]["leaf"];
      tmp_node->is_leaf_node=1;
      //tree.nodes.push_back(tmp_node);
    }
    else {
      tmp_node->child_true = _read_nodes(jTree["children"][0], tree);
      tmp_node->child_false = _read_nodes(jTree["children"][1], tree);
    }
    //tree.nodes.push_back(tmp_node);
    return std::move(tmp_node);

  }

  //std::vector<AbstractNode>
  void read_nodes_from_tree(json &jTree,Tree &tree){
    //std::vector<AbstractNode> nodes;
    tree.nodes = _read_nodes(jTree, tree);
    //return nodes;
  }

} // end namespace
