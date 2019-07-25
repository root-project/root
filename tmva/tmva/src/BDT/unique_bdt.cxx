#include "unique_bdt.h"
#include <string>
#include <map>
#include <iostream>


namespace unique_bdt{


  // counter
  int Node::count=0;



  // Reading functions
  void write_node_members(json const &jTree, std::unique_ptr<Node> &tmp_node){
    tmp_node->split_threshold = jTree["split_condition"];
    //tmp_node->node_id = jTree["nodeid"];
    //tmp_node->child_id_true = jTree["yes"];
    //tmp_node->child_id_false = jTree["no"];

    std::string tmp_str = jTree["split"].get<std::string>();
    tmp_str.erase(tmp_str.begin(), tmp_str.begin()+1); // remove initial "f"
    tmp_node->split_variable = std::stoi(tmp_str);
  }

  /// Need a nlohmann::json object from an xgboost saved format
  std::unique_ptr<Node> _read_nodes(json const &jTree, Tree &tree){
    bool is_leaf_node = (
      (jTree["children"][0].find("leaf") != jTree["children"][0].end())
      && (jTree["children"][0].find("nodeid") != jTree["children"][0].end())
    );

    std::unique_ptr<Node> tmp_node(new Node);
    write_node_members(jTree, tmp_node);
    if (is_leaf_node){
      tmp_node->leaf_true = jTree["children"][0]["leaf"];
      tmp_node->leaf_false = jTree["children"][1]["leaf"];
      tmp_node->is_leaf_node=1;
    }
    else {
      tmp_node->child_true = _read_nodes(jTree["children"][0], tree);
      tmp_node->child_false = _read_nodes(jTree["children"][1], tree);
    }
    return std::move(tmp_node);
  }

  /// read tree structure from xgboost json file
  void read_nodes_from_tree(json const &jTree,Tree &tree){
    tree.nodes = _read_nodes(jTree, tree);
  }

} // end namespace

// end file
