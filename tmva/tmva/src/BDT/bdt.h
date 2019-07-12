#include <string>
#include <map>



/*
struct Nodes {
  std::vector<LeafNode> leaf_nodes;
  std::vector<Node> normal_nodes;
} ;


//class tree
class Tree{
public:
  Nodes nodes;
  //std::map< unsigned int, std::map<unsigned int, AbstractNode> > nodes;
  //std::vector<AbstractNode> nodes; // change to vector of pointers
  double inference(double event[]){
    return 0;
  }
};

class Forest{// or BDT?
public:
  std::vector<Tree> trees;
  double inference(double event[]);
};
*/
// Definition of printname using scope resolution operator ::
/*int Node::get_child_num()
{
    return this->child_number;
}*/

class Node3{
public:
  bool is_leaf_node;
  double split_value = 0;
  int split_variable;
  int node_id;
  Node3* child_true;
  Node3* child_false;
  int depth;
  int missing; // what is missing?
  int child_id_true;
  int child_id_false;
  double leaf_true, leaf_false;
  double inference(double event[]){
    if (this->is_leaf_node){
      return ((event[split_variable] < split_value) ? leaf_true : leaf_false);
    }
    else{
      return ((event[split_variable] < split_value) ?
              child_true->inference(event) : child_false->inference(event));
    }
  }
};




// end
