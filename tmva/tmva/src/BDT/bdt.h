#include <string>
#include <map>


// Test class

class AbstractNode{ // TODO make it abstract with pure virtual method
private:
protected:
public:
  std::string kind;
  double threshold = 0;
  std::string split_variable;
  int node_id;
  int depth;
  int missing; // what is missing
  double inference(double event[]){return -1;}
  //float get_threshold(){this->threshold;}

  //virtual std::string getKind();
};


class Node : public AbstractNode{
public:
  int child_id_true;
  int child_id_false;
  AbstractNode* child_true;
  AbstractNode* child_false;
  Node(){kind = "NormalNode";}
  double inference(double event[]){return 1;}
};

class LeafNode : public AbstractNode{
public:
  double leaf_true = 0;
  double leaf_false = 0;
  LeafNode(){kind = "LeafNode";}
  double inference(double event[]){return 2;}
};




//class tree
class Tree{
public:
  //std::map< unsigned int, std::map<unsigned int, AbstractNode> > nodes;
  std::vector<AbstractNode> nodes; // change to vector of pointers
  double inference(double event[]){
    return 0;
  }
};

class Forest{// or BDT?
public:
  std::vector<Tree> trees;
  double inference(double event[]);
};

// Definition of printname using scope resolution operator ::
/*int Node::get_child_num()
{
    return this->child_number;
}*/
