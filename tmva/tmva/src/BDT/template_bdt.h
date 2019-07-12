#include <string>
#include <map>


// Test class

class AbstractNode{ // TODO make it abstract with pure virtual method
private:
protected:
public:
  std::string kind;
  double split_value = 0;
  int split_variable;
  int node_id;
  int depth;
  int missing; // what is missing?
  int child_id_true;
  int child_id_false;
  //virtual double inference(double event[]) = 0;
  //virtual std::string getKind();
};


template <class T>
class Node2 : public AbstractNode
{
  T child_true;
  T child_false;
  double inference(double event[]){
    return inference(event, this->split_variable, this->split_value,
                      this->child_true, this->child_false);
  }
};

double inference(
  double event[], int split_variable,
  double split_value, double child_true, double child_false)
{
  double result = 0; // change to "if then esle" terziary operator
  if (event[split_variable] < split_value){
    result=child_true;
  }
  else {
    result=child_false;
  }
  return result;
}

template <class T>
double inference(
  double event[], int split_variable,
  double split_value, Node2<T> child_true, Node2<T> child_false)
{
  double result = 0; // change to "if then esle" terziary operator
  if (event[split_variable] < split_value){
    result=child_true->inference(event, child_true.split_variable,
                                        child_true.split_value,
                                        child_true.child_true,
                                        child_true.child_false);
  }
  else {
    result=child_false->inference(event, child_false.split_variable,
                                        child_false.split_value,
                                        child_false.child_true,
                                        child_false.child_false);
  }
  return result;
}

/*
class NodesVector
{
  vector <Node2 *> nodes;
};
*/

class Node : public AbstractNode{
public:
  AbstractNode* child_true;
  AbstractNode* child_false;
  Node(){
    kind = "NormalNode";
  }
  /*
  Node(AbstractNode* child_true, AbstractNode* child_false ){
    kind = "NormalNode";
    this->child_true = child_true;
    this->child_false;
  }*/

  ~Node(){
    delete child_true;
    delete child_false;
  }
};

class LeafNode : public AbstractNode{
public:
  double leaf_true = 0;
  double leaf_false = 0;
  LeafNode(){kind = "LeafNode";}
  double inference(double event[]){
    return 2;
  }
};

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

// Definition of printname using scope resolution operator ::
/*int Node::get_child_num()
{
    return this->child_number;
}*/
