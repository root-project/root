#include <string>
#include <map>


// Test class

class AbstractNode{
private:
  // double threshold = 0; --> moved to public
public:
  double threshold = 0;
  //float get_threshold(){this->threshold;}

  //virtual std::string getKind();
};


class Node : public AbstractNode{
public:
  std::string kind = "NormalNode";
  unsigned int level = 0;
  unsigned int child_number = 0;
  AbstractNode* child_1;
  AbstractNode* child_2;
};

class LeafNode : public AbstractNode{
public:
  std::string kind = "LeafNode";
  double leaf_1 = 0;
  double leaf_2 = 0;
};

//class tree
class Tree{
public:
  std::map< unsigned int, std::map<unsigned int, AbstractNode> > nodes;
  double inference(double event[]){
    return 0;
  }
};

// Definition of printname using scope resolution operator ::
/*int Node::get_child_num()
{
    return this->child_number;
}*/
