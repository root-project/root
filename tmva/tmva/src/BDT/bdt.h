#include <string>
#include <map>

class Node{
public:
  bool is_leaf_node=0;
  double split_value = 0;
  int split_variable;
  int node_id;
  Node* child_true;
  Node* child_false;
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
  Node(){
    std::cout << "CREATION\n";
  }
  Node(Node const & ){ // copy constructor
    std::cout << "Copying\n";
    //Allocate the memory first
    /*
    child_true = new Node;
    child_false = new Node;
    //Then copy the value from the passed object
    child_true = node.child_true;
    child_false = node.child_false;*/
  }

  /*
  Node(Node&&) {
    std::cout << "&&\n";
  }*/
  Node(Node&& other){// initialize via default constructor, C++11 only
    swap(*this, other);
    std::cout << "AAAAAAAAAAAAA  &&\n";
  }


  ~Node(){
    std::cout << "DESTROYED\n";
    delete child_true;
    delete child_false;
  }
  friend void swap(Node &first, Node& second) { //nothrow
    // enable ADL (not necessary in our case, but good practice)
    using std::swap;
    std::cout << "swapping\n";

    // by swapping the members of two objects,
    // the two objects are effectively swapped
    swap(first.child_true, second.child_true);
    swap(first.child_false, second.child_false);
    swap(first.is_leaf_node, second.is_leaf_node);
    swap(first.split_value, second.split_value);
    swap(first.split_variable, second.split_variable);
    swap(first.node_id, second.node_id);
    swap(first.leaf_true, second.leaf_true);
    swap(first.leaf_false, second.leaf_false);
    swap(first.depth, second.depth);
    swap(first.missing, second.missing);
    swap(first.child_id_true, second.child_id_true);
    swap(first.child_id_false, second.child_id_false);
  }
  Node& operator= (const Node& other){
    std::cout << "fuuuuuuuuu\n";
    //swap(*this, other); // (2)
    return *this;
  } //Assignment operator

  Node& operator=(Node other){
    std::cout << "swapping\n";
    swap(*this, other); // (2)
    return *this;
  }
  Node& operator=(Node&& other) {
      std::cout << "MOOOOVED \n";
        // this may be called once or twice
        // if called twice, 'other' is the just-moved-from V subobject
        return *this;
    }

};





class Tree{
public:
  std::vector<Node*> nodes;
  double inference(double event[]){
    return nodes.back()->inference(event);
  }
  ~Tree(){
    int i = 0;
    for (auto node : nodes){
      std::cout << i << std::endl;
      i++;
      delete node;
    }
  }
};



// end
