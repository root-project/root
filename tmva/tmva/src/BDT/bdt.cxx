#include "bdt.h"
#include <string>
#include <map>
#include <iostream>

// counter
int Node::count=0;

// friend
void swap(Node &first, Node& second) { //nothrow
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




// constructor
Node::Node(){
  std::cout << "CREATION\n";
  count++;
}

// copy constructor
Node::Node(Node const & node ){
  std::cout << "Copying\n";
  //Allocate the memory first
  /*
  child_true = new Node();
  child_false = new Node();
  //Then copy the value from the passed object
  child_true = node.child_true;
  child_false = node.child_false;
  */
}

// move operator
Node::Node(Node&& other){// initialize via default constructor, C++11 only
  //swap(*this, other);
  std::cout << "AAAAAAAAAAAAA  &&\n";
}

// -------------   Assignement operators
Node& Node::operator=(Node&& other) {
    std::cout << "MOOOOVED \n";
      // this may be called once or twice
      // if called twice, 'other' is the just-moved-from V subobject
      return *this;
  }

Node& Node::operator=(Node other){
  std::cout << "swapping\n";
  swap(*this, other); // (2)
  return *this;
}

Node& Node::operator= (const Node& other){
  std::cout << "fuuuuuuuuu\n";
  //swap(*this, other); // (2)
  return *this;
} //Assignment operator

Node& Node::operator=(const Node&& other){
  std::cout << "ADSFSGF && \n";
}

// Destructor
Node::~Node(){
  std::cout << "DESTROYED\n";
  //delete child_true;
  //delete child_false;
}
