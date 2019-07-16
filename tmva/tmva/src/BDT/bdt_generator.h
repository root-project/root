
#include <string>
#include <map>
#include <iostream>
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>

//#include "bdt.h"
#include "unique_bdt.h"

using namespace unique;

//  /*


void generate_if_statement_for_bdt (std::ostream& fout,
                                          //std::shared_ptr<Node> node
                                          const Node* node
                                        ){

  std::string condition = "(event[" + std::to_string(node->split_variable) + "]"
                        + " < "
                        + std::to_string(node->split_value)
                        + ")";

  // IF part of statement
  fout << "if " << condition << "{" << std::endl;
  if (node->is_leaf_node){
    fout << "// This is a leaf node" << std::endl;
    fout << "result = " << std::to_string(node->leaf_true) << ";" << std::endl;
  }
  else { // if node is not a leaf node
    generate_if_statement_for_bdt(fout, node->child_true.get());
  }
  fout << "}" << std::endl;

  // ELSE part of statement
  fout << "else " << "{ // if condition is not respected" << std::endl;
  if (node->is_leaf_node){
    fout << "// This is a leaf node" << std::endl;
    fout << "result = " << std::to_string(node->leaf_false) << ";" << std::endl;
  }
  else { // if node is not a leaf node
    generate_if_statement_for_bdt(fout, node->child_false.get());
  }
  fout << "}" << std::endl;
}


// */


void generate_code_bdt(std::ostream& fout,
                        Tree &tree,
                        int tree_number
                      ) {

  fout << "// File automatically generated! " << std::endl;
  fout << "/// Functions that defines the"
       << " inference of a single tree" << std::endl;
  fout << std::endl << std::endl;

  fout << "double generated_tree_" << std::to_string(tree_number)
       << "(double event[]){" << std::endl;
  fout << "double result = 0; // variable to store the result" << std::endl;

  generate_if_statement_for_bdt(fout, tree.nodes.get());

  fout << "return result;" << std::endl;
  fout << "}" << std::endl;
}
