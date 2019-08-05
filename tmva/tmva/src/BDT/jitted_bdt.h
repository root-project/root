#ifndef __JITTED_BDT_H_
#define __JITTED_BDT_H_

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
#include "TInterpreter.h" // for gInterpreter


/// generates if then else statements for bdts
void generate_if_statement_for_bdt (std::ostream& fout,
                                          const unique_bdt::Node* node
                                        ){

  if (node->is_leaf_node){
    fout << "// This is a leaf node" << std::endl;
    //fout << "result = " << std::to_string(node->leaf) << ";" << std::endl;
    fout << "result += " << std::to_string(node->leaf) << ";" << std::endl;

  }
  else { // if node is not a leaf node
    std::string condition = "(event[" + std::to_string(node->split_variable) + "]"
                          + " < "
                          + std::to_string(node->split_threshold)
                          + ")";

    // IF part of statement
    fout << "if " << condition << "{" << std::endl;
    generate_if_statement_for_bdt(fout, node->child_true.get());

    fout << "}" << std::endl;

    // ELSE part of statement
    fout << "else " << "{ // if condition is not respected" << std::endl;
    generate_if_statement_for_bdt(fout, node->child_false.get());
    fout << "}" << std::endl;
  }
}

/// generates if then else statements for bdts
/*
void generate_if_statement_for_bdt_old (std::ostream& fout,
                                          //std::shared_ptr<Node> node
                                          const unique_bdt::Node* node
                                        ){

  std::string condition = "(event[" + std::to_string(node->split_variable) + "]"
                        + " < "
                        + std::to_string(node->split_threshold)
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
*/



/// Generate the code for BDTs evaluation
void generate_code_bdt(std::ostream& fout,
                        unique_bdt::Tree &tree,
                        int tree_number,
                        std::string s_id=""
                      ) {
  bool use_namespaces = (!s_id.empty());
  fout << "// File automatically generated! " << std::endl;
  fout << "/// Functions that defines the"
       << " inference of a single tree" << std::endl;
  fout << std::endl << std::endl;

  fout << "#pragma cling optimize(3)" << std::endl << std::endl;

  //fout << "#include <vector>" << std::endl;
  if (use_namespaces){
    // add "s_" to have a valid name
    fout << "namespace s_" << s_id << "{" << std::endl;
  }

  fout << "float generated_tree_" << std::to_string(tree_number)
       << "(const std::vector<float>& event){" << std::endl;
  fout << "float result = 0; // variable to store the result" << std::endl;

  generate_if_statement_for_bdt(fout, tree.nodes.get());

  fout << "return result;" << std::endl;
  fout << "}" << std::endl; // close function scope

  if (use_namespaces){
    fout << "} // end of s_" << s_id << " namespace" << std::endl;
  }
}

// /*
/// Generate the code for BDTs evaluation
void generate_code_forest(std::ostream& fout,
                        std::vector<unique_bdt::Tree> &trees,
                        int number_of_trees,
                        std::string s_id=""
                      ) {
  bool use_namespaces = (!s_id.empty());
  fout << "// File automatically generated! " << std::endl;
  fout << "/// Functions that defines the"
       << " inference of a single tree" << std::endl;
  fout << std::endl << std::endl;

  fout << "#pragma cling optimize(3)" << std::endl << std::endl;
  fout << "   " << std::endl;

  //fout << "#include <vector>" << std::endl;
  if (use_namespaces){
    // add "s_" to have a valid name
    fout << "namespace s_f_" << s_id << "{" << std::endl;
  }

  fout << "bool generated_forest (const std::vector<float>& event){" << std::endl
       << "float result = 0;" << std::endl;

  for (int i = 0; i < number_of_trees; i++) {
    // Function starts here
    generate_if_statement_for_bdt(fout, trees[i].nodes.get());

  }
  fout << "result = 1. / (1. + (1. / std::exp(result)));"<< std::endl
       << "return (result > 0.5) ? 1 : 0;" << std::endl
       << "}" << std::endl;


  // close namespace
  if (use_namespaces){
    fout << "} // end of s_f_" << s_id << " namespace" << std::endl;
  }
}

/*
/// Generate the code for BDTs evaluation
void generate_code_forest_old(std::ostream& fout,
                        std::vector<unique_bdt::Tree> &trees,
                        int number_of_trees,
                        std::string s_id=""
                      ) {
  bool use_namespaces = (!s_id.empty());
  fout << "// File automatically generated! " << std::endl;
  fout << "/// Functions that defines the"
       << " inference of a single tree" << std::endl;
  fout << std::endl << std::endl;

  fout << "#pragma cling optimize(3)" << std::endl << std::endl;
  fout << "   " << std::endl;

  //fout << "#include <vector>" << std::endl;
  if (use_namespaces){
    // add "s_" to have a valid name
    fout << "namespace s_f_" << s_id << "{" << std::endl;
  }

  for (int i = 0; i < number_of_trees; i++) {
    // Function starts here
    fout << "float generated_tree_" << std::to_string(i)
         << "(const std::vector<float>& event){" << std::endl;
    fout << "float result = 0; // variable to store the result" << std::endl;

    generate_if_statement_for_bdt(fout, trees[i].nodes.get());

    fout << "return result;" << std::endl;
    fout << "}" << std::endl; // close function scope
  }
  fout << "bool generated_forest (const std::vector<float>& event){" << std::endl
       << "float preds_tmp = 0;" << std::endl;
  for (int i = 0; i<number_of_trees; i++){
       fout << "preds_tmp += generated_tree_" << std::to_string(i) << " (event);"<< std::endl;
  }
  fout << "preds_tmp = 1. / (1. + (1. / std::exp(preds_tmp)));"<< std::endl
       << "return (preds_tmp > 0.5) ? 1 : 0;" << std::endl
       << "}" << std::endl;


  // close namespace
  if (use_namespaces){
    fout << "} // end of s_" << s_id << " namespace" << std::endl;
  }
}
*/


// */
///*
std::function<float (std::vector<float>)> jit_forest_string(std::string tojit,
                                                            std::string s_namespace=""
                                                          ){
   gInterpreter->Declare(tojit.c_str());
   bool use_namespaces = (!s_namespace.empty());

   std::string func_ref_name;
   if (use_namespaces){
     func_ref_name = "&s_f_" + s_namespace+"::generated_forest";
   }
   else{
     func_ref_name = "&generated_forest";
   }
   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   bool (*func)(std::vector<float>) = reinterpret_cast<bool(*)(std::vector<float>)>(ptr);
   std::function<bool (std::vector<float>)> fWrapped{func};
   return fWrapped;
}
// */


/// Generate the code evaluating all events at once
void generate_code_forest_batch(std::ostream& fout,
                        std::vector<unique_bdt::Tree> &trees,
                        int number_of_trees,
                        std::string s_id=""
                      ) {
  bool use_namespaces = (!s_id.empty());
  //size_t rows = 100000; // events_vector.size()
  //size_t cols = 5; // events_vector[0].size()
  fout << "// File automatically generated! " << std::endl;
  fout << "/// Functions that defines the"
       << " inference of a single tree" << std::endl;
  fout << std::endl << std::endl;

  fout << "#pragma cling optimize(3)" << std::endl << std::endl;
  fout << "   " << std::endl;

  //fout << "#include <vector>" << std::endl;
  if (use_namespaces){
    // add "s_" to have a valid name
    fout << "namespace s_f_event_" << s_id << "{" << std::endl;
  }

  // main function
  fout << "std::vector<bool> evaluate_forest (const std::vector<std::vector<float>>& events_vector){" << std::endl
     //  << "size_t num_events = " << std::to_string(rows) <<";" << std::endl
       << "size_t num_events = " << "events_vector.size()" <<";" << std::endl
       << "std::vector<bool> preds;" << std::endl
       << "preds.reserve(num_events);" << std::endl;

       fout << "std::vector<float> event;" << std::endl;
       fout  << "float result;" << std::endl;

  fout << "for (size_t i=0; i<num_events; i++){" << std::endl
       << "     result = 0;" << std::endl
       << "     event = events_vector[i];" << std::endl;
  for (int i = 0; i < number_of_trees; i++) {
        generate_if_statement_for_bdt(fout, trees[i].nodes.get());
      }
  fout << "     result = 1. / (1. + (1. / std::exp(result)));" << std::endl
       << "     preds[i] = (result > 0.5) ? 1 : 0;" << std::endl
       << "     }" << std::endl;
  fout << "return preds;" << std::endl
       << "}" << std::endl;


  // close namespace
  if (use_namespaces){
    fout << "} // end of s_f_event_" << s_id << " namespace" << std::endl;
  }
}

std::function<std::vector<bool> (std::vector<std::vector<float>>)> jit_event_forest_string(std::string tojit,
                                                            std::string s_namespace=""
                                                          ){
   gInterpreter->Declare(tojit.c_str());
   bool use_namespaces = (!s_namespace.empty());

   std::string func_ref_name;
   if (use_namespaces){
     func_ref_name = "& s_f_event_" + s_namespace+"::evaluate_forest";
   }
   else{
     func_ref_name = "&evaluate_forest";
   }
   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   std::vector<bool> (*func)(std::vector<std::vector<float>>) = reinterpret_cast<std::vector<bool>(*)(std::vector<std::vector<float>>)>(ptr);
   std::function<std::vector<bool> (std::vector<std::vector<float>>)> fWrapped{func};
   return fWrapped;
}



/// JIT function and and wrapp it in std::functions
std::function<float (std::vector<float>)> jit_function_reader_file(int tree_index){
   std::string filename = "generated_files/generated_tree_" + std::to_string(tree_index) + ".h";
   std::string tojit = read_file_string(filename);
   gInterpreter->Declare(tojit.c_str());

   std::string func_ref_name = "&generated_tree_"+std::to_string(tree_index);

   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   float (*func)(std::vector<float>) = reinterpret_cast<float(*)(std::vector<float>)>(ptr);
   std::function<float (std::vector<float>)> fWrapped{func};
   return fWrapped;
}


std::function<float (std::vector<float>)> jit_function_reader_string(int tree_index,
                                                            std::string tojit,
                                                            std::string s_namespace=""
                                                          ){
   gInterpreter->Declare(tojit.c_str());
   bool use_namespaces = (!s_namespace.empty());

   std::string func_ref_name;
   if (use_namespaces){
     func_ref_name = "&s_" + s_namespace+"::generated_tree_"+std::to_string(tree_index);
   }
   else{
     func_ref_name = "&generated_tree_"+std::to_string(tree_index);
   }

   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   float (*func)(std::vector<float>) = reinterpret_cast<float(*)(std::vector<float>)>(ptr);
   std::function<float (std::vector<float>)> fWrapped{func};
   return fWrapped;
}

#endif
// End of file
