#ifndef __JIT_CODE_GENERATORS_HXX_
#define __JIT_CODE_GENERATORS_HXX_

#include <string>
#include <map>
#include <iostream>
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "array_bdt.hxx"
#include "unique_bdt.hxx"
#include "bdt_helpers.hxx"
//////////////////////////////////////////////////////
/// CODE GENERATION FUNCTIONS
//////////////////////////////////////////////////////

/// Helper to write the type as string
template <typename T>
std::string type_as_string(); // primary template
template <>
std::string type_as_string<float>()
{
   return "float";
}
template <>
std::string type_as_string<double>()
{
   return "double";
}

////////////////////////////////////////////////////////////////////////////////
///// BRANCHED VERSION /////////////////////////////////////////////////////////

/// generates if then else statements for bdts
template <typename T>
void generate_if_statement_for_bdt(std::ostream &fout,
                                   // std::shared_ptr<Node> node
                                   const unique_bdt::Node<T> *node)
{

   std::string condition =
      "(event[" + std::to_string(node->split_variable) + "]" + " < " + std::to_string(node->split_threshold) + ")";

   // IF part of statement
   fout << "if " << condition << "{" << std::endl;
   // if (node->is_leaf_node){
   if (!node->child_true) {
      fout << "// This is a leaf node" << std::endl;
      fout << "result += " << std::to_string(node->leaf_true) << ";" << std::endl;
   } else { // if node is not a leaf node
      generate_if_statement_for_bdt<T>(fout, node->child_true.get());
   }
   fout << "}" << std::endl;

   // ELSE part of statement
   fout << "else "
        << "{ // if condition is not respected" << std::endl;
   // if (node->is_leaf_node){
   if (!node->child_false) {
      fout << "// This is a leaf node" << std::endl;
      fout << "result += " << std::to_string(node->leaf_false) << ";" << std::endl;
   } else { // if node is not a leaf node
      generate_if_statement_for_bdt(fout, node->child_false.get());
   }
   fout << "}" << std::endl;
}

/// Generate the code for Forest evaluation
template <typename T>
void generate_code_forest(std::ostream &fout, const std::vector<unique_bdt::Tree<T>> &trees, int number_of_trees,
                          std::string s_id = "")
{
   bool use_namespaces = (!s_id.empty());
   fout << "// File automatically generated! " << std::endl;
   fout << "/// Functions that defines the"
        << " inference of a single tree" << std::endl;
   fout << std::endl << std::endl;

   fout << "#pragma cling optimize(3)" << std::endl << std::endl;
   fout << "   " << std::endl;

   // fout << "#include <vector>" << std::endl;
   if (use_namespaces) {
      // add "s_" to have a valid name
      fout << "namespace jitted_" << s_id << "{" << std::endl;
   }

   fout << "bool generated_forest (const " << type_as_string<T>() << " * event){" << std::endl
        << "" << type_as_string<T>() << " result = 0;" << std::endl;

   for (int i = 0; i < number_of_trees; i++) {
      // Function starts here
      generate_if_statement_for_bdt<T>(fout, trees[i].nodes.get());
   }
   fout << "result = 1. / (1. + (1. / std::exp(result)));" << std::endl
        << "return (result > 0.5) ? 1 : 0;" << std::endl
        << "}" << std::endl;

   // close namespace
   if (use_namespaces) {
      fout << "} // end of s_f_" << s_id << " namespace" << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
///// Branchless version /////

/// Generates long array of thresholds
template <typename T>
void generate_threshold_array(std::ostream &fout, const std::vector<array_bdt::Tree<T>> &trees, int total_array_length)
{
   fout << "const " << type_as_string<T>() << " thresholds[" // I could have done it just const
        << std::to_string(total_array_length) << "] {";
   for (int j = 0; j < trees.size(); j++) {
      for (int i = 0; i < trees[j].thresholds.size(); i++) {
         // fout << std::to_string(trees[j].thresholds[i]);
         fout << trees[j].thresholds[i];
         if (i < trees[j].thresholds.size() - 1) {
            fout << ", ";
         }
      }
      if (j < trees.size() - 1) {
         fout << ", ";
      }
   }
   fout << "};" << std::endl;
}
/// Generates long array of thresholds
template <typename T>
void generate_features_array(std::ostream &fout, const std::vector<array_bdt::Tree<T>> &trees, int total_array_length)
{
   // Define long array of features
   // also possible with constexpr std::array<float,
   fout << "const int features[" << std::to_string(total_array_length) << "] {";
   for (int j = 0; j < trees.size(); j++) {
      for (int i = 0; i < trees[j].features.size(); i++) {
         fout << std::to_string(trees[j].features[i]);
         if (i < trees[j].features.size() - 1) {
            fout << ", ";
         }
      }
      if (j < trees.size() - 1) {
         fout << ", ";
      }
   }
   fout << "};" << std::endl;
}

/// generates if then else statements for bdts
template <typename T>
void generate_branchless_tree(std::ostream &fout, const array_bdt::Tree<T> &tree, int tree_index)
{
   fout << "index=0;" << std::endl;
   for (unsigned int j = 0; j < tree.tree_depth; ++j) {
      fout << "index = 2*index + 1 + (event[features[" << std::to_string(tree_index) << "+index]] > thresholds["
           << std::to_string(tree_index) << "+index]);" << std::endl; // write
   }
   fout << "result += thresholds[" << std::to_string(tree_index) << "+index];" << std::endl;
}

/// Generate the code for Forest evaluation
template <typename T>
void generate_code_forest(std::ostream &fout, const std::vector<array_bdt::Tree<T>> &trees, int number_of_trees,
                          std::string s_id = "")
{
   int total_array_length = 0;
   for (auto &tree : trees) {
      total_array_length += tree.array_length;
   }
   bool use_namespaces = (!s_id.empty());
   fout << "// File automatically generated! " << std::endl;
   fout << "/// Functions that defines the"
        << " inference on a forest, in a branchless manner" << std::endl;
   fout << std::endl << std::endl;

   fout << "#pragma cling optimize(3)" << std::endl << std::endl;
   fout << "   " << std::endl;

   if (use_namespaces) {
      fout << "namespace jitted_" << s_id << "{" << std::endl;
   }

   // Generates the arrays with all cuts, and features on which to cut
   generate_threshold_array<T>(fout, trees, total_array_length);
   generate_features_array<T>(fout, trees, total_array_length);

   fout << "bool generated_forest (const " << type_as_string<T>() << " *event){" << std::endl
        << "" << type_as_string<T>() << " result = 0;" << std::endl;

   fout << "unsigned int index=0;" << std::endl;

   int current_tree_index = 0;
   for (int i = 0; i < number_of_trees; i++) {
      generate_branchless_tree<T>(fout, trees[i], current_tree_index);
      current_tree_index += trees[i].array_length;
   }
   fout << "result = 1. / (1. + (1. / std::exp(result)));" << std::endl;
   fout << "return (result > 0.5);" << std::endl;

   fout << "} // end of function" << std::endl;

   // close namespace
   if (use_namespaces) {
      fout << "} // end of <prefix>" << s_id << " namespace" << std::endl;
   }
}

#endif
// End of file
