#ifndef __JIT_CODE_GENERATORS_HXX_
#define __JIT_CODE_GENERATORS_HXX_

#include "BranchlessTree.hxx"
#include "BranchedTree.hxx"
#include "TreeHelpers.hxx"

#include "TInterpreter.h" // for gInterpreter

#include <fstream>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
///// JITTING FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// JIT forest from a string containing generated code with -O3 optimization
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] tojit contains the code to be jitted
/// \param[in] s_namespace string containing the namespace name
/// \param[out] jitted function
template <typename T>
std::function<bool(const T *event)> jit_forest(const std::string &tojit, const std::string s_namespace = "")
{
   gInterpreter->Declare(tojit.c_str());
   bool use_namespaces = (!s_namespace.empty());

   std::string func_ref_name;
   if (use_namespaces)
      func_ref_name = "#pragma cling optimize(3)\n & jitted_" + s_namespace + "::generated_forest";
   else
      func_ref_name = "#pragma cling optimize(3)\n & generated_forest";
   auto ptr                = gInterpreter->Calc(func_ref_name.c_str());
   bool (*func)(const T *) = reinterpret_cast<bool (*)(const T *)>(ptr);
   std::function<bool(const T *)> fWrapped{func};
   return fWrapped;
}

////////////////////////////////////////////////////////////////////////////////
///// CODE GENERATION FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Helper to write the type as string
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
template <typename T> // primary template
std::string type_as_string()
{
   static_assert(std::is_floating_point<T>::value,
                 "This function has not been specialized for the provided type. Type has to be a floating_point type");
}
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
template <>
std::string type_as_string<long double>()
{
   return "long double";
}

////////////////////////////////////////////////////////////////////////////////
/// Generates headers for generated code
///
/// \param[in] fout stream where the code is written
/// \param[in] s_id namespace postfix
void generate_file_header(std::ostream &fout, const std::string &s_id)
{
   bool use_namespaces = (!s_id.empty());

   fout << "// Code automatically generated! " << std::endl;
   fout << "/// Functions that defines the"
        << " inference on a forest" << std::endl;
   fout << std::endl << std::endl;

   fout << "#pragma cling optimize(3)" << std::endl << std::endl;

   if (use_namespaces) {
      fout << "namespace jitted_" << s_id << "{" << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Generates objective functions for generated code
///
/// \param[in] fout stream where the code is written
/// \param[in] s_obj_func name of objective functions
void generate_objective_function(std::ostream &fout, const std::string &s_obj_func)
{
   const std::string s_logistic = "logistic";
   const std::string s_identity = "identity";
   if (s_obj_func.compare(s_logistic)) {
      fout << "1. / (1. + (1. / std::exp(result)));" << std::endl;
   } else if (s_obj_func.compare(s_identity)) {
      fout << "result;" << std::endl;
   } else {
      throw std::invalid_argument("Unknown objective function for JITTING");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Generates footer for generated code
///
/// \param[in] fout stream where the code is written
/// \param[in] s_id namespace postfix
/// \param[in] s_obj_func name of objective functions
void generate_file_footer(std::ostream &fout, const std::string &s_id, const std::string &s_obj_func)
{
   bool use_namespaces = (!s_id.empty());
   fout << "return ";
   generate_objective_function(fout, s_obj_func);
   fout << std::endl;
   fout << "} // end of function" << std::endl;
   // close namespace
   if (use_namespaces) {
      fout << "} // end of <prefix>" << s_id << " namespace" << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
///// BRANCHED VERSION /////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Generates if then else statements for bdts given a node
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] fout stream where the code is written
/// \param[in] node to be written down
template <typename T>
void generate_if_statement_for_bdt(std::ostream &fout, const BranchedTree::Node<T> *node)
{

   std::string condition =
      "(event[" + std::to_string(node->split_variable) + "]" + " < " + std::to_string(node->split_threshold) + ")";
   // IF part of statement
   fout << "if " << condition << "{" << std::endl;
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
   if (!node->child_false) {
      fout << "// This is a leaf node" << std::endl;
      fout << "result += " << std::to_string(node->leaf_false) << ";" << std::endl;
   } else { // if node is not a leaf node
      generate_if_statement_for_bdt(fout, node->child_false.get());
   }
   fout << "}" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Generate the code for Forest evaluation
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] fout stream where the code is written
/// \param[in] tree to be written down
/// \param[in] s_id namespace postfix
/// \param[in] s_obj_func name of objective functions
template <typename T>
void generate_code_forest(std::ostream &fout, const std::vector<BranchedTree::Tree<T>> &trees, std::string s_id = "",
                          const std::string &s_obj_func = "logistic")
{
   generate_file_header(fout, s_id);
   fout << type_as_string<T>() << " generated_forest (const " << type_as_string<T>() << " * event){" << std::endl
        << "" << type_as_string<T>() << " result = 0;" << std::endl;
   for (int i = 0; i < trees.size(); i++) {
      generate_if_statement_for_bdt<T>(fout, trees[i].nodes.get());
   }
   generate_file_footer(fout, s_id, s_obj_func);
}

////////////////////////////////////////////////////////////////////////////////
///// Branchless version /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Generates long array of thresholds
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] fout stream where the code is written
/// \param[in] tree to be written down
/// \param[in] total_array_length
template <typename T>
void generate_threshold_array(std::ostream &fout, const std::vector<BranchlessTree::Tree<T>> &trees,
                              const int total_array_length)
{
   fout << "const " << type_as_string<T>() << " thresholds[" // I could have done it just const
        << std::to_string(total_array_length) << "] {";
   for (int j = 0; j < trees.size(); j++) {
      for (int i = 0; i < trees[j].thresholds.size(); i++) {
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

/// \todo these two next functions perform code duplication: solve this!
////////////////////////////////////////////////////////////////////////////////
/// Generates long array of features
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] fout stream where the code is written
/// \param[in] tree to be written down
/// \param[in] total_array_length
template <typename T>
void generate_features_array(std::ostream &fout, const std::vector<BranchlessTree::Tree<T>> &trees,
                             const int total_array_length)
{
   // also possible with constexpr std::array<float,
   fout << "const int features[" << total_array_length << "] {";
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

////////////////////////////////////////////////////////////////////////////////
/// Generates long array of thresholds
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] fout stream where the code is written
/// \param[in] tree to be written down
/// \param[in] tree_index index of the tree in the containing vector of trees
template <typename T>
void generate_branchless_tree(std::ostream &fout, const BranchlessTree::Tree<T> &tree, const int tree_index_thresholds,
                              const int tree_index_features)
{
   fout << "index=0;" << std::endl;
   for (size_t j = 0; j < tree.tree_depth; ++j) {
      fout << "index = 2*index + 1 + (event[features[" << tree_index_features << "+index]] > thresholds["
           << tree_index_thresholds << "+index]);" << std::endl; // write
   }
   fout << "result += thresholds[" << tree_index_thresholds << "+index];" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Generate the code for Forest evaluation
///
/// \tparam T type for the prediction. Usually floating point type (float, double, long double)
/// \param[in] fout stream where the code is written
/// \param[in] tree to be written down
/// \param[in] s_id namespace postfix
/// \param[in] s_obj_func name of objective functions
template <typename T>
void generate_code_forest(std::ostream &fout, const std::vector<BranchlessTree::Tree<T>> &trees, std::string s_id = "",
                          const std::string &s_obj_func = "logistic")
{
   int total_array_length_thresholds = 0;
   int total_array_length_features   = 0;
   for (auto &tree : trees) {
      total_array_length_thresholds += tree.thresholds.size();
      total_array_length_features += tree.features.size();
   }
   generate_file_header(fout, s_id);
   // Generates the arrays with all cuts, and features on which to cut
   generate_threshold_array<T>(fout, trees, total_array_length_thresholds);
   generate_features_array<T>(fout, trees, total_array_length_features);

   fout << type_as_string<T>() << " generated_forest (const " << type_as_string<T>() << " *event){" << std::endl
        << "" << type_as_string<T>() << " result = 0;" << std::endl;

   fout << "int index=0;" << std::endl;

   int tree_index_thresholds = 0;
   int tree_index_features   = 0;
   for (int i = 0; i < trees.size(); i++) {
      generate_branchless_tree<T>(fout, trees[i], tree_index_thresholds, tree_index_features);
      tree_index_thresholds += trees[i].thresholds.size();
      tree_index_features += trees[i].features.size();
   }
   generate_file_footer(fout, s_id, s_obj_func);
}

#endif
// End of file
