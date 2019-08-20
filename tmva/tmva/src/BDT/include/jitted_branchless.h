#ifndef __JITTED_BRANCHLESS_H_
#define __JITTED_BRANCHLESS_H_
//#include "unique_bdt.h"
#include "array_bdt.h"
#include "TInterpreter.h" // for gInterpreter
#include "bdt_helpers.h"

/*
unsigned int array_length = 8;
inline float decision_tree(float* event, const float features [array_length], const float tree [array_length]) {
    unsigned int i = 0;
    i = 2*i + 1 + (event[features[i]] >= tree[i]); // write
    i = 2*i + 1 + (event >= tree[i]);
    return leaf[i];  // keep leaves?
}
*/

/// Generates long array of thresholds
void generate_threshold_array(std::ostream& fout,
                              std::vector<array_bdt::Tree> &trees,
                              int total_array_length
                                        ){
  fout << "const float thresholds["  // I could have done it just const
       << std::to_string(total_array_length)
       << "] {";
  for (int j = 0; j<trees.size(); j++) {
    for (int i = 0; i<trees[j].thresholds.size(); i++){
      fout << std::to_string(trees[j].thresholds[i]);
      if (i < trees[j].thresholds.size()-1){
          fout << ", ";
      }
    }
    if (j < trees.size()-1){
        fout << ", ";
    }
  }
  fout << "};" << std::endl;
}
/// Generates long array of thresholds
void generate_features_array(std::ostream& fout,
                             std::vector<array_bdt::Tree> &trees,
                             int total_array_length
                            ){
  // Define long array of features
  fout << "const int features["
       << std::to_string(total_array_length)
       << "] {";
  for (int j = 0; j<trees.size(); j++) {
    for (int i = 0; i<trees[j].features.size(); i++){
      fout << std::to_string(trees[j].features[i]);
      if (i < trees[j].features.size()-1){
          fout << ", ";
      }
    }
    if (j < trees.size()-1){
        fout << ", ";
    }
  }
  fout << "};" << std::endl;
}

/// Generates long array of thresholds
void generate_threshold_array2(std::ostream& fout,
                              std::vector<array_bdt::Tree> &trees,
                              int total_array_length
                                        ){
  fout << "constexpr std::array<float, "  // I could have done it just const
       << std::to_string(total_array_length)
       << "> thresholds {";
  for (int j = 0; j<trees.size(); j++) {
    for (int i = 0; i<trees[j].thresholds.size(); i++){
      fout << std::to_string(trees[j].thresholds[i]);
      if (i < trees[j].thresholds.size()-1){
          fout << ", ";
      }
    }
    if (j < trees.size()-1){
        fout << ", ";
    }
  }
  fout << "};" << std::endl;
}
/// Generates long array of thresholds
void generate_features_array2(std::ostream& fout,
                             std::vector<array_bdt::Tree> &trees,
                             int total_array_length
                            ){
  // Define long array of features
  fout << "constexpr std::array<int, "
       << std::to_string(total_array_length)
       << "> features {";
  for (int j = 0; j<trees.size(); j++) {
    for (int i = 0; i<trees[j].features.size(); i++){
      fout << std::to_string(trees[j].features[i]);
      if (i < trees[j].features.size()-1){
          fout << ", ";
      }
    }
    if (j < trees.size()-1){
        fout << ", ";
    }
  }
  fout << "};" << std::endl;
}

/// generates if then else statements for bdts
void generate_branchless_tree (std::ostream& fout,
                            const array_bdt::Tree &tree,
                            int tree_index
                                        ){
  fout << "index=0;" << std::endl;
  for (unsigned int j = 0; j<tree.tree_depth; ++j){
    fout << "index = 2*index + 1 + (event[features[" << std::to_string(tree_index)
         << "+index]] > thresholds[" << std::to_string(tree_index)
         << "+index]);" << std::endl; // write
  }
  fout << "result += thresholds[" << std::to_string(tree_index)
       << "+index];" << std::endl;
}



/// Generate the code for Forest evaluation
void generate_code_branchless_forest(std::ostream& fout,
                        std::vector<array_bdt::Tree> &trees,
                        int number_of_trees,
                        std::string s_id=""
                      ) {
  int total_array_length = 0;
  for (auto &tree:trees){
    total_array_length += tree.array_length;
  }
  bool use_namespaces = (!s_id.empty());
  fout << "// File automatically generated! " << std::endl;
  fout << "/// Functions that defines the"
       << " inference on a forest, in a branchless manner" << std::endl;
  fout << std::endl << std::endl;

  fout << "#pragma cling optimize(3)" << std::endl << std::endl;
  fout << "   " << std::endl;

  //fout << "#include <vector>" << std::endl;
  if (use_namespaces){
    // add "s_" to have a valid name
    fout << "namespace branchless_" << s_id << "{" << std::endl;
  }

  // Generates the arrays with all cuts, and features on which to cut
  generate_threshold_array(fout, trees, total_array_length);
  generate_features_array(fout, trees, total_array_length);

  fout << "bool branchless_generated_forest (const float *event){" << std::endl
       << "float result = 0;" << std::endl;

  fout << "unsigned int index=0;" << std::endl;

  int current_tree_index = 0;
  for (int i = 0; i < number_of_trees; i++) {
    generate_branchless_tree(fout, trees[i], current_tree_index);
    current_tree_index += trees[i].array_length;

  }
  fout << "result = 1. / (1. + (1. / std::exp(result)));"<< std::endl
       << "return (result > 0.5);" << std::endl;

  fout << "} // end of function" << std::endl;


  // close namespace
  if (use_namespaces){
    fout << "} // end of s_f_" << s_id << " namespace" << std::endl;
  }
}



/// JIT forest from sringed code
std::function<bool (const float* event)> jit_branchless_forest(std::string tojit,
                                                            const std::string s_namespace=""
                                                          ){
   gInterpreter->Declare(tojit.c_str());
   bool use_namespaces = (!s_namespace.empty());

   std::string func_ref_name;
   if (use_namespaces)
     func_ref_name = "& branchless_" + s_namespace+"::branchless_generated_forest";
   else
     func_ref_name = "& branchless_generated_forest";
   auto ptr = gInterpreter->Calc(func_ref_name.c_str());
   bool (*func)(const float*) = reinterpret_cast<bool(*)(const float *)>(ptr);
   std::function<bool (const float *)> fWrapped{func};
   return fWrapped;
}

#endif
