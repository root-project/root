#ifndef __FOREST2_H_
#define __FOREST2_H_

#include "json.hpp"
#include "unique_bdt.hxx"
#include "array_bdt.hxx"
#include "bdt_helpers.hxx"
#include "jit_functions.hxx"
#include "jit_code_generators.hxx"

// using json = nlohmann::json;

std::string generated_files_path = "generated_files/"; // For DEBUG

template <typename T, typename Dummy>
class ForestBase {
protected:
   std::vector<unique_bdt::Tree<T>> _LoadFromJson(const std::string &filename, const bool &bool_sort_trees = true);
   /// Event by event prediction

   /// these two possibilities can take place:
   std::function<bool(T)> objective_func = binary_logistic<T>; // Default objective function

public:
   void set_objective_function(std::string func_name); // or int KIND
   void inference(const T *events_vector, unsigned int rows, unsigned int cols, std::vector<bool> &preds);
   void inference(const T *events_vector, unsigned int rows, unsigned int cols, std::vector<bool> &preds,
                  unsigned int loop_size); // Batched version
};

/// Branched version of the Forest (unique_ptr representation)
template <typename T>
class ForestBranched : public ForestBase<T, ForestBranched<T>> {

private:
public:
   std::vector<unique_bdt::Tree<T>> trees;
   void                             LoadFromJson(const std::string &key, const std::string &filename)
   {
      this->trees = this->_LoadFromJson(filename);
   }
};

/// Branched version of the Forest (topologically ordered representation)
template <typename T>
class ForestBranchless : public ForestBase<T, ForestBranchless<T>> {

private:
public:
   std::vector<array_bdt::Tree<T>> trees;
   void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

template <typename T>
class ForestBranchedJIT : public ForestBase<T, ForestBranchedJIT<T>> {

private:
   std::function<bool(const std::vector<float> &)> jitted_forest;

public:
   std::vector<array_bdt::Tree<T>> trees;
   void                            inference(const T *events_vector,

                                             unsigned int rows, unsigned int cols, std::vector<bool> &preds);
   void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

////////////////////////////////////////////////////////////////////////////////
///// functions implementations /////

/// Inference functions ///
/// Inference for non-jitted functions
template <typename T, typename Dummy>
void ForestBase<T, Dummy>::inference(const T *events_vector, unsigned int rows, unsigned int cols,
                                     std::vector<bool> &preds)
{
   T preds_tmp;
   for (size_t i = 0; i < rows; i++) {
      preds_tmp = 0;
      for (auto &tree : static_cast<Dummy &>(*this).trees) {
         preds_tmp += tree.inference(events_vector[i * cols]);
      }
      preds.push_back(this->objective_func(preds_tmp));
   }
}

template <typename T>
void ForestBranchedJIT<T>::inference(const T *events_vector, unsigned int rows, unsigned int cols,
                                     std::vector<bool> &preds)
{
   for (size_t i = 0; i < rows; i++) {
      preds.push_back(this->jitted_forest(i * cols));
   }
}

/// Loading functions ///
/// Load to unique_ptr implementation
template <typename T, typename Dummy>
std::vector<unique_bdt::Tree<T>> ForestBase<T, Dummy>::_LoadFromJson(const std::string &json_file,
                                                                     const bool &       bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<unique_bdt::Tree<T>> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree<T>(json_model[i], trees[i]);
   }

   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), unique_bdt::cmp<T>);
   }
   return std::move(trees);
}

template <typename T>
void ForestBranchless<T>::LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees)
{
   std::string my_config       = read_file_string(filename);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<array_bdt::Tree<T>> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      array_bdt::read_nodes_from_tree<T>(json_model[i], trees[i]);
   }

   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), array_bdt::cmp<T>);
   }

   this->trees = trees;
}

template <typename T>
void ForestBranchedJIT<T>::LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees)
{
   std::vector<unique_bdt::Tree<T>> trees           = this->_LoadFromJson(json_filename);
   int                              number_of_trees = trees.size();
   // JIT
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(std::rand()) + std::to_string(my_time);
   // std::cout << "current time used as namespace: " << s_namespace_name << std::endl;

   std::stringstream ss;
   generate_code_forest(ss, trees, number_of_trees, s_namespace_name);
   s_trees = ss.str();

   // write to file for debug
   std::filebuf fb;
   std::string  filename;
   if (bool_sort_trees == true)
      filename = generated_files_path + "generated_ordered_forest.h";
   else
      filename = generated_files_path + "generated_forest.h";

   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest<T>(os, trees, number_of_trees, s_namespace_name);
   fb.close();

   // JIT functions  // TODO clean here below
   std::function<bool(const float *)> func;
   func                = jit_branched_forest<T>(s_trees, s_namespace_name);
   this->jitted_forest = func;
}

#endif
