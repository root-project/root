#ifndef __RFORESTINFERENCE_HXX_
#define __RFORESTINFERENCE_HXX_

#include "ForestHelpers.hxx"

// using json = nlohmann::json;

std::string generated_files_path = "generated_files/"; // For DEBUG

//\todo:  Put this into namespace

template <typename T, typename forestType>
class ForestBase {
protected:
   std::vector<BranchedTree::Tree<T>> _LoadFromJson(const std::string &filename, const bool &bool_sort_trees = true);
   /// Event by event prediction

   /// these two possibilities can take place:
   std::function<T(T)> objective_func = logistic_function<T>; // Default objective function
   // std::function<bool(T)> objective_func = binary_logistic<T>; // Default objective function
   std::string s_obj_func = "logistic";

public:
   forestType trees;
   void       set_objective_function(std::string func_name); // or int KIND
   void       inference(const T *events_vector, unsigned int rows, unsigned int cols, T *preds);
   void       inference(const T *events_vector, unsigned int rows, unsigned int cols, T *preds,
                        unsigned int loop_size); // Batched version
   // void inference(const std::vector<std::vector<T>> &events_vector, unsigned int rows, unsigned int cols, T *preds);
   void _predict(T *predictions, const unsigned int num_predictions, unsigned int *);
};

/// Branched version of the Forest (unique_ptr representation)
template <typename T>
class ForestBranched : public ForestBase<T, std::vector<BranchedTree::Tree<T>>> {
private:
public:
   void LoadFromJson(const std::string &key, const std::string &filename)
   {
      this->trees = this->_LoadFromJson(filename);
   }
};

/// Branched version of the Forest (topologically ordered representation)
template <typename T>
class ForestBranchless : public ForestBase<T, std::vector<BranchlessTree::Tree<T>>> {

private:
public:
   void LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees = true);
   // void LoadFromJson2(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

template <typename T>
// class ForestBaseJIT : public ForestBase<T, std::function<bool(const std::vector<float> &)>> {
class ForestBaseJIT : public ForestBase<T, std::function<bool(const T *)>> {
private:
public:
   void inference(const T *events_vector, unsigned int rows, unsigned int cols, T *preds);
   // void inference(const std::vector<std::vector<T>> &events_vector, unsigned int rows, unsigned int cols, T *preds);
   // void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

template <typename T>
class ForestBranchedJIT : public ForestBaseJIT<T> {
public:
   void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

template <typename T>
class ForestBranchlessJIT : public ForestBaseJIT<T> {
public:
   void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

////////////////////////////////////////////////////////////////////////////////
///// functions implementations /////

//////////////////////////////////// Inference functions ///////////////////////

/// Inference for non-jitted functions
template <typename T, typename treeType>
void ForestBase<T, treeType>::inference(const T *events_vector, unsigned int rows, unsigned int cols, T *preds)
{
   T preds_tmp;
   for (size_t i = 0; i < rows; i++) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(events_vector + i * cols); //[i * cols]
      }
      preds[i] = this->objective_func(preds_tmp);
   }
}

/*
template <typename T, typename treeType>
void ForestBase<T, treeType>::inference(const T *events_vector, unsigned int rows, unsigned int cols, T *preds)
{
   T preds_tmp;
   for (size_t i = 0; i < rows; i++) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(events_vector + i * cols); //[i * cols]
      }
      preds[i] = this->objective_func(preds_tmp);
   }
}
// */

template <typename T>
void ForestBaseJIT<T>::inference(const T *events_vector, unsigned int rows, unsigned int cols,
                                 T *preds) // T *preds)
{
   for (size_t i = 0; i < rows; i++) {
      // preds[i]
      preds[i] = this->trees(events_vector + i * cols);
   }
}

///////////////////////////// Loading functions ////////////////////////////////
/// Load to unique_ptr implementation
template <typename T, typename treeType>
std::vector<BranchedTree::Tree<T>> ForestBase<T, treeType>::_LoadFromJson(const std::string &json_file,
                                                                          const bool &       bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<BranchedTree::Tree<T>> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      BranchedTree::read_nodes_from_tree<T>(json_model[i], trees[i]);
   }

   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), BranchedTree::cmp<T>);
   }
   return std::move(trees);
}

template <typename T>
void ForestBranchless<T>::LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees)
{
   std::vector<BranchedTree::Tree<T>> trees_unique = this->_LoadFromJson(json_filename, bool_sort_trees);

   this->trees = Branched2BranchlessTrees(trees_unique);
}

template <typename T>
void ForestBranchedJIT<T>::LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees)
{
   std::vector<BranchedTree::Tree<T>> trees = this->_LoadFromJson(json_filename, bool_sort_trees);

   // write to file for debug
   std::string filename = generated_files_path + "generated_forest.h";
   write_generated_code_to_file<T, BranchedTree::Tree<T>>(trees, this->s_obj_func, filename);

   this->trees = JitTrees<T, BranchedTree::Tree<T>>(trees, this->s_obj_func);
}

template <typename T>
void ForestBranchlessJIT<T>::LoadFromJson(const std::string &key, const std::string &json_filename,
                                          bool bool_sort_trees)
{
   std::vector<BranchedTree::Tree<T>>   trees_unique = this->_LoadFromJson(json_filename, bool_sort_trees);
   std::vector<BranchlessTree::Tree<T>> trees        = Branched2BranchlessTrees(trees_unique);

   // write to file for debug
   std::string filename = generated_files_path + "generated_forest.h";
   write_generated_code_to_file<T, BranchlessTree::Tree<T>>(trees, this->s_obj_func, filename);

   this->trees = JitTrees<T, BranchlessTree::Tree<T>>(trees, this->s_obj_func);
}

/// Deprecated functions
/*
/// REading model directly from json
template <typename T>
void ForestBranchless<T>::LoadFromJson2(const std::string &key, const std::string &filename, bool bool_sort_trees)
{
   std::string my_config       = read_file_string(filename);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<BranchlessTree::Tree<T>> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      BranchlessTree::read_nodes_from_tree<T>(json_model[i], trees[i]);
   }

   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), BranchlessTree::cmp<T>);
   }

   this->trees = trees;
}
*/

/*
/// inference on a vector of vectors for non jitted
template <typename T, typename treeType>
void ForestBase<T, treeType>::inference(const std::vector<std::vector<T>> &events_vector, unsigned int rows,
                                        unsigned int cols, T *preds)
{
   T preds_tmp;
   for (size_t i = 0; i < rows; i++) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(events_vector[i].data()); //[i * cols]
      }
      preds[i] = this->objective_func(preds_tmp);
   }
}
*/

/*
/// inference on a vector of vectors for jitted
template <typename T>
void ForestBaseJIT<T>::inference(const std::vector<std::vector<T>> &events_vector, unsigned int rows, unsigned int cols,
                                 T *preds)
{
   for (size_t i = 0; i < rows; i++) {
      preds[i] = this->trees(events_vector[i].data());
   }
}
*/
#endif
