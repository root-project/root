#ifndef __RFORESTINFERENCE_HXX_
#define __RFORESTINFERENCE_HXX_

#include "ForestHelpers.hxx"

std::string generated_files_path = "generated_files/"; // For DEBUG

///  \todo:  Put this into namespace

/**
 * \class ForestBase
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 * \tparam forestType type of the underlying Forest
 */
template <typename T, typename forestType>
class ForestBase {
protected:
   /// Loads a forest from a json file
   std::vector<BranchedTree::Tree<T>> _LoadFromJson(const std::string &key, const std::string &filename,
                                                    const bool &bool_sort_trees = true);

   std::string         s_obj_func;     ///< Default string_describing the objective function
   std::function<T(T)> objective_func; ///< Default objective function

public:
   forestType trees; ///< Store the forest, either as vector or jitted function

   ForestBase() : s_obj_func("logistic"), objective_func(logistic_function<T>) {}

   /// Set objective function from a string name
   // void set_objective_function(const std::string &func_name); // or int KIND
   /// Inference event by events
   void inference(const T *events_vector, const int rows, const int cols, T *preds);
   /// Inference in a loop blocked fashion
   void inference(const T *events_vector, const int rows, const int cols, T *preds, const int loop_size);

   /// Goes from probability to classification
   // void _predict(T *predictions, const int num_predictions, int *);
};

/**
 * \class ForestBranched
 * Branched version of the Forest (unique_ptr representation)
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 */
template <typename T>
class ForestBranched : public ForestBase<T, std::vector<BranchedTree::Tree<T>>> {
public:
   /// Load Forest from a json file
   void LoadFromJson(const std::string &key, const std::string &filename, const bool &bool_sort_trees = true)
   {
      this->trees = this->_LoadFromJson(key, filename, bool_sort_trees);
   }
};

/**
 * \class ForestBranchless
 * Branchless version of the Forest (topologically ordered representation)
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 */
template <typename T>
class ForestBranchless : public ForestBase<T, std::vector<BranchlessTree::Tree<T>>> {
public:
   /// Load Forest from a json file
   void LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees = true);
};

/**
 * \class ForestBase
 * Branched version of the Forest (topologically ordered representation)
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 */
template <typename T>
class ForestBaseJIT : public ForestBase<T, std::function<T(const T *)>> {
public:
   /// Inference event by events
   void inference(const T *events_vector, const int rows, const int cols, T *preds);
   /// Inference in a loop blocked fashion
   void inference(const T *events_vector, const int rows, const int cols, T *preds, const int loop_size);
};

template <typename T>
class ForestBranchedJIT : public ForestBaseJIT<T> {
public:
   /// Load Forest from a json file
   void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

template <typename T>
class ForestBranchlessJIT : public ForestBaseJIT<T> {
public:
   /// Load Forest from a json file
   void LoadFromJson(const std::string &key, const std::string &filename, bool bool_sort_trees = true);
};

////////////////////////////////////////////////////////////////////////////////
///// functions definitions /////
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////// Inference functions ///////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \param[in] events_vector pointer to data containing the events
/// \param[in] rows number of events in events_vecctor
/// \param[in] cols number of features per events i.e. columns in events_vector
/// \param[in] preds pointer to the pre-allocated data that's gonna be filled by this function
///*
template <typename T, typename treeType>
void ForestBase<T, treeType>::inference(const T *events_vector, const int rows, const int cols, T *preds)
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
//*/

////////////////////////////////////////////////////////////////////////////////
/// \param[in] events_vector pointer to data containing the events
/// \param[in] rows number of events in events_vecctor
/// \param[in] cols number of features per events i.e. columns in events_vector
/// \param[in] preds pointer to the pre-allocated data that's gonna be filled by this function
/// \param[in] loop_size
/*
template <typename T, typename treeType>
void ForestBase<T, treeType>::inference(const T *events_vector, const int rows, const int cols, T *preds,
                                        const int loop_size)
{
   int rest = rows % loop_size;

   int index     = 0;
   int num_trees = this->trees.size();
   T   preds_tmp = 0;

   T *preds_tmp_arr = new T[loop_size]{0};

   for (; index < rows - rest; index += loop_size) {
      for (int i = 0; i < num_trees; i++) {
         for (int j = 0; j < loop_size; j++) {
            preds_tmp_arr[j] += trees[i].inference(events_vector + (index + j) * cols);
         }
      }
      for (int j = 0; j < loop_size; j++) {
         preds[index + j] = (this->objective_func(preds_tmp_arr[j]));
         preds_tmp_arr[j] = 0;
      }
   }
   /// rest loop
   for (int j = index; j < rows; j++) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(events_vector + j * cols);
      }
      preds[j] = this->objective_func(preds_tmp);
   }
   delete[] preds_tmp_arr;
}
// */ //

// /*
template <typename T, typename treeType>
void ForestBase<T, treeType>::inference(const T *events_vector, const int rows, const int cols, T *preds,
                                        const int loop_size)
{
   for (int j = 0; j < rows; j++) {
      preds[j] = 0;
   }

   for (int i = 0; i < this->trees.size(); i++) {
      for (int j = 0; j < rows; j++) {
         preds[j] += trees[i].inference(events_vector + j * cols);
      }
   }

   for (int j = 0; j < rows; j++) {
      preds[j] = this->objective_func(preds[j]);
   }
}
//*/

////////////////////////////////////////////////////////////////////////////////
/// \param[in] events_vector pointer to data containing the events
/// \param[in] rows number of events in events_vecctor
/// \param[in] cols number of features per events i.e. columns in events_vector
/// \param[in] preds pointer to the pre-allocated data that's gonna be filled by this function
template <typename T>
void ForestBaseJIT<T>::inference(const T *events_vector, const int rows, const int cols,
                                 T *preds) // T *preds)
{
   for (int i = 0; i < rows; i++) {
      // preds[i]
      preds[i] = this->trees(events_vector + i * cols);
      // std::cout << preds[i] << "  \n";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \param[in] events_vector pointer to data containing the events
/// \param[in] rows number of events in events_vecctor
/// \param[in] cols number of features per events i.e. columns in events_vector
/// \param[in] preds pointer to the pre-allocated data that's gonna be filled by this function
/// \param[in] loop_size
template <typename T>
void ForestBaseJIT<T>::inference(const T *events_vector, const int rows, const int cols, T *preds, const int loop_size)
{
   int rest  = rows % loop_size;
   int index = 0;
   for (; index < rows - rest; index += loop_size) {
      for (int j = index; j < index + loop_size; j++) {
         preds[j] = this->trees(events_vector + j * cols);
      }
   }
   // reminder loop
   for (int j = index; j < rows; j++) {
      preds[j] = this->trees(events_vector + j * cols);
   }
}

///////////////////////////// Loading functions ////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \param[in] key string with the key for the correct model inside the file
/// \param[in] json_file string with the filename that contains the models
/// \param[in] bool_sort_trees should the forest be ordered? `default=true`
/// \param[out] Branched representation of the forest, with is used also to derive other forests
template <typename T, typename treeType>
std::vector<BranchedTree::Tree<T>> ForestBase<T, treeType>::_LoadFromJson(const std::string &key,
                                                                          const std::string &json_file,
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

////////////////////////////////////////////////////////////////////////////////
/// \param[in] key string with the key for the correct model inside the file
/// \param[in] json_file string with the filename that contains the models
/// \param[in] bool_sort_trees should the forest be ordered? `default=true`
/// \param[out] Forest representation
template <typename T>
void ForestBranchless<T>::LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees)
{
   std::vector<BranchedTree::Tree<T>> trees_unique = this->_LoadFromJson(key, json_filename, bool_sort_trees);

   this->trees = Branched2BranchlessTrees(trees_unique);
}

////////////////////////////////////////////////////////////////////////////////
/// \param[in] key string with the key for the correct model inside the file
/// \param[in] json_file string with the filename that contains the models
/// \param[in] bool_sort_trees should the forest be ordered? `default=true`
/// \param[out] Forest representation (jitted function)
template <typename T>
void ForestBranchedJIT<T>::LoadFromJson(const std::string &key, const std::string &json_filename, bool bool_sort_trees)
{
   std::vector<BranchedTree::Tree<T>> trees = this->_LoadFromJson(key, json_filename, bool_sort_trees);

   // write to file for debug
   std::string filename = generated_files_path + "generated_forest.h";
   write_generated_code_to_file<T, BranchedTree::Tree<T>>(trees, this->s_obj_func, filename);

   this->trees = JitTrees<T, BranchedTree::Tree<T>>(trees, this->s_obj_func);
}

////////////////////////////////////////////////////////////////////////////////
/// \param[in] key string with the key for the correct model inside the file
/// \param[in] json_file string with the filename that contains the models
/// \param[in] bool_sort_trees should the forest be ordered? `default=true`
/// \param[out] Forest representation (jitted function)
template <typename T>
void ForestBranchlessJIT<T>::LoadFromJson(const std::string &key, const std::string &json_filename,
                                          bool bool_sort_trees)
{
   std::vector<BranchedTree::Tree<T>>   trees_unique = this->_LoadFromJson(key, json_filename, bool_sort_trees);
   std::vector<BranchlessTree::Tree<T>> trees        = Branched2BranchlessTrees(trees_unique);

   // write to file for debug
   std::string filename = generated_files_path + "generated_forest.h";
   write_generated_code_to_file<T, BranchlessTree::Tree<T>>(trees, this->s_obj_func, filename);

   this->trees = JitTrees<T, BranchlessTree::Tree<T>>(trees, this->s_obj_func);
}

#endif
