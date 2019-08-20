#ifndef __FOREST_H_
#define __FOREST_H_

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>
#include <array>
#include <utility>

#include "unique_bdt.h"
#include "array_bdt.h"
#include "jitted_bdt.h"
#include "bdt_helpers.h"

#include "TInterpreter.h" // for gInterpreter

using json = nlohmann::json;

std::string generated_files_path = "generated_files/";

bool unique_cmp(const unique_bdt::Tree &a, const unique_bdt::Tree &b)
{
   if (a.nodes->split_variable == b.nodes->split_variable) {
      return a.nodes->split_threshold < b.nodes->split_threshold;
   } else {
      return a.nodes->split_variable < b.nodes->split_variable;
   }
}

bool unique_cmp_feats(const unique_bdt::Tree &a, const unique_bdt::Tree &b)
{
   return a.nodes->split_variable < b.nodes->split_variable;
}


bool array_cmp(const array_bdt::Tree &a, const array_bdt::Tree &b)
{
   if (a.features[0] == b.features[0]) {
      return a.thresholds[0] < b.thresholds[0];
   } else {
      return a.features[0] < b.features[0];
   }
}

///////////////////////////////////////////////////
/// Wrapping class that contains the forest.
/// It support different kind of specialization, one for each forest kind
/// Kinds: array, unique, JIT, ForestsJIT, ForestJITWhole
///////////////////////////////////////////////////
template <class T>
class Forest {
public:                                                                /// TODO: change to private
   std::function<bool(float)> objective_func = binary_logistic<float>; // Default objective function
public:
   std::string    events_file = "./data_files/events.csv";
   std::vector<T> trees;
   static int     counter;

   Forest() { counter++; }
   ~Forest() {}

   void get_Forest(std::string json_file, bool bool_sort_trees = false);
   void get_Forest(std::string json_file, const std::vector<std::vector<float>> &events_vector);

   void read_events_csv(std::string csv_file);

   void do_predictions(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &);
   void do_predictions(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds,
                              int loop_size);
   // Remove before the last implementations
   void do_predictions_batch(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds,
                             int loop_size);
};

////////////////////////////////////////////////////
/// ----- Non specialized definitions -----
template <class T>
int Forest<T>::counter = 0;

template <class T>
void Forest<T>::read_events_csv(std::string csv_file)
{
   if (!csv_file.empty()) {
      this->events_vector = read_csv<float>(events_file);
   } else {
      this->events_vector = read_csv<float>(this->events_file);
   }
}

/// Default do_predictions (unique representation)
template <class T>
void Forest<T>::do_predictions(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   float preds_tmp;
   for (auto &event : events_vector) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(event);
      }
      preds.push_back(this->objective_func(preds_tmp));
   }
}

template <class T>
void Forest<T>::do_predictions_batch(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds,
                                     int loop_size)
{
   int rest = events_vector.size() % loop_size;

   float preds_tmp;
   int   index = 0;
   for (; index < events_vector.size() - rest; index += loop_size) {
      for (int j = index; j < index + loop_size; j++) {
         preds_tmp = 0;
         for (auto &tree : this->trees) {
            preds_tmp += tree.inference(events_vector[j]);
         }
         preds.push_back(this->objective_func(preds_tmp));
      }
   }
   // reminder loop
   for (int j = index; j < events_vector.size(); j++) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(events_vector[j]);
      }
      preds.push_back(this->objective_func(preds_tmp));
   }
}

template <class T>
void Forest<T>::do_predictions(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds,
                                      const int loop_size)
{
   int rest = events_vector.size() % loop_size;

   int   index     = 0;
   int   num_trees = this->trees.size();
   float preds_tmp = 0;

   float *preds_tmp_arr = new float[loop_size]{0};

   for (; index < events_vector.size() - rest; index += loop_size) {
      for (int i = 0; i < num_trees; i++) {
         for (int j = index; j < index + loop_size; j++) {
            preds_tmp_arr[j - index] += trees[i].inference(events_vector.at(j));
         }
      }
      for (int j = 0; j < loop_size; j++) {
         preds.push_back(this->objective_func(preds_tmp_arr[j]));
         // preds_tmp_arr[j] = this->objective_func(preds_tmp_arr[j]);
         preds_tmp_arr[j] = 0;
      }
      // std::copy(preds_tmp_arr, preds_tmp_arr + loop_size, std::back_inserter(preds));
   }
   // rest loop
   for (int j = index; j < events_vector.size(); j++) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(events_vector[j]);
      }
      preds.push_back(this->objective_func(preds_tmp));
   }
   delete[] preds_tmp_arr;
}

////////////////////////////////////////////////////////////////
/// ----------- Specialization unique_ptr ------------------- //
/// Unique pointer representation of the Forest
template <>
void Forest<unique_bdt::Tree>::get_Forest(std::string json_file, bool bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   // std::cout << "### trees: " << number_of_trees << std::endl;

   std::vector<unique_bdt::Tree> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }

   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), unique_cmp);
   }

   // for (int i = 0; i < number_of_trees; i++) {
   //    std::cout << trees[i].nodes->split_variable << " :  " << trees[i].nodes->split_threshold << std::endl;
   //}

   this->trees = std::move(trees);
}

////////////////////////////////////////////////////////////////
/// ---------------- Specialization array ------------------- //
template <>
void Forest<array_bdt::Tree>::get_Forest(std::string json_file, bool bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<array_bdt::Tree> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      array_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }

   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), array_cmp);
   }

   //for (int i = 0; i < number_of_trees; i++) {
    //   std::cout << trees[i].features[0] << " :  " << trees[i].thresholds[0] << std::endl;
   //}

   this->trees = trees;
}

////////////////////////////////////////////////////////////////
/// ----------- Specialization JIT Forest ------------------- //
template <>
void Forest<std::function<bool(const std::vector<float>&)>>::get_Forest(std::string json_file, bool bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   // create tmp unique trees
   std::vector<unique_bdt::Tree> trees;
   trees.resize(number_of_trees);
   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }
   if (bool_sort_trees == true) {
      std::sort(trees.begin(), trees.end(), unique_cmp);
   }

   // JIT
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);
   // std::cout << "current time used as namespace: " << s_namespace_name << std::endl;

   std::stringstream ss;
   generate_code_forest(ss, trees, number_of_trees, s_namespace_name);
   s_trees = ss.str();

   // write to file for debug
   std::filebuf fb;
   std::string  filename;
   if (bool_sort_trees == true)
     filename = generated_files_path+"generated_ordered_forest.h";
   else
    filename = generated_files_path+"generated_forest.h";

   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest(os, trees, number_of_trees, s_namespace_name);
   fb.close();

   // JIT functions
   std::function<bool(const std::vector<float>&)> func;
   func = jit_forest_string(s_trees, s_namespace_name);
   this->trees.push_back(func);
}

template <>
void Forest<std::function<bool(const std::vector<float>&)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   for (auto &event : events_vector) {
      preds.push_back(this->trees[0](event));
   }
}

////////////////////////////////////////////////////////////////
/// ---------- Specialization JIT Forest&events ------------- //
template <>
void Forest<std::function<std::vector<bool>(std::vector<std::vector<float>>)>>::get_Forest(std::string json_file,
                                                                                           bool        bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   // create tmp unique trees
   std::vector<unique_bdt::Tree> trees;
   trees.resize(number_of_trees);
   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }

   // Generate code
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);

   std::stringstream ss;
   generate_code_forest_batch(ss, trees, number_of_trees, s_namespace_name);
   s_trees = ss.str();

   // write to file for debug
   std::filebuf fb;
   std::string  filename = generated_files_path+"/evaluate_forest.h";
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest_batch(os, trees, number_of_trees, s_namespace_name);
   fb.close();

   // JIT functions
   std::function<std::vector<bool>(std::vector<std::vector<float>>)> func;
   func = jit_event_forest_string(s_trees, s_namespace_name);
   this->trees.push_back(func);
}

template <>
void Forest<std::function<std::vector<bool>(std::vector<std::vector<float>>)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   // preds = std::move(this->trees[0](events_vector));
   preds = this->trees[0](events_vector);
}

////////////////////////////////////////////////////////////////
/// ---------- Specialization JIT Forest&events&batch ------------- //
template <>
void Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>>::get_Forest(
   std::string json_file, const std::vector<std::vector<float>> &events_vector)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   // create tmp unique trees
   std::vector<unique_bdt::Tree> trees;
   trees.resize(number_of_trees);
   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }

   // Generate code
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);

   std::stringstream ss;
   generate_code_forest_batch_array(ss, trees, number_of_trees, events_vector.size(), s_namespace_name);
   s_trees = ss.str();

   // write to file for debug
   std::filebuf fb;
   std::string  filename = generated_files_path+"evaluate_forest_batch.h";
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest_batch_array(os, trees, number_of_trees, events_vector.size(), s_namespace_name);
   fb.close();

   // JIT functions
   std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)> func;
   func = jit_event_forest_string_batch(s_trees, s_namespace_name);
   this->trees.push_back(func);
}

template <>
void Forest<std::function<void(const std::vector<std::vector<float>> &, std::vector<bool> &)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   this->trees[0](events_vector, preds);
}
// */


////////////////////////////////////////////////////////////////
/// ---------- Specialization JIT branchless ------------- //
template <>
void Forest<std::function<bool(const float *)>>::get_Forest(
   std::string json_file, bool bool_sort_trees)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();


   // Read array_trees
   std::vector<array_bdt::Tree> trees;
   trees.resize(number_of_trees);
   for (int i = 0; i < number_of_trees; i++)
      array_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   if (bool_sort_trees == true)
      std::sort(trees.begin(), trees.end(), array_cmp);

   // Generate code
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);

   std::stringstream ss;
   generate_code_branchless_forest(ss, trees, number_of_trees, s_namespace_name);
   s_trees = ss.str();

   // write to file for debug
   std::filebuf fb;
   std::string  filename = generated_files_path+"branchless_generated_forest.h";
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_branchless_forest(os, trees, number_of_trees, s_namespace_name);
   fb.close();

   // JIT functions
   std::function<bool(const float *)> func;
   func = jit_branchless_forest(s_trees, s_namespace_name);
   this->trees.push_back(func);
}

template <>
void Forest<std::function<bool(const float *)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
 {
    for (auto &event : events_vector) {
       preds.push_back(this->trees[0](event.data()));
    }
 }
// */

#endif
// End file
