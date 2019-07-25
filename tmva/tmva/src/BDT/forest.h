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

// See how to specialize from different base classes
template <class T>
class Forest {
public:
   std::string    events_file = "./data_files/events.csv";
   std::vector<T> Forest;

   std::function<bool(float)> objective_func = binary_logistic; // this could be changed

   void test() { std::cout << "test \n"; }
   void get_Forest(std::string json_file = "aaa") { std::cout << json_file << std::endl; }
   void read_events_csv(std::string csv_file = "")
   {
      if (!csv_file.empty()) {
         this->events_vector = read_csv(events_file);
      } else {
         this->events_vector = read_csv(this->events_file);
      }
   }
   std::vector<bool> do_predictions(std::vector<std::vector<float>> events_vector)
   {

      // preds.clear();
      std::vector<bool>  preds;
      float              prediction = 0;
      std::vector<float> preds_tmp;

      preds.reserve(events_vector.size());
      for (auto &event : events_vector) {
         preds_tmp.clear();
         for (auto &tree : this->Forest) {
            prediction = tree.inference(event);
            preds_tmp.push_back(prediction);
         }
         preds.push_back(this->objective_func(vec_sum(preds_tmp)));
      }
      return preds;
   }
};

// ------------------ Specialization unique_ptr -------------------- //
template <>
void Forest<unique_bdt::Tree>::get_Forest(std::string json_file)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<unique_bdt::Tree> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }
   this->Forest = std::move(trees);
}

// ------------------------ Specialization array ------------------ //
template <>
void Forest<array_bdt::Tree>::get_Forest(std::string json_file)
{
   std::string my_config       = read_file_string(json_file);
   auto        json_model      = json::parse(my_config);
   int         number_of_trees = json_model.size();

   std::vector<array_bdt::Tree> trees;
   trees.resize(number_of_trees);

   for (int i = 0; i < number_of_trees; i++) {
      array_bdt::read_nodes_from_tree(json_model[i], trees[i]);
   }
   this->Forest = trees;
}

// ------------------------ Specialization Jitted ------------------ //
template <>
void Forest<std::function<float(std::vector<float>)>>::get_Forest(std::string json_file)
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

   // JIT
   std::vector<std::string> s_trees;
   s_trees.resize(number_of_trees);
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(my_time);
   std::cout << "current time used as namespace: " << s_namespace_name << std::endl;

   for (int i = 0; i < number_of_trees; i++) {
      std::stringstream ss;
      generate_code_bdt(ss, trees[i], i, s_namespace_name);
      s_trees[i] = ss.str();
   }

   // Read functions
   std::function<float(std::vector<float>)>              func;
   std::vector<std::function<float(std::vector<float>)>> function_vector;
   for (int i = 0; i < number_of_trees; i++) {
      func = jit_function_reader_string(i, s_trees[i], s_namespace_name);
      function_vector.push_back(func);
   }

   this->Forest = function_vector;
}

template <>
std::vector<bool> Forest<std::function<float(std::vector<float>)>>::do_predictions(
   std::vector<std::vector<float>> events_vector)
{
   // preds.clear();
   std::vector<bool>  preds;
   std::vector<float> preds_tmp;
   preds_tmp.reserve(this->Forest.size());
   preds.reserve(events_vector.size());
   for (auto &event : events_vector) {
      preds_tmp.clear();
      for (auto &tree : this->Forest) {
         preds_tmp.push_back(tree(event));
      }
      preds.push_back(binary_logistic(vec_sum(preds_tmp)));
   }
   return preds;
}

#endif
// End file
