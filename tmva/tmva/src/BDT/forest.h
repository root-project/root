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
   std::vector<T> trees;
   static int     counter;

   Forest() { counter++; }
   ~Forest() {}

   // std::function<bool(float)> objective_func = binary_logistic; // this could be changed
   void write_forest_to_file();
   void test() { std::cout << "test \n"; }
   void get_Forest(std::string json_file = "aaa") { std::cout << json_file << std::endl; }
   void read_events_csv(std::string csv_file);

   // std::vector<bool> do_predictions(const std::vector<std::vector<float>> &events_vector);
   void do_predictions(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &);
};

// ----------------- Generic ------------------
template <class T>
void Forest<T>::read_events_csv(std::string csv_file)
{
   if (!csv_file.empty()) {
      this->events_vector = read_csv(events_file);
   } else {
      this->events_vector = read_csv(this->events_file);
   }
}

template <class T>
void Forest<T>::do_predictions(const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   float preds_tmp;
   preds.reserve(events_vector[0].size());
   for (auto &event : events_vector) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree.inference(event);
      }
      preds.push_back(binary_logistic(preds_tmp));
   }
}

template <class T>
void Forest<T>::write_forest_to_file()
{
   // this does nothing
}

template <class T>
int Forest<T>::counter = 0;

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
   this->trees = std::move(trees);
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
   this->trees = trees;
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
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);
   // std::cout << "current time used as namespace: " << s_namespace_name << std::endl;

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

   this->trees = function_vector;
}

template <>
void Forest<std::function<float(std::vector<float>)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   float preds_tmp = 0;
   for (auto &event : events_vector) {
      preds_tmp = 0;
      for (auto &tree : this->trees) {
         preds_tmp += tree(event);
      }
      preds.push_back(binary_logistic(preds_tmp));
   }
}

template <>
void Forest<std::function<float(std::vector<float>)>>::test()
{
   std::cout << "AAAAAA\n";
}

// ------------------------ Specialization FORESTJIT ------------------ //
template <>
void Forest<std::function<bool(std::vector<float>)>>::get_Forest(std::string json_file)
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
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);
   // std::cout << "current time used as namespace: " << s_namespace_name << std::endl;

   std::stringstream ss;
   generate_code_forest(ss, trees, number_of_trees, s_namespace_name);
   s_trees = ss.str();

   // write to file
   std::filebuf fb;
   std::string  filename = "./generated_files/generated_forest.h";
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest(os, trees, number_of_trees, s_namespace_name);
   fb.close();

   // Read functions
   std::function<bool(std::vector<float>)> func;
   func = jit_forest_string(s_trees, s_namespace_name);
   std::vector<std::function<bool(std::vector<float>)>> tmp{func};
   this->trees = tmp;
}

template <>
void Forest<std::function<bool(std::vector<float>)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   for (auto &event : events_vector) {
      preds.push_back(this->trees[0](event));
   }
}

// ------------------------ Specialization FORESTJIT EVENTS ------------------ //
template <>
void Forest<std::function<std::vector<bool>(std::vector<std::vector<float>>)>>::get_Forest(std::string json_file)
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
   std::string s_trees;
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(this->counter) + std::to_string(my_time);
   // std::cout << "current time used as namespace: " << s_namespace_name << std::endl;

   std::stringstream ss;
   generate_code_forest_batch(ss, trees, number_of_trees, s_namespace_name);
   s_trees = ss.str();

   // write to file
   std::filebuf fb;
   std::string  filename = "./generated_files/evaluate_forest.h";
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest_batch(os, trees, number_of_trees, s_namespace_name);
   fb.close();

   // Read functions
   std::function<std::vector<bool>(std::vector<std::vector<float>>)> func;
   func = jit_event_forest_string(s_trees, s_namespace_name);
   // std::function<std::vector<bool>(std::vector<std::vector<float>>)> tmp{func};
   this->trees.push_back(func);
}

template <>
void Forest<std::function<std::vector<bool>(std::vector<std::vector<float>>)>>::do_predictions(
   const std::vector<std::vector<float>> &events_vector, std::vector<bool> &preds)
{
   preds = std::move(this->trees[0](events_vector));
}

#endif
// End file
