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
//#include "jitted_bdt.h"

using json = nlohmann::json;


// See how to specialize from different base classes
template <class T>
class TreeWrapper{
public:

  std::string events_file = "./data_files/events.csv";

  void test(){std::cout << "test \n";}
  void get_Forest(std::string json_file="aaa"){
    std::cout << json_file << std::endl;
  }

  void read_events_csv(std::string csv_file=""){
    if (!csv_file.empty()){
      this->events_vector=read_csv(events_file);
    }
    else{
      this->events_vector=read_csv(this->events_file);
    }

  }
  std::vector<T> Forest;

  std::vector<bool>
  do_predictions(std::vector<std::vector<float>> events_vector){

    //preds.clear();
    std::vector<bool> preds;
    float prediction = 0;
    std::vector<float> preds_tmp;

    preds.reserve(events_vector.size());
    for (auto &event : events_vector){
      preds_tmp.clear();
      for (auto & tree : this->Forest){
        prediction = tree.inference(event);
        preds_tmp.push_back(prediction);
      }
      preds.push_back(binary_logistic(vec_sum(preds_tmp)));
    }
    return preds;
  }

};



/*
template <>
class TreeWrapper<unique_bdt::Tree>{
public:

  void get_Forest(std::string json_file="model.json"){
    std::string my_config = read_file_string(json_file);
    auto json_model = json::parse(my_config);
    int number_of_trees = json_model.size();
    //unique_bdt::Tree trees[number_of_trees];

    std::vector<unique_bdt::Tree> trees;
    trees.resize(number_of_trees);


    for (int i=0; i<number_of_trees; i++){
      unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
    }
    this->Forest=std::move(trees);
  }
  std::vector<unique_bdt::Tree> Forest;
};
*/

// ------------------ Specialization unique_ptr -------------------- //
template<>
void TreeWrapper<unique_bdt::Tree>::get_Forest(std::string json_file)
{
  std::string my_config = read_file_string(json_file);
  auto json_model = json::parse(my_config);
  int number_of_trees = json_model.size();

  std::vector<unique_bdt::Tree> trees;
  trees.resize(number_of_trees);


  for (int i=0; i<number_of_trees; i++){
    unique_bdt::read_nodes_from_tree(json_model[i], trees[i]);
  }
  this->Forest=std::move(trees);
}

// ------------------------ Specialization array ------------------ //
template<>
void TreeWrapper<array_bdt::Tree>::get_Forest(std::string json_file)
{
  std::string my_config = read_file_string(json_file);
  auto json_model = json::parse(my_config);
  int number_of_trees = json_model.size();

  std::vector<array_bdt::Tree> trees;
  trees.resize(number_of_trees);

  for (int i=0; i<number_of_trees; i++){
    array_bdt::read_nodes_from_tree(json_model[i], trees[i]);
  }
  this->Forest=trees;
}




#endif
// End file
