#ifndef __SETUP_BDT_H_
#define __SETUP_BDT_H_

#include "TreeHelpers.hxx"


std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/test.csv";
std::string json_model_file = "./data/model.json";

const std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);

std::vector<bool>               preds;


#endif
