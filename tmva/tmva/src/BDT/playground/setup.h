#ifndef __SETUP_BDT_H_
#define __SETUP_BDT_H_

#include "bdt_helpers.h"


std::string events_file     = "./data/events.csv";
std::string preds_file      = "./data/test.csv";
std::string json_model_file = "./data/model.json";

std::vector<std::vector<float>> events_vector = read_csv<float>(events_file);

std::vector<bool>               preds;


#endif
