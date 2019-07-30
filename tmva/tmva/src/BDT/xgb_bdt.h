#ifndef __XGB_BDT_H_
#define __XGB_BDT_H_

#include <iostream>
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>
#include <array>
#include <chrono>
#include <ctime>      // for date
#include <functional> // for std::fucntion

#include "bdt_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>

namespace xgb_bdt {

float *csv_to_array(std::string fname)
{
   std::vector<std::vector<float>> matrix_vector = read_csv(fname);

   int cols = matrix_vector[0].size();
   int rows = matrix_vector.size();

   float matric_array[rows][cols];
   for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) matric_array[i][j] = matrix_vector[i][j];

   return matric_array;
}

} // namespace xgb_bdt

#endif
