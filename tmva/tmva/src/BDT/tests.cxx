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
#include <utility>

#include "TInterpreter.h" // for gInterpreter

#include "bdt_helpers.h"

#include "unique_bdt.h"
#include "array_bdt.h"
#include "forest.h"

#include "gtest/gtest.h"

//#include <xgboost/c_api.h> // for xgboost
//#include "generated_files/evaluate_forest2.h"

using json = nlohmann::json;

int square(int v)
{
   return v * v;
}

TEST(someTest, testOne)
{
   ASSERT_EQ(5, 5);
}
