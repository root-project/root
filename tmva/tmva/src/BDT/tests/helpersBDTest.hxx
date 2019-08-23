#include "gtest/gtest.h"
#include "bdt_helpers.hxx"

TEST(helpersBDT, test_write_vector_vector)
{
   std::string filename     = "./data/events.csv";
   std::string tmp_filename = "./data/tmp.csv";

   std::vector<std::vector<float>> data = read_csv<float>(filename);
   write_csv<float>(tmp_filename, data);
   std::vector<std::vector<float>> data2 = read_csv<float>(tmp_filename);

   ASSERT_EQ(data.size(), data2.size());

   for (size_t i = 0; i < data.size(); i++) {
      for (size_t j = 0; j < data[0].size(); j++) {
         ASSERT_FLOAT_EQ(data[i][j], data2[i][j]);
      }
   }
}
