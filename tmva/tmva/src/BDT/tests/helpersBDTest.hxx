#include "gtest/gtest.h"
#include "TreeHelpers.hxx"

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

TEST(helpersBDT, test_write_vector)
{
   std::string filename     = "./data/python_predictions.csv";
   std::string tmp_filename = "./data/tmp_write.csv";

   std::vector<std::vector<bool>> data = read_csv<bool>(filename);

   std::vector<bool> data2(data.size());
   for (int i = 0; i < data.size(); i++) {
      data2[i] = data[i][0];
   }
   write_csv<bool>(tmp_filename, data2);
   std::vector<std::vector<bool>> data3 = read_csv<bool>(tmp_filename);

   ASSERT_EQ(data.size(), data3.size());

   for (size_t i = 0; i < data.size(); i++) {
      ASSERT_EQ(data[i][0], data3[i][0]);
   }
}

template <typename T>
void test_write_vector_single()
{
   std::string filename     = "./data/events.csv";
   std::string tmp_filename = "./data/tmp.csv";

   std::vector<std::vector<float>> data = read_csv<float>(filename);

   std::vector<float> data2(data.size());
   for (int i = 0; i < data.size(); i++) {
      data2[i] = data[i][0];
   }

   write_csv<float>(tmp_filename, data2);
   std::vector<std::vector<float>> data3 = read_csv<float>(tmp_filename);

   ASSERT_EQ(data.size(), data3.size());
   for (size_t i = 0; i < data.size(); i++) {
      ASSERT_FLOAT_EQ(data[i][0], data3[i][0]);
   }
}
TEST(helpersBDT, test_write_vector_s)
{

   test_write_vector_single<float>();
   test_write_vector_single<double>();
   test_write_vector_single<long double>();
}
