#include <RooDataSet.h>
#include <RooLinkedList.h>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

// Covers https://github.com/root-project/root/issues/20904
TEST(RooLinkedList, InitializeHashTableForNonEmptyLinkedList)
{
   const int max_n = 100;

   for (int n = 1; n < max_n; ++n) {

      RooLinkedList dataList;

      for (int i = 0; i < n; ++i) {
         auto d = new RooDataSet{"dataset_" + std::to_string(i), "", {}};

         if (dataList.size() > 50 && dataList.getHashTableSize() == 0) {
            dataList.setHashTableSize(200);
         }
         dataList.Add(d);
      }

      auto found = dataList.FindObject("dataset_0");

      ASSERT_TRUE(found) << "dataset retrieval failed with " << n << " datasets";

      dataList.Delete();
   }
}
