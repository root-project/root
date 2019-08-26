#include "gtest/gtest.h"
#include "helpersBDTest.hxx"
#include "forestBDTest.hxx"
//#include "arrayBDTest.hxx"

int main(int argc, char **argv)
{
   // std::srand(0);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
