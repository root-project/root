#include "gtest/gtest.h"
#include "helpersBDTest.hxx"
#include "forestBDTest.hxx"
//#include "arrayBDTest.hxx"

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
