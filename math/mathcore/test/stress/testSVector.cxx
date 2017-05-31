// @(#)root/test:$Id$
// Author: Lorenzo Moneta       06/2005
//         Martin Storø Nyfløtt 05/2017

#include "gtest/gtest.h"

#include "StatFunction.h"
#include "TestHelper.h"
#include "VectorTest.h"

using namespace ROOT::Math;

//--------------------------------------------------------------------------------------
// test of generic Svector
//--------------------------------------------------------------------------------------

TEST(TestSMatrix, TestSVector)
{
   const int nGen = 10000;
   const int Dim = 6;
   // test the matrix if D2 is not equal to 1
   VectorTest<Dim> a(nGen);
   a.GenDataN();

   typedef SVector<double, Dim> SV;
   std::vector<SV> v1;
   v1.reserve(nGen);

   double s1 = 0;
   // double scale = 1;
   double sref1 = 0;

   std::string name = "SVector<double," + Util::ToString(Dim) + ">";

   a.TestCreateSV(v1);
   EXPECT_TRUE(IsNear(name + " creation", v1.size(), nGen, 1));
   s1 = a.TestAdditionSV(v1);
   EXPECT_TRUE(IsNear(name + " addition", s1, a.Sum(), Dim * 4));
   sref1 = s1;

   // test the io
   double fsize = 0;
   int ir = 0;

   std::string typeName = "ROOT::Math::" + name;
   double estSize = nGen * 8 * Dim + 10000;
   double scale = 0.1 / std::numeric_limits<double>::epsilon();
   fsize = a.TestWrite(v1, typeName);
   EXPECT_TRUE(IsNear(name + " write", fsize, estSize, scale));
   ir = a.TestRead(v1);
   EXPECT_TRUE(IsNear(name + " read", ir, 0, 1));
   s1 = a.TestAdditionSV(v1);
   EXPECT_TRUE(IsNear(name + " after read", s1, sref1, 1));
}
