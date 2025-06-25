/// \file TClingUtilsTests.cxx
///
/// \brief The file contain unit tests which test the TClingUtils.h
///
/// \author Vassil Vassilev <vvasilev@cern.ch>
///
/// \date Aug, 2019
///
/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TClingUtils.h>
#include <TClass.h>
#include <TInterpreter.h>

#include <ROOT/FoundationUtils.hxx>

#include "gtest/gtest.h"

#include <fstream>
#include <deque>

TEST(TClingUtilsTests, GetCppName)
{
   using namespace ROOT::TMetaUtils;

   // Test for converting a string to a valid C/C++ variable name
   std::string validVarName;
   GetCppName(validVarName,
              "some+input-string*with/special&characters123+-*/&%|^><=~.()[]{};#?`!,$:\"@\'\\@abc$and spaces");
   ASSERT_EQ(validVarName, "somepLinputmIstringmUwithdIspecialaNcharacters123pLmImUdIaNpEoRhAgRlEeQwAdOoPcPoBcBlBrBsChS"
                           "qMbTnOcOdAcLdQaTsQfIaTabcdAandsPspaces");
}

TEST(TClingUtilsTests, GetRealPath)
{
#ifndef R__WIN32
   // Create a file in the current folder
   std::string cwd = ROOT::FoundationUtils::GetCurrentDir();

   ASSERT_FALSE(cwd.empty());

   std::ofstream file;
   file.open("./realfile1", std::ios::out);
   file.close();
   file.open("./realfile2", std::ios::out);
   file.close();
   // Create symlinks
   ASSERT_GE(symlink("./realfile1", "symlink_realfile1"), 0);
   ASSERT_GE(symlink((cwd + "realfile2").c_str(), "symlink_realfile2"), 0);

   // Recursive symlinks
   ASSERT_GE(symlink("./symlink_realfile1", "symlink1_symlink_realfile1"), 0);
   ASSERT_GE(symlink((cwd + "symlink_realfile2").c_str(), "symlink1_symlink_realfile2"), 0);

   ASSERT_GE(symlink("./symlink_realfile2", "symlink2_symlink_realfile2"), 0);
   ASSERT_GE(symlink((cwd + "symlink_realfile1").c_str(), "symlink2_symlink_realfile1"), 0);

   ASSERT_GE(symlink((cwd + ".//.///symlink_realfile1").c_str(), "symlink3_symlink_realfile1"), 0);

   using namespace ROOT::TMetaUtils;
   std::string realfile1 = GetRealPath("./realfile1");
   std::string realfile2 = GetRealPath("./realfile2");
   ASSERT_EQ(realfile1, GetRealPath("./symlink_realfile1"));
   ASSERT_EQ(realfile1, GetRealPath("./symlink_realfile1"));
   ASSERT_EQ(realfile1, GetRealPath("./symlink1_symlink_realfile1"));
   ASSERT_EQ(realfile1, GetRealPath("./symlink2_symlink_realfile1"));
   ASSERT_EQ(realfile1, GetRealPath("./symlink3_symlink_realfile1"));

   ASSERT_EQ(realfile2, GetRealPath("./realfile2"));
   ASSERT_EQ(realfile2, GetRealPath("./symlink_realfile2"));
   ASSERT_EQ(realfile2, GetRealPath("./symlink1_symlink_realfile2"));
   ASSERT_EQ(realfile2, GetRealPath("./symlink2_symlink_realfile2"));

   std::remove("./symlink3_symlink_realfile1");
   std::remove("./symlink2_symlink_realfile2");
   std::remove("./symlink2_symlink_realfile1");
   std::remove("./symlink1_symlink_realfile2");
   std::remove("./symlink1_symlink_realfile1");
   std::remove("./symlink_realfile1");
   std::remove("./symlink_realfile2");
   std::remove("./realfile1");
   std::remove("./realfile2");
#endif // not R__WIN32
}

TEST(TClingUtilsTests, CollectionSizeof)
{
   // https://its.cern.ch/jira/browse/ROOT-9889
   EXPECT_EQ(sizeof(std::deque<short>), TClass::GetClass("std::deque<short>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<unsigned short>), TClass::GetClass("std::deque<unsigned short>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<int>), TClass::GetClass("std::deque<int>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<unsigned int>), TClass::GetClass("std::deque<unsigned int>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<long>), TClass::GetClass("std::deque<long>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<unsigned long>), TClass::GetClass("std::deque<unsigned long>")->GetClassSize());
}

TEST(TClingUtilsTests, ReSubstTemplateArg)
{
   // #18811
   gInterpreter->Declare("template <typename T> struct S {};"
                         "template <typename T1, typename T2> struct Two { using value_type = S<T2>; };"
                         "template <typename T> struct One { Two<int, int>::value_type *t; };");

   auto c = TClass::GetClass("One<std::string>");
   c->BuildRealData();
}
