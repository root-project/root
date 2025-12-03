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

// This test must under no circumstances include other ROOT headers. It must not call into Cling, TInterpreter, TClass.
// It is meant for unit testing of TClingUtils and statically links ClingUtils and Cling/Clang/LLVM libraries. If the
// test was to do any of the above, there would be two copies of functions around (in libCling.so and the test binary)
// with not much guarantees which ones are called at which moment.

#include <DllImport.h>
#include <ROOT/FoundationUtils.hxx>

#include "gtest/gtest.h"

#include <fstream>

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

// Forward-declare gCling to not include TInterpreter.h just for checking that the interpreter has not been initialized.
class TInterpreter;
R__EXTERN TInterpreter *gCling;

class InterpreterCheck : public testing::Environment {
   void TearDown() override { ASSERT_EQ(gCling, nullptr); }
};

testing::Environment *gInterpreterCheck = testing::AddGlobalTestEnvironment(new InterpreterCheck);
