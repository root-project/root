#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TSystem.h"

#include <fstream>

TEST(TSystem, CompileMacroSimple)
{
   const auto srcFileName = "testmacro.C";

   std::ofstream srcFile;
   srcFile.open(srcFileName);
   srcFile << "int testmacro(){return 42;}";
   srcFile.close();

   gSystem->CompileMacro(srcFileName, "hs");
   ROOT_EXPECT_INFO(gSystem->CompileMacro(srcFileName), "ACLiC",
                    "unmodified script has already been compiled and loaded");
   ROOT_EXPECT_INFO(gSystem->CompileMacro(srcFileName, "h"), "ACLiC",
                    "The macro has been already built and loaded according to its checksum.");
   gSystem->Unlink(srcFileName);
}
