#include <TSystem.h>
#include <TInterpreter.h>

#include "gtest/gtest.h"

#include <fstream>

TEST(DictGen, ExtraIncludes)
{
   auto macroName = "myextramacro.C";
   auto macroCode = R"MACROCODE(
#ifdef __ROOTCLING__
#pragma extra_include "myextra.h";
#endif

#include <iostream>

void a(){}
)MACROCODE";

    std::ofstream macroFile (macroName);
    macroFile << macroCode;
    macroFile.close();

   auto includeName = "myextra.h";
   auto extraInclude = R"EXTRAINCLUDE(
#ifndef __EXTRA_HEADER_H
#define __EXTRA_HEADER_H
int f(){return 42;};
#endif
)EXTRAINCLUDE";

    std::ofstream eiFile (includeName);
    eiFile << extraInclude;
    eiFile.close();


   // Here we perform these steps:
   // Compile macro which has a linkdef section with extra includes
   // The extra includes get parsed. They contain a function, f
   // The value returned by f is checked to be correct 
   gInterpreter->ProcessLine(".L myextramacro.C+");
   auto r = gInterpreter->ProcessLine("f()");
   EXPECT_EQ(r, 42U);

   gSystem->Unlink(macroName);
   gSystem->Unlink(includeName);
}