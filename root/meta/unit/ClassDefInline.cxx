#include "gtest/gtest.h"

#include <TInterpreter.h>

#define GetFromInterp(TYPE, WHAT) \
   (TYPE)gInterpreter->ProcessLine("ROOT6284::" #WHAT ";")

TEST(ClassDefInline, Basics) {

   gInterpreter->Declare("#line 12 \"AFileName.cxx\"\n"
                         "class ROOT6284: public TObject {\n"
                         "ClassDefInline(ROOT6284, 42); };");

   EXPECT_STREQ("ROOT6284", GetFromInterp(const char*, Class_Name()));
   EXPECT_STREQ("AFileName.cxx", GetFromInterp(const char*, DeclFileName()));
   EXPECT_EQ(13, GetFromInterp(int, DeclFileLine()));
   EXPECT_EQ(nullptr, GetFromInterp(const char*, ImplFileName()));
   EXPECT_EQ(-1, GetFromInterp(int, ImplFileLine()));
   EXPECT_EQ(42, GetFromInterp(int, Class_Version()));
};
