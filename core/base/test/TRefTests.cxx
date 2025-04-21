#include "gtest/gtest.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "TStreamerElement.h"
#include "TVirtualStreamerInfo.h"

#include <ROOT/TestSupport.hxx>

// See ROOT-7052
TEST(TRef, Exec)
{

   // Needed because we do not generate dictionaries and we do no not want to see warnings
   ROOT::TestSupport::CheckDiagsRAII checkDiag;
   checkDiag.requiredDiag(kWarning, "TStreamerInfo::Build", "MyClass: ", /*matchFullMessage=*/false);

   gInterpreter->ProcessLine("class Foo1 : public TRef {\n"
                             "   int i;\n"
                             "ClassDef(Foo1, 1);\n"
                             "};"
                             "class Foo2 : public TRefArray {\n"
                             "   int i;\n"
                             "ClassDef(Foo2, 1);\n"
                             "};"
                             "class TRefFoo {int i;};\n"
                             "class Foo3 {int i;};\n"
                             " class MyClass {\n"
                             "    Foo1 m1; // EXEC:GetFoo\n"
                             "    Foo2 m2; // EXEC:GetFoo\n"
                             "    TRefFoo m3; // EXEC:GetFoo\n"
                             "    Foo3 m4; // EXEC:GetFoo\n"
                             "};");

   auto c = TClass::GetClass("MyClass");
   auto si = c->GetStreamerInfo();
   int o;
   auto se1 = si->GetStreamerElement("m1", o);
   auto se2 = si->GetStreamerElement("m2", o);
   auto se3 = si->GetStreamerElement("m3", o);
   auto se4 = si->GetStreamerElement("m4", o);

   EXPECT_EQ(1, se1->GetExecID());
   EXPECT_EQ(1, se2->GetExecID());
   EXPECT_EQ(0, se3->GetExecID());
   EXPECT_EQ(0, se4->GetExecID());
}