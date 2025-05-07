#include "TClass.h"
#include "TClassRef.h"
#include "TDataMember.h"
#include "TGlobal.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TListOfDataMembers.h"
#include "TROOT.h"

#include "gtest/gtest.h"

TEST(TClingDataMemberInfo, UsingDecls)
{
   gInterpreter->Declare(R"CODE(
namespace UsingDecls {

struct EmptyStruct {
};

class Base {
protected:
  int protectedVar;
  void protectedFunc();
  void protectedFunc(float);

public:
  int func();
  float publicVar;
  static double publicStatic;
  void publicFunc();
};

class Derived: public Base {
private:
  using Base::func;

protected:
  using Base::publicStatic;

public:
  using Base::protectedVar;
  using Base::publicVar;
  using Base::protectedFunc;
};

class Rederived: public Derived {
  using Derived::publicStatic;
};

struct BagOfThings {
  union {
    int a;
    float b;
  };
  struct {
    short var;
  };
  enum {
    kEnumConst1 = 12,
    kEnumConst2 = 12
  };

  struct Nested {
    int doNotFindMe;
  };
};

struct DerivedBag: BagOfThings {
  using BagOfThings::a;
};

namespace EmptyNamespace {}

namespace SingleVarNamespace {
  int SingleVar;
}
namespace SingleUsingDeclNamespace {
  using SingleVarNamespace::SingleVar;
}
namespace WithInlineNamespace {
  inline namespace InlineNamespace {
    float InlineNamespaceVar;
    inline namespace AnotherInline {
      double InlineInlineNamespaceVar;
    }
    namespace Named {
      float InlineNamedNamespaceVar;
    }
    namespace {
      constexpr char InInlineAnon = 'A';
    }
  }
  namespace {
    Float16_t InAnon;
  }
}

namespace Split {
}
namespace Split {
  int One;
}
namespace Split {
  short Two;
}
namespace Split {
}
namespace Split {
  int Three;
}
namespace Split {
  extern short Two;
}
class Outer {
  struct {
    int InnerPrivate;
  };
public:
  struct {
    long InnerPublic;
  };
};
template <class T>
struct Tmplt {
protected:
  T fMember;
};
}
)CODE");

   struct VarInfo_t {
      const char *fName;
      const char *fType;
      long fPropertyToTest;
   };

   auto checkClass = [](const char *className, std::vector<VarInfo_t> varInfos) {
      TClass *cl = TClass::GetClass(className);
      ASSERT_NE(cl, nullptr) << "Cannot find class " << className;
      TListOfDataMembers *vars = (TListOfDataMembers *)cl->GetListOfDataMembers();
      TList allVars;
      allVars.AddAll(vars);
      allVars.AddAll(cl->GetListOfUsingDataMembers());

      EXPECT_EQ(allVars.GetSize(), varInfos.size()) << "for scope " << className << (allVars.ls(), "") << '\n';
      for (auto &&varInfo : varInfos) {
         TDataMember *dm = (TDataMember *)allVars.FindObject(varInfo.fName);
         ASSERT_NE(dm, nullptr);
         EXPECT_STREQ(dm->GetName(), varInfo.fName) << " for var " << varInfo.fName << " in class " << className;
         if (varInfo.fType) {
            EXPECT_STREQ(dm->GetTrueTypeName(), varInfo.fType)
               << " for var " << varInfo.fName << " in class " << className;
         }
         EXPECT_EQ(dm->Property() & varInfo.fPropertyToTest, varInfo.fPropertyToTest)
            << " for var " << varInfo.fName << " in class " << className;
      }
   };

   checkClass("UsingDecls::EmptyStruct", {});

   checkClass("UsingDecls::Base", {{"protectedVar", "int", kIsProtected},
                                   {"publicVar", "float", kIsPublic},
                                   {"publicStatic", "double", kIsPublic | kIsStatic}});

   checkClass("UsingDecls::Derived", {{"publicStatic", "double", kIsProtected | kIsStatic},
                                      {"protectedVar", "int", kIsPublic},
                                      {"publicVar", "float", kIsPublic}});

   checkClass("UsingDecls::Rederived", {{"publicStatic", "double", kIsPrivate}});

   checkClass("UsingDecls::BagOfThings", {{"a", "int", kIsPublic | kIsUnionMember},
                                          {"b", "float", kIsPublic | kIsUnionMember},
                                          {"var", "short", kIsPublic},
                                          {"kEnumConst1", nullptr, kIsEnum | kIsPublic},
                                          {"kEnumConst2", nullptr, kIsEnum | kIsPublic}});

   checkClass("UsingDecls::EmptyNamespace", {});

   checkClass("UsingDecls::SingleVarNamespace", {{"SingleVar", "int", kIsPublic}});

   checkClass("UsingDecls::SingleUsingDeclNamespace", {{"SingleVar", "int", kIsPublic}});

   checkClass("UsingDecls::WithInlineNamespace",
              {{"InlineNamespaceVar", "float", kIsPublic | kIsStatic},
               {"InlineInlineNamespaceVar", "double",
                kIsPublic | kIsStatic}, // LIMITATION: overrides earlier InlineNamespace::InlineInlineNamespaceVar
               {"InInlineAnon", "const char", kIsConstexpr},
               {"InAnon", "Float16_t", 0}});

   checkClass("UsingDecls::Split", {{"One", "int", 0}, {"Two", "short", 0}, {"Three", "int", 0}});

   checkClass("UsingDecls::Outer", {{"InnerPrivate", "int", kIsPrivate}, {"InnerPublic", "long", kIsPublic}});

   checkClass("UsingDecls::Tmplt<Double32_t>", {{"fMember", "Double32_t", kIsProtected}});
}


TEST(TClingDataMemberInfo, Lookup)
{
   gInterpreter->Declare(R"CODE(
namespace DMLookup {
class Outer {
  struct {
    TObject InnerPrivate;
  };
protected:
  struct {
    TNamed InnerProtected;
  };
};
})CODE");
   ASSERT_TRUE(TClass::GetClass("DMLookup::Outer"));
   ASSERT_TRUE(TClass::GetClass("DMLookup::Outer")->GetListOfDataMembers());

   ASSERT_TRUE(TClass::GetClass("DMLookup::Outer")->GetListOfDataMembers()->FindObject("InnerPrivate"));
   auto *dmInnerPrivate = (TDataMember*)TClass::GetClass("DMLookup::Outer")->GetListOfDataMembers()->FindObject("InnerPrivate");
   EXPECT_EQ(dmInnerPrivate->Property(), kIsPrivate | kIsClass | kIsNotReacheable);

   ASSERT_TRUE(TClass::GetClass("DMLookup::Outer")->GetListOfDataMembers()->FindObject("InnerProtected"));
   auto *dmInnerProtected = (TDataMember*)TClass::GetClass("DMLookup::Outer")->GetListOfDataMembers()->FindObject("InnerProtected");
   EXPECT_EQ(dmInnerProtected->Property(), kIsProtected | kIsClass | kIsNotReacheable);
}

TEST(TClingDataMemberInfo, Offset)
{
   gInterpreter->ProcessLine("int ROOT_7459 = 42; ROOT_7459++;");
   EXPECT_EQ(43, *(int*)gROOT->GetGlobal("ROOT_7459")->GetAddress());

   gInterpreter->ProcessLine("constexpr int var1 = 1;");
   EXPECT_EQ(1, *(int*)gROOT->GetGlobal("var1")->GetAddress());

   gInterpreter->ProcessLine("static constexpr int var2 = -2;");
   EXPECT_EQ(-2, *(int*)gROOT->GetGlobal("var2")->GetAddress());

   gInterpreter->ProcessLine("const float var3 = 3.1;");
   EXPECT_FLOAT_EQ(3.1, *(float*)gROOT->GetGlobal("var3")->GetAddress());

   gInterpreter->ProcessLine("static const double var4 = 4.2;");
   EXPECT_DOUBLE_EQ(4.2, *(double*)gROOT->GetGlobal("var4")->GetAddress());

   // Make sure ROOT's Core constexpr constants work
   EXPECT_EQ(3000, *(int*)gROOT->GetGlobal("kError")->GetAddress());

#if defined(R__USE_CXXMODULES) && defined(R__HAS_GEOM)
   // gGeoManager is defined in the Geom libraries and we want to make sure we
   // do not load it when autoloading is off. We can only test this in modules
   // mode because gGeoManager is not part of the PCH and non-modular ROOT has
   // header parsing and autoloading coupled leading to redundant load of
   // libGeom at gROOT->GetGlobal time.
   TGlobal *GeoManagerInfo = gROOT->GetGlobal("gGeoManager");
   TInterpreter::SuspendAutoLoadingRAII autoloadOff(gInterpreter);
   EXPECT_EQ(-1L, (ptrdiff_t)GeoManagerInfo->GetAddress());
#endif // R__USE_CXXMODULES and R__HAS_GEOM
}
