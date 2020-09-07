#include "TClass.h"
#include "TClassRef.h"
#include "TGlobal.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TListOfFunctions.h"
#include "TMethod.h"
#include "TROOT.h"

#include "gtest/gtest.h"

TEST(TClingMethodInfo, Prototype)
{
  TClass *cl = TClass::GetClass("TObject");
  ASSERT_NE(cl, nullptr);
  TMethod *meth = (TMethod*)cl->GetListOfMethods()->FindObject("SysError");
  ASSERT_NE(meth, nullptr);
  EXPECT_STREQ(meth->GetPrototype(), "void TObject::SysError(const char* method, const char* msgfmt,...) const");
}

TEST(TClingMethodInfo, ROOT10789)
{
  gInterpreter->Declare(R"CODE(
namespace ROOT10789 {

struct EmptyStruct {
  /*
  EmptyStruct();
  EmptyStruct(const EmptyStruct &);
  EmptyStruct(EmptyStruct &&);
  EmptyStruct & operator=(const EmptyStruct &);
  EmptyStruct & operator=(EmptyStruct &&);
  ~EmptyStruct();
  */
};

class Base {
protected:
  int protectedVar;
  void protectedFunc();
  void protectedFunc(float);

public:
  int func();
  int publicVar;
  void publicFunc();
  void publicFunc(int);

  Base(int);
  /*
  Base(const Base &);
  Base(Base &&);
  Base &operator=(const Base &);
  Base &operator=(Base &&);
  ~Base();
  */
};

class Derived: public Base {
private:
  using Base::publicFunc; // Base::publicFunc(), Base::publicFunc(int)
protected:
  using Base::func; // Base::func()
public:
  Derived() = delete;
  Derived(float);
  using Base::Base; // Base::Base(int); SKIPPED: Base::Base(const Base &); Base::Base(Base &&); 

  using Base::protectedVar;
  using Base::publicVar;
  using Base::protectedFunc; // Base::protectedFunc(); Base::protectedFunc(float);
  /*
  Derived(const Derived &);
  Derived(Derived &&);
  Derived &operator=(const Derived &);
  Derived &operator=(Derived &&);
  ~Derived();
  */
};

class Rederived: public Derived {
  using Derived::protectedFunc;
};

struct BagOfThings {
  union {
    int a;
    float b;
  };
  struct {
    int var;
  };
  enum {
    kEnumConst = 12
  };

  struct Nested {
    void doNotFindMe();
  };

protected:
  int Deleted() = delete;
};

namespace EmptyNamespace {}

namespace SingleFuncNamespace {
  int SingleFunc();
}
namespace SingleUsingDeclNamespace {
  using SingleFuncNamespace::SingleFunc;
}
namespace WithInlineNamespace {
  inline namespace InlineNamespace {
    void InlineNamespaceFunc();
    inline namespace AnotherInline {
      void InlineInlineNamespaceFunc();
    }
    namespace Named {
      void InlineNamedNamespaceFunc();
    }
    namespace {
      int InAnon();
    }
  }
  namespace {
    int InAnon(int);
  }
}

namespace Split {
}
namespace Split {
  int One();
}
namespace Split {
  int Two();
}
namespace Split {
}
namespace Split {
  int Three();
}
namespace Split {
  int Two(int);
}
}
)CODE");

  struct FuncInfo_t{
    const char* fName;
    int fNOverloads;
    long fPropertyToTest;
  };

  auto checkClass = [](const char* className, int nMethods, std::vector<FuncInfo_t> funcInfos)
    {
      TClass *cl = TClass::GetClass(className);
      ASSERT_NE(cl, nullptr);
      TListOfFunctions *methods = (TListOfFunctions *) cl->GetListOfMethods();
      EXPECT_EQ(methods->GetSize(), nMethods)
        << "for scope " << className << (methods->ls(), "") << '\n';
      for (auto &&funcInfo: funcInfos) {
        TList *overloads = methods->GetListForObject(funcInfo.fName);
        ASSERT_NE(overloads, nullptr);
        EXPECT_EQ(overloads->GetSize(), funcInfo.fNOverloads)
          << "for " << className << "::" << funcInfo.fName << (overloads->ls(), "") << '\n';
        for (auto meth: TRangeDynCast<TMethod>(overloads))
          EXPECT_EQ(meth->Property() & funcInfo.fPropertyToTest, funcInfo.fPropertyToTest) << "for method " << meth->GetPrototype()
            << " in class " << className;
      }
    };

  checkClass("ROOT10789::EmptyStruct", 6,
    {{"EmptyStruct", 3, kIsPublic}, {"~EmptyStruct", 1, kIsPublic}});

  checkClass("ROOT10789::Base", 11,
    {{"func", 1, kIsPublic}, {"protectedFunc", 2, kIsProtected}, {"Base", 3, kIsPublic}});

  checkClass("ROOT10789::Derived", 12,
    {{"publicFunc", 2, kIsPrivate}, {"protectedFunc", 2, kIsPublic}, {"publicFunc", 2, kIsPrivate},
     {"Derived", 7 /*includes *all* ctors, even base-class-used*/, kIsPublic}});

  checkClass("ROOT10789::Rederived", 7,
    {{"protectedFunc", 2, kIsPrivate}, {"Rederived", 3, kIsPublic}});

  checkClass("ROOT10789::BagOfThings", 6, {});

  checkClass("ROOT10789::EmptyNamespace", 0, {});

  checkClass("ROOT10789::SingleFuncNamespace", 1, {{"SingleFunc", 1, 0}});

  checkClass("ROOT10789::SingleUsingDeclNamespace", 1, {{"SingleFunc", 1, 0}});

  checkClass("ROOT10789::WithInlineNamespace", 4,
    {{"InlineNamespaceFunc", 1, 0}, {"InlineInlineNamespaceFunc", 1, 0}, {"InAnon", 2, 0}});

  checkClass("ROOT10789::Split", 4,
    {{"One", 1, 0}, {"Two", 2, 0}, {"Three", 1, 0}});
}

TEST(TClingMethodInfo, DerivedCtorROOT11010) {

  gInterpreter->Declare(R"CODE(
int ROOT11010baseMemCtorCalled = 0;
int ROOT11010baseMemDtorCalled = 0;
int ROOT11010baseCtorCalled = 0;
int ROOT11010baseDtorCalled = 0;
int ROOT11010derivMemCtorCalled = 0;
int ROOT11010derivMemDtorCalled = 0;
int ROOT11010derivDtorCalled = 0;

namespace ROOT11010 {
  struct BaseMem {
    int mem[4] = {17, 17, 17, 17};
    BaseMem() { ++ROOT11010baseMemCtorCalled; }
    ~BaseMem() { ++ROOT11010baseMemDtorCalled; }
  };
  class Base {
    BaseMem memB;
  public:
    Base(std::string s) {
      ++ROOT11010baseCtorCalled;
      if (s != "TheStringCtorArg") {
	       std::cerr << "THE CTOR ARG IS WRONG!" << s << std::endl;
         throw 8714659826;
      }
    }
    virtual ~Base() {
      ++ROOT11010baseDtorCalled;
    }
  };

  struct DerivMem {
    double mem[3] = {42., 42., 42. };
    DerivMem() { ++ROOT11010derivMemCtorCalled; }
    ~DerivMem() { ++ROOT11010derivMemDtorCalled; }
  };

  class Derived: public Base {
    DerivMem memD;
  public:
    using Base::Base;
    ~Derived() override {
      ++ROOT11010derivDtorCalled;
    }
  };
}
)CODE");

  TClassRef c("ROOT11010::Derived");
  ASSERT_TRUE(c) << "Cannot find TClass for ROOT11010::Derived";

  TFunction *f = (TFunction*)c->GetListOfMethods()->FindObject("Derived");
  ASSERT_TRUE(f) << "Cannot find constructor for ROOT11010::Derived";

  CallFunc_t* callf = gInterpreter->CallFunc_Factory();
  MethodInfo_t* meth = gInterpreter->MethodInfo_Factory(f->GetDeclId());
  gInterpreter->CallFunc_SetFunc(callf, meth);
  gInterpreter->MethodInfo_Delete(meth);
  const TInterpreter::CallFuncIFacePtr_t &faceptr = gInterpreter->CallFunc_IFacePtr(callf);
  gInterpreter->CallFunc_Delete(callf);   // does not touch IFacePtr


  auto GetInt = [](const char* name) {
    auto *globals = gROOT->GetListOfGlobals();
    std::string fullname("ROOT11010");
    fullname += name;
    return *(int*)((TGlobal*)globals->FindObject(fullname.c_str()))->GetAddress();
  };

  EXPECT_EQ(GetInt("baseMemCtorCalled"), 0);
  EXPECT_EQ(GetInt("baseMemDtorCalled"), 0);
  EXPECT_EQ(GetInt("baseCtorCalled"), 0);
  EXPECT_EQ(GetInt("baseDtorCalled"), 0);

  EXPECT_EQ(GetInt("derivMemCtorCalled"), 0);
  EXPECT_EQ(GetInt("derivMemDtorCalled"), 0);
  EXPECT_EQ(GetInt("derivDtorCalled"), 0);

  void* argbuf[8];
  void* objresult = nullptr;
  std::string s("TheStringCtorArg");
  argbuf[0] = &s;
  faceptr.fGeneric(0, 1, argbuf, &objresult);
  EXPECT_NE(objresult, nullptr);

  EXPECT_EQ(GetInt("baseMemCtorCalled"), 1);
  EXPECT_EQ(GetInt("baseMemDtorCalled"), 0);
  EXPECT_EQ(GetInt("baseCtorCalled"), 1);
  EXPECT_EQ(GetInt("baseDtorCalled"), 0);

  EXPECT_EQ(GetInt("derivMemCtorCalled"), 1);
  EXPECT_EQ(GetInt("derivMemDtorCalled"), 0);
  EXPECT_EQ(GetInt("derivDtorCalled"), 0);

  c->Destructor(objresult);

  EXPECT_EQ(GetInt("baseMemCtorCalled"), 1);
  EXPECT_EQ(GetInt("baseMemDtorCalled"), 1);
  EXPECT_EQ(GetInt("baseCtorCalled"), 1);
  EXPECT_EQ(GetInt("baseDtorCalled"), 1);

  EXPECT_EQ(GetInt("derivMemCtorCalled"), 1);
  EXPECT_EQ(GetInt("derivMemDtorCalled"), 1);
  EXPECT_EQ(GetInt("derivDtorCalled"), 1);
}
