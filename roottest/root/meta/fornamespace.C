namespace MySpace {
   class MyClass {
   public:
      int a;
      MyClass() : a(0) {}
   };
   int funcInNS() { return 42; }
}

void fornamespace () {};
int globalFunc() { return 43; }
int globalFuncInline();
inline int globalFuncInline() { return 44; }

#include "TFile.h"
#include "TClass.h"

namespace Outer {
   template <typename T> struct InOuter {
      InOuter(): Storage(-42.), Also() {}
      long Storage;
      T Also;
   };
   namespace Second {
      template <typename T> struct InSecond {
         InOuter<T> I;
      };
   };
   namespace Inner {
      template <typename T> struct InInner {
         Second::InSecond<T> S;
         InOuter<int> O;
      };
      struct Cl {
         InInner<int> I;
         Inner::InInner<int> II;
         InInner<InInner<int> > I_I;
         Inner::InInner<Inner::InInner<int> > II_II;
         Outer::Inner::InInner<int> OII;
         Outer::Inner::InInner<Outer::Inner::InInner<int> > OII_OII;

         InOuter<int> O;
         Outer::InOuter<int> OO;
         InOuter<InOuter<int> > O_O;
         Outer::InOuter<Outer::InOuter<int> > OO_OO;
         Outer::InOuter<InInner<int> > OO_I;
         Outer::InOuter<Inner::InInner<int> > OO_II;
         Outer::InOuter<Outer::Inner::InInner<int> > OO_OII;

         Second::InSecond<int> S;
         Outer::Second::InSecond<int> OS;
         Outer::Second::InSecond<Second::InSecond<int> > OS_S;
         Outer::Second::InSecond<Outer::Second::InSecond<int> > OS_OS;
      };
   }
}

void storeACl() {
   TFile* file = TFile::Open("fornamespace.root", "RECREATE");
   Outer::Inner::Cl cl;
   file->WriteObject(&cl, "cl");
   file->ls();
   file->Write();
   file->GetStreamerInfoList()->FindObject("Outer::Inner::Cl")->ls();
   delete file;

   TClass::GetClass("Outer::Inner::Cl")->Dump(&cl, kTRUE /*noAddr*/);
}
