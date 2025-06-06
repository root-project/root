#include "TObject.h"
#include <iostream>
#include "TClass.h"
#include "TSystem.h"
#include "TInterpreter.h"

#ifndef DEFINED_DERIVE
#define DEFINED_DERIVE
class Derived : public TObject {
   int fValue;
   ClassDefOverride(Derived, 1);
};
#endif

int i = Derived::Class()->GetClassVersion();

// auto l = ((Derived::Class()->BuildRealData()) , (Derived::Class()->GetListOfRealData()));
// This is illegal, it will given incorrect result and lead to errors like:
// This is because it is 'too soon' neither the header information nor the rootpcm information has been made available
// yet.

auto p = Derived::Class() -> Property();
// Also illegal (too soon) and lead to "just":
//    Error in <TClass::LoadClassInfo>: no interpreter information for class Derived is available even though it has a TClass initialization routine.

int execInitOrder()
{
   auto cl = Derived::Class();
   if (!cl) {
      std::cerr << "Could not find the TClass for Derived\n";
      return 2;
   }
   if (!cl->GetClassInfo()) {
      std::cerr << "No ClassInfo found by TClass for Derived\n";
      return 3;
   }

   auto inh = cl->InheritsFrom(TObject::Class());
   std::cout << "Derived is found to " << (inh ? "" : "not ") << "inherit from TObject" << std::endl;
   int res = (inh == 1) ? 0 : 1;

   if (cl->Property() == -1) {
      std::cerr << "Problem initializing Derived' Property\n";
      ++res;
   }
   if (cl->GetState() != TClass::kHasTClassInit) {
      std::cerr << "Unexpected state or Derived: " << cl->GetState() << '\n';
      ++res;
   }
   cl->BuildRealData();
   if (cl->GetListOfRealData()->GetEntries() != 3) {
      std::cerr << "Unexpected number of real data for Derived: " << cl->GetListOfRealData()->GetEntries()
                << " [3 was the expected number]\n";
      cl->GetListOfRealData()->ls();
      ++res;
   } else if (!cl->GetListOfRealData()->At(0) || strcmp(cl->GetListOfRealData()->At(0)->GetName(), "fValue") != 0) {
      std::cerr << "Unexpected name for real data for Derived: " << cl->GetListOfRealData()->At(0)->GetName()
                << " [fValue was the expected name]\n";
      cl->GetListOfRealData()->ls();
      ++res;
   }

   gInterpreter->Declare("#include \"InitOrder.h\"");

   cl = TClass::GetClass("UserClass");
   if (!cl) {
      std::cerr << "Could not find the TClass for UserClass\n";
      return 2;
   }
   if (cl->Property() == -1) {
      std::cerr << "Problem initializing UserClass' Property\n";
      ++res;
   }
   if (cl->GetState() != TClass::kInterpreted) {
      std::cerr << "Unexpected state for UserClass: " << cl->GetState() << '\n';
      ++res;
   }

   auto lres = gSystem->Load("libInitOrder");
   if (lres != 0) {
      std::cerr << "Error could not load libInitOrder";
      return 3;
   }

   cl = TClass::GetClass("UserClass");
   if (!cl) {
      std::cerr << "Could not find the TClass for UserClass after loading the library\n";
      return 4;
   }
   if (cl->Property() == -1) {
      std::cerr << "Problem initializing UserClass' Property after loading the library\n";
      ++res;
   }
   if (cl->GetState() != TClass::kHasTClassInit) {
      std::cerr << "Unexpected state after loading the library for UserClass: " << cl->GetState() << '\n';
      ++res;
   }

   gSystem->Unload("libInitOrder");

   auto cl2 = TClass::GetClass("UserClass");
   if (cl != cl2) {
      std::cerr << "Address of TClass for UserClass unexpectedly changed after unload\n";
      ++res;
   }
   if (cl2->Property() == -1) {
      std::cerr << "Problem initializing UserClass' Property after loading the library\n";
      ++res;
   }
   if (cl2->GetState() != TClass::kInterpreted) {
      std::cerr << "Unexpected state after loading the library for UserClass: " << cl2->GetState() << '\n';
      ++res;
   }

   lres = gSystem->Load("libInitOrder");
   if (lres != 0) {
      std::cerr << "Error could not load libInitOrder";
      return 5;
   }

   cl = TClass::GetClass("UserClass");
   if (!cl) {
      std::cerr << "Could not find the TClass for UserClass after loading the library\n";
      return 6;
   }
   if (cl->Property() == -1) {
      std::cerr << "Problem initializing UserClass' Property after loading the library\n";
      ++res;
   }
   if (cl->GetState() != TClass::kHasTClassInit) {
      std::cerr << "Unexpected state after loading the library for UserClass: " << cl->GetState() << '\n';
      ++res;
   }

   cl = TClass::GetClass("UserClassViaTypedef");
   if (!cl) {
      std::cerr << "Could not find the TClass for UserClassViaTypedef after loading the library\n";
      return 7;
   }
   if (cl->GetState() != TClass::kHasTClassInit) {
      std::cerr << "Unexpected state after loading the library for UserClassViaTypedef: " << cl->GetState() << '\n';
      ++res;
   }

   cl = TClass::GetClass("UserClassViaUsing");
   if (!cl) {
      std::cerr << "Could not find the TClass for UserClassViaUsing after loading the library\n";
      return 8;
   }
   if (cl->GetState() != TClass::kHasTClassInit) {
      std::cerr << "Unexpected state after loading the library for UserClassViaUsing: " << cl->GetState() << '\n';
      ++res;
   }

   cl = TClass::GetClass("UserClassNotSelected");
   if (!cl) {
      std::cerr << "Could not find the TClass for UserClassNotSelected after loading the library\n";
      return 9;
   }
   if (cl->GetState() != TClass::kInterpreted) {
      std::cerr << "Unexpected state after loading the library for UserClassNotSelected: " << cl->GetState() << '\n';
      ++res;
   }

   // Unloading and the Reloading the libInitOrderDups library will lead to the error:
   /*
   libInitOrder dictionary forward declarations' payload:11:67: error: typedef redefinition with different types ('LorentzVector<PxPyPzE4D<...>>' vs 'LorentzVector<PxPyPzE4D<...>>') typedef
   ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > Vector4D __attribute__((annotate("$clingAutoload$/local2/pcanal/cint_working/rootcling/root/roottest/root/meta/tclass/InitOrder.h")));
                                                                     ^
   ./InitOrder.h:7:68: note: previous definition is here
   typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >  Vector4D;
                                                                      ^
   */
   cl = TClass::GetClass("ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >");
   if (!cl) {
      std::cerr << "Could not find the TClass for ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > "
                   "before loading the 'dups' library\n";
      return 10;
   }
   if (cl->GetState() != TClass::kHasTClassInit) {
      std::cerr << "Unexpected state after loading the library for "
                   "ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >"
                << cl->GetState() << '\n';
      ++res;
   }
   auto lres2 = gSystem->Load("libInitOrderDups");
   if (lres2 != 0) {
      std::cerr << "Error could not load libInitOrderDups";
      return 11;
   }

   return res;
}
