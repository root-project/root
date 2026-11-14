#include "TBranchProxy.h"
#include "TBranchProxyDirector.h"
#include "TTree.h"

R__LOAD_LIBRARY(libTreePlayer)

#include <iostream>
#include <memory>
#include <vector>

struct CopyCollectionProxyHit {
   double fX;
};

#ifdef __ROOTCLING__
// clang-format off
#pragma link C++ class CopyCollectionProxyHit+;
#pragma link C++ class std::vector<CopyCollectionProxyHit>+;
// clang-format on
#endif

int CopyCollectionProxy()
{
   TTree tree("t", "collection proxy copies");
   std::vector<CopyCollectionProxyHit> values{{1.}, {2.}, {3.}};
   tree.Branch("values", &values);
   tree.Fill();
   values = {{4.}, {5.}};
   tree.Fill();

   ROOT::Internal::TBranchProxyDirector director(&tree, 0);
   using Proxy = ROOT::Internal::TStlImpProxy<CopyCollectionProxyHit>;
   std::unique_ptr<Proxy> survivingCopy;
   {
      Proxy original(&director, "values");
      Proxy uninitializedCopy(original);
      if (!original.Read() || !original.GetCollection()) {
         std::cerr << "Could not initialize the collection proxy." << std::endl;
         return 1;
      }
      if (original.GetEntries() != 3 || original[0].fX != 1. || uninitializedCopy[2].fX != 3.) {
         std::cerr << "Could not read the original or its uninitialized copy." << std::endl;
         return 1;
      }

      survivingCopy = std::make_unique<Proxy>(original);
      if (survivingCopy->GetEntries() != 3 || (*survivingCopy)[1].fX != 2.) {
         std::cerr << "An initialized copy must have its own bound collection proxy." << std::endl;
         return 1;
      }

      // Binding another collection to the original must not change the copy.
      std::vector<CopyCollectionProxyHit> otherValues{{42.}};
      {
         TVirtualCollectionProxy::TPushPop context(original.GetCollection(), &otherValues);
         if (survivingCopy->GetEntries() != 3 || (*survivingCopy)[0].fX != 1.) {
            std::cerr << "Copies share mutable collection state." << std::endl;
            return 1;
         }
      }

      original.Reset();
      if (survivingCopy->GetEntries() != 3 || (*survivingCopy)[2].fX != 3.) {
         std::cerr << "Resetting the original affected its copy." << std::endl;
         return 1;
      }
   }

   if (survivingCopy->GetEntries() != 3 || (*survivingCopy)[0].fX != 1.) {
      std::cerr << "Destroying the original affected its copy." << std::endl;
      return 1;
   }
   director.SetReadEntry(1);
   if (survivingCopy->GetEntries() != 2 || (*survivingCopy)[1].fX != 5.) {
      std::cerr << "The surviving copy could not read the next entry." << std::endl;
      return 1;
   }
   return 0;
}
