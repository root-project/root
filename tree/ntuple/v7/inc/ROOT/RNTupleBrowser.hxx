//
//  RNTupleBrowser.hpp
//  
//
//  Created by SFT Group on 19.07.19.
//

#ifndef RNTupleBrowser_hpp
#define RNTupleBrowser_hpp

#include <ROOT/RNTuple.hxx>
#include <ROOT/RPageStorageRoot.hxx>

#include <TBrowser.h>
#include <TDirectory.h>
#include <TH1F.h>
#include <TKey.h>
#include <TNamed.h>
#include <TROOT.h>

#include <vector>

namespace ROOT {
namespace Experimental {
   
   class RNTupleBrowseField;

class RNTupleBrowser: public TNamed {
public:
   std::vector<TNamed*> fNTupleBrowsePtrVec;
   
   TDirectory* fDirectory;
   std::unique_ptr<RNTupleModel> fReadModel;
   std::unique_ptr<RNTupleReader> fReaderPtr;
   
   
   RNTupleBrowser(TDirectory* directory=nullptr);
   
   void SetDirectory(TDirectory* directory) {
      fDirectory = directory;
      std::string fullPath = fDirectory->GetPath();
      std::string rootFileName = std::string(fullPath, 0, fullPath.find(".root")+5);
      std::unique_ptr<ROOT::Experimental::Detail::RPageSource> sourcePtr = std::make_unique<ROOT::Experimental::Detail::RPageSourceRoot>(fDirectory->GetName(), rootFileName, fDirectory);
      fReaderPtr = std::make_unique<ROOT::Experimental::RNTupleReader>(std::move(sourcePtr));
   }
   
   void Browse(TBrowser *b);
   Bool_t IsFolder() const { return kTRUE; }
   
   ~RNTupleBrowser() {
      std::cout << "The destructor was called\n";
      int x = 0;
      for(auto ptr : fNTupleBrowsePtrVec) {
         delete ptr;
         x++;
      }
      std::cout << x << " ptrs deleted\n";
   }
   
   ClassDef(RNTupleBrowser,2)
};
   
   
      
class RNTupleBrowseField: public TNamed {
public:
   int fX;
   RNTupleBrowseField(std::string name = "TestField") { fName = name; }
   
   
   void Browse(TBrowser *b) {b->Add(this);}
   
   
   //Bool_t IsFolder() const { return kTRUE; }
   ClassDef(RNTupleBrowseField, 2)
};

// RFieldBrowser evtl. einfaches Field -> Histogram
// bei komplizierteren mehrere Felder
   
} // namespace Experimental
} // namespace ROOT


#endif /* RNTupleBrowser_hpp */


