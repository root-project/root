//
//  RNTupleBrowser.cpp
//  
//
//  Created by SFT Group on 19.07.19.
//

#include "ROOT/RNTupleBrowser.hxx"
#include <Rtypes.h>

ClassImp(ROOT::Experimental::RNTupleBrowser);

ROOT::Experimental::RNTupleBrowser::RNTupleBrowser(TDirectory* directory): fDirectory{directory}
{
   std::string fullPath = fDirectory->GetPath();
   std::string fileName = std::string(fullPath, 0, fullPath.find(".root")+5);
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> sourcePtr = std::make_unique<ROOT::Experimental::Detail::RPageSourceRoot>(fDirectory->GetName(), fileName, fDirectory);
   fReaderPtr = std::make_unique<ROOT::Experimental::RNTupleReader>(std::move(sourcePtr));
   std::cout << "The Constructor was called\n";
}

ClassImp(ROOT::Experimental::RNTupleBrowseField);

   void ROOT::Experimental::RNTupleBrowser::Browse(TBrowser *b) {
   
      for(const auto &f : *(fReaderPtr->GetModel()->GetRootField())) {
         RNTupleBrowseField* field = new RNTupleBrowseField(f.GetName());
         field->Browse(b);
         fNTupleBrowsePtrVec.emplace_back(field);
      }
   //fReaderPtr->PrintInfo();
   
   //TH1F *h1 = new TH1F("h1", "x distribution", 100, -4, 4);
   //h1->Fill(1);
   //b->Add(h1);
   //if (gPad) {
   // gPad->Update();
   //}
   
}
