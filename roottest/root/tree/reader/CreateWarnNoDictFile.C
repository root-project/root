#ifndef __CreateWarnNoDictFile__
#define __CreateWarnNoDictFile__

#include "TClass.h"
#include "TFile.h"
#include "TTree.h"

#include <map>
#include <memory>
#include <string>

#ifdef __ROOTCLING__
#pragma link C++ class std::map<std::string, bool>+;
#endif

int CreateWarnNoDictFile() 
{

   using rareType = std::map<std::string, bool>;
   auto c = TClass::GetClass<rareType>();

   const auto fName = "warnNoDict.root";
   const auto tName = "t";
   const auto colName = "col";

   auto f = std::make_unique<TFile>(fName, "RECREATE");
   auto t = std::make_unique<TTree>(tName, "test tree");
   rareType o;
   o["foo"] = true;
   t->Branch(colName, &o);
   t->Fill();
   t->Write();
   return 0;
}

#endif