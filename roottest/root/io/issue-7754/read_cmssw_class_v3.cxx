#pragma once

#include "Rtypes.h"
#include <vector>
#include <iostream>
#include <iomanip>

struct SubContent {

   int fValue = 0;
   float fNewValue = 0;
   ClassDef(SubContent, 3);
};

struct OtherSubContent {
   int fAnother = 0;
   ClassDef(OtherSubContent, 3);
};

struct Content {
   int fBefore = 0;
   SubContent fSubObj;
   float fAfter = 0;
   char padding[240];
   double fNewData = 0.0;
   
   ClassDef(Content, 3);
};

void Set(Content &c, int seed = 1)
{
   c.fBefore = seed;
   c.fSubObj.fValue = seed + 1;
   c.fAfter = seed + 10;
}

void Print(const Content &c)
{
   std::cout << "Content.fBefore (transient): " << c.fBefore << "\n";
   std::cout << "Content.fSubObj.fValue: " << c.fSubObj.fValue << "\n";
   std::cout << "Content.fSubObj.fNewValue: " << c.fSubObj.fNewValue << "\n";
   std::cout << "Content.fAfter: " << c.fAfter << "\n";
   std::cout << "Content.fNewData " << c.fNewData << "\n";
}

struct Holder {

   std::vector<Content> obj;
   ClassDef(Holder, 3);
};

void Print(const Holder &h)
{
   std::cout << "Holder has " << h.obj.size() << " value" << ((h.obj.size() == 1) ? "" : "s") <<  ".\n";
   for(auto &c : h.obj) {
      Print(c);
   }
   
}

#ifdef __ROOTCLING__
#pragma read sourceClass="SubContent" targetClass="SubContent" versions="[2]" source="int fValue" target="fValue" code="{ fValue = 100 * onfile.fValue; }"
#pragma read sourceClass="SubContent" targetClass="SubContent" versions="[2]" source="int fValue" target="fNewValue" code="{ fNewValue = 3000 * onfile.fValue; }"
#pragma read sourceClass="Content" targetClass="Content" versions="[2]" source="float fAfter" target="fAfter" code="{ fAfter = 10 * onfile.fAfter; }"
#pragma read sourceClass="Content" targetClass="Content" versions="[2]" source="float fAfter" target="fNewData" code="{ fNewData = 20 * onfile.fAfter; }"

#endif

#include "TFile.h"
#include "TTree.h"

int read(const char *filename = "evosubobj.root")
{
   TFile *file = TFile::Open(filename, "READ");
   if (!file)
      return 1;
   TTree *tree = file->Get<TTree>("tree");
   if (!tree)
      return 2;

   Holder *h = nullptr;
   tree->SetBranchAddress("holder.", &h);
   tree->GetEntry(0);
   //tree->Print("debugInfo");
   //tree->Print("debugAddress");

   if (!h)
      return 3;

   Print(*h);

   return 0;
};

int read_cmssw_class_v3() {
   return read();
}
