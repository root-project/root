#pragma once

#include <Rtypes.h>
#include <vector>
#include <iostream>
#include <iomanip>

struct SubContent {

   int fValue = 0;
   ClassDef(SubContent, 2);
};

struct Content {
   int fBefore = 0;     //!
   SubContent fSubObj;
   float fAfter = 0;
   ClassDef(Content, 2);
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
   std::cout << "Content.fAfter: " << c.fAfter << "\n";
}

struct Holder {

   std::vector<Content> obj;
   ClassDef(Holder, 2);
};

void Print(const Holder &h)
{
   std::cout << "Holder has " << h.obj.size() << " value" << ((h.obj.size() == 1) ? "" : "s") <<  ".\n";
   for(auto &c : h.obj) {
      Print(c);
   }  
}

#include "TFile.h"
#include "TTree.h"

void write(const char *filename = "evosubobj.root")
{
   TFile *file = TFile::Open(filename, "RECREATE");
   TTree *tree = new TTree("tree", "tree");

   Holder h;
   Content c;
   Set(c, 1);
   h.obj.push_back(c);

   Print(h);

   tree->Branch("holder.", &h);
   tree->Fill();

   file->Write();
};

int write_cmssw_class_v2() {
  write();
  return 0;
}

