#include <vector>
#include <iostream>
#include "Rtypes.h"

template <typename T>
struct Wrapper
{
   bool present = true;
   T obj;
};

struct MatchedCSCSegment
{
   float someValue = 0.0;

   MatchedCSCSegment(float in = 0.0) : someValue{in} {}

   // The simple update failed if the class version is not set.
   // ClassDef(MatchedCSCSegment, 5);
};

struct CSCSegment
{
   float someValue = 0.0;

   operator MatchedCSCSegment()
   {
      return MatchedCSCSegment{someValue};
   }

   std::vector<CSCSegment> theDuplicateSegments;
   // ClassDef(CSCSegment, 3);
};

#ifdef __ROOTCLING__
#pragma link C++ class MatchedCSCSegment+;
#pragma link C++ class CSCSegment+;
#pragma link C++ class Wrapper<std::vector<CSCSegment>>+;
#endif

#include "TFile.h"
#include "TTree.h"

int ROOT14491writedata(const char *filename = "oldfile14491.root")
{
   auto file = TFile::Open(filename, "RECREATE");
   auto t = new TTree("t", "t");
   Wrapper<std::vector<CSCSegment>> w;

   CSCSegment c;
   c.theDuplicateSegments.push_back(CSCSegment{});

   w.obj.push_back(c);

   t->Branch("seg_split.", &w, 32000, 99);
   t->Branch("seg_unsplit.", &w, 32000, 0);

   t->Fill();
   file->Write();
   // t->Print();
   delete file;
   return 0;
}
