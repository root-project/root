// See https://github.com/root-project/root/issues/14491#issuecomment-1917587621

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

   // The simple update failed if the class version was not set.
   // ClassDef(MatchedCSCSegment, 5);
};

struct CSCSegment
{
   float someValue;

   operator MatchedCSCSegment()
   {
      return MatchedCSCSegment{someValue};
   }

   std::vector<MatchedCSCSegment> theDuplicateSegments;

   // ClassDef(CSCSegment, 4);
};

#ifdef __ROOTCLING__
#pragma link C++ class MatchedCSCSegment+;
#pragma link C++ class CSCSegment+;
#pragma link C++ class Wrapper<std::vector<CSCSegment>>+;
#pragma read sourceClass="CSCSegment" targetClass="MatchedCSCSegment"
#endif

#include "TFile.h"
#include "TTree.h"

void test(TTree *t, const char *bname)
{
   std::string formula = bname;
   formula += ".obj.theDuplicateSegments@.size()";
   t->Scan(formula.c_str());
}

int ROOT14491simpleread()
{
   auto file = TFile::Open("oldfile14491.root", "READ");
   auto t = file->Get<TTree>("t");
   t->LoadTree(0);
   // gDebug = 7;
   test(t, "seg_split");
   // gDebug = 0;
   test(t, "seg_unsplit");
   file->Close();
   delete file;
   return 0;
}
