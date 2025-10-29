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
   float someValue = 0.0;

   operator MatchedCSCSegment()
   {
      return MatchedCSCSegment{someValue};
   }

   std::vector<MatchedCSCSegment> theDuplicateSegments;

   // ClassDef(CSCSegment, 4);
};

static const char *dups_label = "dups ";
static const char *copy_label = "copy ";

#ifdef __ROOTCLING__
#pragma link C++ class MatchedCSCSegment+;
#pragma link C++ class CSCSegment+;
#pragma link C++ class Wrapper<std::vector<CSCSegment>>+;
#pragma read sourceClass="CSCSegment" targetClass="CSCSegment" version="[1-11]" \
   checksum="[0x94f3cbee, 2499005422]" \
   source="std::vector<CSCSegment> theDuplicateSegments" target="theDuplicateSegments" \
   code = "{ std::cout << dups_label << onfile.theDuplicateSegments.size() << std::endl; \
      std::copy(onfile.theDuplicateSegments.begin(), onfile.theDuplicateSegments.end(), \
                std::back_inserter(theDuplicateSegments)); \
      std::cout << copy_label << theDuplicateSegments.size() << std::endl; }"
#endif

#include "TFile.h"
#include "TTree.h"

void test(TTree *t, const char *bname)
{
   std::string formula = bname;
   formula += ".obj.theDuplicateSegments@.size()";
   t->Scan(formula.c_str());
}

int ROOT14491readdata()
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
