#include <vector>
#include "TFile.h"
#include "TError.h"

class Data {
public:
   int fValue;
   Data() : fValue(0) {}
   Data(const Data &rhs) : fValue(rhs.fValue) {}
   Data(int in) : fValue(in) {}
};

#ifdef __ROOTCLING__
#pragma link C++ class vector<Data>+;
// #pragma read sourceClass="Container" targetClass="vector<Data>"
#endif

void execStlInheritNew(const char *filename = "inheritstl.root")
{
   TFile *f = TFile::Open(filename,"READ");

   std::vector<Data> *ptr = nullptr;
   f->GetObject("cont",ptr);
   if (!ptr) {
      Error("readfile","Could not read the container");
   } else {
      if (ptr->size())
         Info("readfile","The vector has the size %ld and content %d",ptr->size(),ptr->back().fValue);
      else
         Info("readfile","The vector has the size %ld and no content",ptr->size());
   }

   std::vector<int> *ptrint = nullptr;
   f->GetObject("contint",ptrint);
   if (!ptrint) {
      Error("readfile","Could not read the container of int");
   } else {
      if (ptrint->size())
         Info("readfile","The vector of int has the size %ld and content %d",ptrint->size(),ptrint->back());
      else
         Info("readfile","The vector of int has the size %ld and no content",ptrint->size());
   }

}