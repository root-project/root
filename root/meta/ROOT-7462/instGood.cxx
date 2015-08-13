#ifndef instGood_cxx
#define instGood_cxx

#include "instHeader.h"

template <> class Inner<int> { public: int fValue; };

//sdecltest("std::pair<string,Outer<int>>")
void func2() {
   std::pair<string,Inner<int>> p;
}



#include <map>

class Outer {
public:
   std::pair<string,Inner<int>> fContent;

   Outer() { fContent.first = "not set"; fContent.second.fValue = -1; }

   void Print() {
      printf("string=%s\n",fContent.first.c_str());
      printf("value=%d\n",fContent.second.fValue);
   }
};



#if 0
#ifdef __ROOTCLING__
#pragma link C++ class Outer+;
#pragma link C++ class pair<string,Inner<int> >+;
#endif
#endif



#include "TFile.h"
#include <memory>

void writeFile(const char *filename = "inst.root")
{
   std::unique_ptr<TFile> file(TFile::Open(filename,"RECREATE"));
   Outer out;
   out.fContent.first = "set for writing";
   out.fContent.second.fValue = 11;
   file->WriteObject(&out,"object");

}

#endif
