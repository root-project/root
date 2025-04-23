#include <string>
#include "TFile.h"
#include <iostream>
#include <vector>

struct MyString : public std::string {
   MyString() {}
   MyString(const char *input) : string(input) {}
  virtual ~MyString() {}
};

struct TMyVector : public std::vector<const char*>, TObject  {
   TMyVector() {}
   TMyVector(const char *input) { push_back(input); }
   virtual ~TMyVector() {}
   ClassDef(TMyVector,1);
};


void writeBaseString() 
{
   // gROOT->LoadMacro("t.C+")
   MyString str("hello");
   TMyVector vec("hello");
   
   TFile fo("strtest.root","RECREATE");
   fo.WriteObject(&str,"str");
   fo.WriteObject(&vec,"vec");
   fo.Close();
}

void readBaseString() 
{
   //gROOT->LoadMacro("t.C+")
   TFile fi("strtest.root");
   MyString* s;
   fi.GetObject("str",s);
   TMyVector *vec;
   fi.GetObject("vec",vec);
}

void runbaseString(int what = 1) 
{
   switch(what) {
   case 0: writeBaseString();
      break;
   case 1: readBaseString();
      break;
   default:
      std::cerr << "Unexpected command value " << what << std::endl;
   }
}
