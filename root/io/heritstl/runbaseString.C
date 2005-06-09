#include <string>
#include "TFile.h"
#include <iostream>

struct MyString : public std::string {
   MyString() {}
   MyString(const char *input) : string(input) {}
  virtual ~MyString() {}
};

void writeBaseString() 
{
   // gROOT->LoadMacro("t.C+")
   MyString str("hello");
   TFile fo("strtest.root","RECREATE");
   fo.WriteObject(&str,"str");
   fo.Close();
}

void readBaseString() 
{
   //gROOT->LoadMacro("t.C+")
   TFile fi("strtest.root");
   MyString* s;
   fi.GetObject("str",s);
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
