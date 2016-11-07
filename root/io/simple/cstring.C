#include "TString.h"
#include <string>
#include "TError.h"


class TXmlEx4 {
   public: 
   TXmlEx4() : fStr2(0), fStr3(0) {}

   const char* fStr2;
   const char* fStr3;
   TString *fMember = nullptr;
   std::string *fStdMember = nullptr;
};

#include "TFile.h"

void cstring() {
   TXmlEx4 a;
   TFile f("test.root","RECREATE");
   f.WriteObject(&a,"xmltest");
   TXmlEx4 *ptr;
   f.GetObject("xmltest",ptr);
   if (!ptr) Fatal("cstring test","Unable to read back object.");

   if (ptr->fStdMember == nullptr) Warning("cstring test","Change in behavior for null valued std::string pointer");
   else if (ptr->fStdMember->size() != 0) Error("cstring test","null valued std::string pointer not read-back correctly (size() == %ld)",ptr->fStdMember->size());

   if (ptr->fMember == nullptr) { /* all good */ } 
   else if (ptr->fMember->Length() != 0) Error("cstring test","null valued TString pointer not read-back correctly (Length() == %d)",ptr->fMember->Length());


   if (ptr->fStr2) Error("cstring test","fStr2 is not nullptr");
   if (ptr->fStr3) Error("cstring test","fStr2 is not nullptr");
}
