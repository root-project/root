#include "MyDerived.cxx"

#ifdef __ROOTCLING__
#pragma link C++ class MyDerived+;
#endif

#include "TFile.h"
#include "TKey.h"
#include <iostream>

void execSkipbaseclass (const char * fileName = "skipbaseclass.root" ) {

   TFile * file = TFile::Open(fileName);
   if (file) {
      TList * list = file->GetListOfKeys();
      if (list) {
         Int_t nent = list->GetEntries();
         for (Int_t i=0; i<nent; ++i) {
            TKey * key = (TKey*)list->At(i);
            if (key) {
               std::cout << "Reading " << key->GetName() << std::endl;
               //gDebug=1;
               TObject * obj = file->Get(key->GetName());
               //obj->Dump();
               //gDebug=0;
            }
         }
      }
      file->Close();
   }
}