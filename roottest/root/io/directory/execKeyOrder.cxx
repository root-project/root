#include "TKey.h"
#include "TFile.h"
#include <iostream>
#include "TList.h"
#include "THashTable.h"
#include "THashList.h"

TKey* GetKey(TFile* f,const char *name, Short_t cycle =9999)
{
   TKey *key;
   TIter next(f->GetListOfKeys());
   while ((key = (TKey *) next())) {
      if (!strcmp(name, key->GetName())) {
         if (cycle == 9999)             return key;
         if (cycle >= key->GetCycle())  return key;
      }
   }
   return nullptr;
}

void listKeys(TFile *f) {

   cout << "From the list of keys:\n";
   TKey *key;
   Int_t i = 0;
   TIter next(f->GetListOfKeys());
   while ((key = (TKey *) next())) {
      cout<<i<<" "<<key<<" "<<key->GetCycle()<<endl;
      ++i;
   }

   auto l2 =((THashList *)(f->GetListOfKeys()))->GetListForObject("bla");   // new root uses this and returns first match
   cout << "From the list of hash for the key:\n";
   i = 0;
   next = l2;
   while ((key = (TKey *) next())) {
      cout<<i<<" "<<key<<" "<<key->GetCycle()<<endl;
      ++i;
   }
}

void checkCompleteKeys(TFile *f) {

   TList* l = f->GetListOfKeys();                // newest objects at front
   cout<<"lists of keys"<<endl;
   for(Int_t i=0;i<l->GetEntries();i++){
      TKey* k = (TKey*) l->At(i);
      cout<<i<<" "<<k<<" "<<k->GetCycle()<<endl;

   }

   cout<<"GetKey"<<endl;
   TKey* k = (TKey*) f->GetKey("bla");    // different results
   cout<<k<<" "<<k->GetCycle()<<endl;

   cout<<"FindKey"<<endl;
   k = (TKey*) f->FindKey("bla");    // different results
   cout<<k<<" "<<k->GetCycle()<<endl;

   cout<<"FindKeyAny"<<endl;
   k = (TKey*) f->FindKeyAny("bla");      // gives in both cases the same, but could be problem (includes subdirs)
   cout<<k<<" "<<k->GetCycle()<<endl;

   auto l2 =((THashList *)(f->GetListOfKeys()))->GetListForObject("bla");   // new root uses this and returns first match
   cout<<"lists of keys hash"<<endl;                                              // newst objects at end
   for(Int_t i=0;i<l2->GetEntries();i++){
      TKey* k = (TKey*) l2->At(i);
      cout<<i<<" "<<k<<" "<<k->GetCycle()<<endl;

   }

   cout<<"manual GetKey"<<endl;     // works both cases
   k =  GetKey(f,"blub");
   if (!k) cout << " not found\n";
   else cout<<k<<" "<<k->GetCycle()<<endl;
   k =  GetKey(f,"bla");
   if (!k) cout << " not found\n";
   else cout<<k<<" "<<k->GetCycle()<<endl;

}

void checkKeys(TFile *f) {

   TList* l = f->GetListOfKeys();                // newest objects at front
   cout<<"lists of keys"<<endl;
   for(Int_t i=0;i<l->GetEntries();i++){
      TKey* k = (TKey*) l->At(i);
      cout<<i<<" "<<" "<<k->GetCycle()<<endl;

   }

   cout<<"GetKey"<<endl;
   TKey* k = (TKey*) f->GetKey("bla");    // different results
   cout<<" "<<k->GetCycle()<<endl;

   cout<<"FindKey"<<endl;
   k = (TKey*) f->FindKey("bla");    // different results
   cout<<" "<<k->GetCycle()<<endl;

   cout<<"FindKeyAny"<<endl;
   k = (TKey*) f->FindKeyAny("bla");      // gives in both cases the same, but could be problem (includes subdirs)
   cout<<" "<<k->GetCycle()<<endl;

   auto l2 =((THashList *)(f->GetListOfKeys()))->GetListForObject("bla");   // new root uses this and returns first match
   cout<<"lists of keys hash"<<endl;                                              // newst objects at end
   for(Int_t i=0;i<l2->GetEntries();i++){
      TKey* k = (TKey*) l2->At(i);
      cout<<i<<" "<<" "<<k->GetCycle()<<endl;

   }

   cout<<"manual GetKey"<<endl;     // works both cases
   k =  GetKey(f,"blub");
   if (!k) cout << " not found\n";
   else cout<<" "<<k->GetCycle()<<endl;
   k =  GetKey(f,"bla");
   if (!k) cout << " not found\n";
   else cout<<" "<<k->GetCycle()<<endl;

}


void execKeyOrder()
{
   // run macro with old 5.34.01  or 5.34.14 (or later ) to compare
   TFile* f = new TFile("test_execKeyOrder.root","RECREATE");
   TNamed n("bla","blub");

   n.Write();
   n.Write();
   n.Write();
   n.Write();

   checkKeys(f);

   delete f;
   f = new TFile("test_execKeyOrder.root","READ");
   checkKeys(f);

}

