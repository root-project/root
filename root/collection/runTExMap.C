#ifndef __CINT__

#include "TExMap.h"
#include "Riostream.h"

#endif


Bool_t TestOne(ULong_t h, Long_t k, Long_t v)
{
   cout << "TestOne: " << h << ", " << k << ", "<< v << endl << endl;

   TExMap *m = new TExMap;

   // pre
   if (m->GetSize() != 0) {
      cout << "Pre: Not Empty" << endl;
      return kFALSE;
   }

   // insert
   m->Add(h, k, v);

   if (m->GetSize() != 1) {
      cout << "Insert: GetSize() != 1" << endl;
      return kFALSE;
   }

   // find
   Long_t nv = m->GetValue(h, k);

   if (v != nv) {
      cout << "Find: returned value different from value" << endl;
      return kFALSE;
   }

   // remove
   m->Remove(h, k);

   // post
   if (m->GetSize() != 0) {
      cout << "Post: Not Empty" << endl;
      return kFALSE;
   }

   delete m;

   return kTRUE;
}


const ULong_t  ulv[] = { 0, 1, 2, 0xFFFFFFFF /*kMaxUlong*/ };
const Int_t    kNUlv = sizeof(ulv) / sizeof(ULong_t);

const Long_t  lv[] = { 0x80000000/*kMinLong*/, -1, 0, 1, 0x7FFFFFFF /*kMaxlong*/ };
const Int_t   kNLv = sizeof(lv) / sizeof(Long_t);


void runTExMap()
{
   cout << endl << "TExMap Basic Tests Start" << endl << endl;

   TExMap *m = 0;

   cout << "TExMap()" << endl << endl;

   m = new TExMap;
   m->Print();
   cout << "Capacity(): " << m->Capacity() << endl;
   cout << "GetSize(): " << m->GetSize() << endl;
   delete m;

   cout << endl << "TExMap(1000)" << endl << endl;

   m = new TExMap(1000);
   m->Print();
   cout << "Capacity(): " << m->Capacity() << endl;
   cout << "GetSize(): " << m->GetSize() << endl;
   delete m;

   cout << endl << "TExMap(0)" << endl << endl;

   m = new TExMap(0);
   m->Print();
   cout << "Capacity(): " << m->Capacity() << endl;
   cout << "GetSize(): " << m->GetSize() << endl;
   delete m;

   cout << endl << "TExMap(-1)" << endl << endl;

   m = new TExMap(-1);
   m->Print();
   cout << "Capacity(): " << m->Capacity() << endl;
   cout << "GetSize(): " << m->GetSize() << endl;
   delete m;

   for( int i=0; i < kNUlv; i++) {
      for( int j=0; j < kNLv; j++) {
         for( int k=0; k < kNLv; k++) {
            if ( !TestOne(ulv[i], lv[j], lv[k]) ) return;
         }
      }
   }

   cout << endl << "TExMap Basic Tests End" << endl << endl;
}
