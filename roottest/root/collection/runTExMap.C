#ifndef __CINT__

#include "TExMap.h"
#include "Riostream.h"

#endif


#include "Rtypes.h"


Bool_t TestUpdate()
{
   return kTRUE;
}


typedef ULong_t (*hashfun_t)(Long_t);

ULong_t hash_min(Long_t /* k */ ) { return 0; }
ULong_t hash_max(Long_t /* k */ ) { return ~0; }
ULong_t hash_identity(Long_t k) { return k; }
ULong_t hash_div4(Long_t k) { return k / 4; }


Bool_t TestRange(hashfun_t hfunc)
{
   TExMap *m = new TExMap;

   {for( int i=0; i < 1000; i++ ) {
      m->Add(hfunc(i), i, i);

      if (m->GetSize() != i+1) {
         cout << "TestRange: Insert: GetSize() != " << i+1 << endl;
         delete m;
         return kFALSE;
      }
   }}

   {for( int i=0; i < 1000; i++ ) {
      Long_t v = m->GetValue(hfunc(i), i);
      if (v != i) {
         cout << "TestRange: GetValue: key " << i << " val " << v << endl;
         delete m;
         return kFALSE;
      }
   }}

   {for( int i=0; i < 1000; i++ ) {
      m->Remove(hfunc(i), i);

      if (m->GetSize() != (1000-1-i)) {
         cout << "TestRange: Remove(" << i << "): GetSize() != " << 1000-1-i << endl;
         delete m;
         return kFALSE;
      }
   }}

   return kTRUE;
}


Bool_t TestDoubleAdd(hashfun_t hfunc, Long_t k)
{
   TExMap *m = new TExMap;

   m->Add(hfunc(k), k, k);

   cout << "(Expect Error) " << flush;

   m->Add(hfunc(k), k, k);

   if (m->GetSize() != 1) {
      cout << "TestDoubleAdd: GetSize() != 1" << endl;
      delete m;
      return kFALSE;
   }

   Long_t v = m->GetValue(hfunc(k), k);
   if (v != k) {
      cout << "TestDoubleAdd: GetValue: key " << k << " val " << v << endl;
      delete m;
      return kFALSE;
   }

   m->Remove(hfunc(k), k);

   if (m->GetSize() != 0) {
      cout << "TestDoubleAdd: GetSize() != 0" << endl;
      delete m;
      return kFALSE;
   }

   delete m;
   return kTRUE;
}


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


const ULong_t  ulv[] = { 0, 1, 2, kMaxULong };
const Int_t    kNUlv = sizeof(ulv) / sizeof(ULong_t);

const Long_t  lv[] = { kMinLong, -1, 0, 1, kMaxLong };
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

   cout << endl << "TestRange(hash_identity)" << endl << endl;
   if ( !TestRange(hash_identity) ) return;

   cout << endl << "TestRange(hash_min)" << endl << endl;
   if ( !TestRange(hash_min) ) return;

   cout << endl << "TestRange(hash_max)" << endl << endl;
   if ( !TestRange(hash_max) ) return;

   cout << endl << "TestRange(hash_div4)" << endl << endl;
   if ( !TestRange(hash_div4) ) return;


   int i;
   for(i=0; i < kNLv; i++) {

      cout << endl << "TestDoubleAdd(hash_identity, " << lv[i]<< ")" << endl << endl;
      if ( !TestDoubleAdd(hash_identity, lv[i]) ) return;

      cout << endl << "TestDoubleAdd(hash_min, " << lv[i]<< ")" << endl << endl;
      if ( !TestDoubleAdd(hash_min, lv[i]) ) return;

      cout << endl << "TestDoubleAdd(hash_max, " << lv[i]<< ")" << endl << endl;
      if ( !TestDoubleAdd(hash_max, lv[i]) ) return;

      cout << endl << "TestDoubleAdd(hash_div4, " << lv[i]<< ")" << endl << endl;
      if ( !TestDoubleAdd(hash_div4, lv[i]) ) return;
   }

   cout << endl << "TExMap Basic Tests End" << endl << endl;
}
