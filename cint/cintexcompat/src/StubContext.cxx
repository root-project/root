// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "StubContext.h"

#include "CINTdefs.h"
#include "CINTFunctional.h"

#include "Reflex/Member.h"
#include "Reflex/Type.h"
#include "Reflex/Tools.h"

#include "G__ci.h"
#include "Api.h"

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT {
namespace Cintex {

//______________________________________________________________________________
class StubContexts : public vector<StubContext_t*>  {
public:
   static StubContexts& Instance() {
      static StubContexts s_cont;
      return s_cont;
   }
private:
   StubContexts() {}
   ~StubContexts()  {
      for (vector<StubContext_t*>::iterator j = begin(); j != end(); ++j) {
         delete(*j);
      }
      clear();
   }
};

//______________________________________________________________________________
StubContext_t::StubContext_t(Member mbr)
: fMbr(mbr)
, fMethodCode(0)
, fNewdelfuncs(0)
, fInitialized(false)
{
   if (mbr.IsConstructor() || mbr.IsDestructor()) {
      Member getnewdelfuncs = mbr.DeclaringScope().MemberByName("__getNewDelFunctions");
      if (getnewdelfuncs) {
         static Type tNewdelfuncs = Type::ByTypeInfo(typeid(fNewdelfuncs));
         Object ret(tNewdelfuncs, (void*) &fNewdelfuncs);
         getnewdelfuncs.Invoke(&ret);
      }
   }
   StubContexts::Instance().push_back(this);
}

//______________________________________________________________________________
StubContext_t::~StubContext_t()
{
   Free_code((void*)fMethodCode);
}

//______________________________________________________________________________
void StubContext_t::Initialize()
{
   int param_cnt = fMbr.TypeOf().FunctionParameterSize();
   fTreat.resize(param_cnt);
   // pre-process parameters and remember the treatment that is needed to be done
   for (int i = 0; i < param_cnt; ++i) {
      Type pt = fMbr.TypeOf().FunctionParameterAt(i).FinalType();
      if (pt.IsReference() && !pt.IsConst()) {
         if (pt.IsPointer()) {
            fTreat[i] = '*';
         }
         else {
            fTreat[i] = '&';
         }
      }
      else if (pt.IsFundamental() || pt.IsEnum()) {
         if (pt.TypeInfo() == typeid(float)) {
            fTreat[i] = 'f';
         }
         else if (pt.TypeInfo() == typeid(double)) {
            fTreat[i] = 'd';
         }
         else if (pt.TypeInfo() == typeid(long double)) {
            fTreat[i] = 'q';
         }
         else if (pt.TypeInfo() == typeid(long long)) {
            fTreat[i] = 'n';
         }
         else if (pt.TypeInfo() == typeid(unsigned long long)) {
            fTreat[i] = 'm';
         }
         else {
            fTreat[i] = 'i';
         }
      }
      else {
         fTreat[i] = 'u';
      }
   }
   fInitialized = true;
}

//______________________________________________________________________________
void StubContext_t::ProcessParam(G__param* libp)
{
   // Process param type.
   static std::vector<G__value> fParcnv;
   fParam.resize(libp->paran);
   fParcnv.resize(libp->paran);
   for (int i = 0; i < libp->paran; ++i) {
      switch (fTreat[i]) {
         case 'd':
            fParcnv[i].obj.d  = G__double(libp->para[i]);
            fParam[i] = &fParcnv[i].obj.d;
            break;
         case 'f':
            fParcnv[i].obj.fl = (float)G__double(libp->para[i]);
            fParam[i] = &fParcnv[i].obj.fl;
            break;
         case 'n':
            fParcnv[i].obj.ll = G__Longlong(libp->para[i]);
            fParam[i] = &fParcnv[i].obj.ll;
            break;
         case 'm':
            fParcnv[i].obj.ull = G__ULonglong(libp->para[i]);
            fParam[i] = &fParcnv[i].obj.ull;
            break;
         case 'q':
            fParcnv[i].obj.ld = G__Longdouble(libp->para[i]);
            fParam[i] = &fParcnv[i].obj.ld;
            break;
         case 'i':
            fParcnv[i].obj.i  = G__int(libp->para[i]);
            fParam[i] = &fParcnv[i].obj.i;
            break;
         case '*':
            fParam[i] = libp->para[i].ref ? (void*)libp->para[i].ref : &libp->para[i].obj.i;
            break;
         case '&':
            fParam[i] = (void*)libp->para[i].ref;
            break;
         case 'u':
            fParam[i] = (void*)libp->para[i].obj.i;
            break;
      }
   }
}

//______________________________________________________________________________
void StubContext_t::ProcessResult(G__value* result, void* objaddr)
{
   // Map a return value from a reflex function stub to
   // the return value which would have come from a cint
   // function stub.
   Type ret_type = fMbr.TypeOf().ReturnType();
   char type = ret_type.RepresType();
   void* obj = objaddr;
   result->ref = 0;
   if (ret_type.FinalType().IsReference()) {
      obj = *(void**)objaddr;
      result->ref = (long) obj;
   }
   switch (type) {
      case 'B': // pointer to unsigned char
      case 'C': // pointer to char
      case 'D': // pointer to double
      case 'F': // pointer to float
      case 'G': // pointer to bool
      case 'H': // pointer to unsigned int
      case 'I': // pointer to int
      case 'K': // pointer to unsigned long
      case 'L': // pointer to long
      case 'M': // pointer to unsigned long long
      case 'N': // pointer to long long
      case 'R': // pointer to unsigned short
      case 'S': // pointer to short
      case 'U': // pointer to class
      case 'Y': // pointer to void
         G__letint(result, type, (long) (long*)*(void**)obj);
         break;
      case 'Q': // pointer to function
         G__letint(result, type, (long) (int*)obj);
         break;
      case 'b': // unsigned char
         G__letint(result, type, (long) *(unsigned char*)obj);
         break;
      case 'c': // char
         G__letint(result, type, (long) *(char*)obj);
         break;
      case 'd': // double
         G__letdouble(result, type, (double) *(double*)obj);
         break;
      case 'f': // float
         G__letdouble(result, type, (double) *(float*)obj);
         break;
      case 'g': // bool
         G__letint(result, type, (long) *(bool*)obj);
         break;
      case 'h': // unsigned int
         G__letint(result, type, (long) *(unsigned int*)obj);
         break;
      case 'i': // int
         G__letint(result, type, (long) *(int*)obj);
         break;
      case 'k': // unsigned long
         G__letint(result, type, (long) *(unsigned long*)obj);
         break;
      case 'l': // long
         G__letint(result, type, (long) *(long*)obj);
         break;
      case 'm': // unsigned long long
         G__letULonglong(result, type, *(unsigned long long*)obj);
         break;
      case 'n': // long long
         G__letLonglong(result, type, *(long long*)obj);
         break;
      case 'q': // long double
         G__letLongdouble(result, type, *(long double*)obj);
         break;
      case 'r': // unsigned short
         G__letint(result, type, (long) *(unsigned short*)obj);
         break;
      case 's': // short
         G__letint(result, type, (long) *(short*)obj);
         break;
      case 'u': // class
         G__letpointer(result, (long) obj, ret_type);
         result->ref = (long) obj;
         break;
      case 'y': // void
         G__setnull(result);
         break;
   }
}

//______________________________________________________________________________
void* StubContext_t::GetReturnAddress(G__value* result) const
{
   // Extract the memory location of the return value given the return type of fMethod
   Type ret_final_type = fMbr.TypeOf().ReturnType().FinalType();
   if (ret_final_type.IsPointer()) {
      return &result->obj.i;
   }
   if (ret_final_type.IsReference()) {
      return &result->ref;
   }
   char type = ret_final_type.RepresType();
   switch (type) {
      case 'f':
      case 'd':
         return &result->obj.d;
      case 'q':
         return &result->obj.ld;
      case 'y':
         return 0;
      case 'm':
      case 'n':
         return &result->obj.ll;
   }
   return &result->obj.i;
}

} // namespace Cintex
} // namespace ROOT
