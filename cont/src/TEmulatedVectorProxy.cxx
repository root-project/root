// @(#)root/cont:$Name:  $:$Id: TEmulatedVectorProxy.cxx,v 1.2 2004/01/27 19:50:31 brun Exp $
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEmulatedVectorProxy                                                 //
//                                                                      //
// Proxy around an emulated stl vector                                  //
//                                                                      //
// In particular this is used to implement splitting, emulation,        //
// and TTreeFormula access to STL vector for which we do not have       //
// access to the compiled copde             .                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEmulatedVectorProxy.h"
#include <vector>
#include "Api.h"
#include "TDataType.h"
#include "Property.h"
#include "TClassEdit.h"
#include "TClass.h"
#include "TError.h"
#include "TROOT.h"

namespace std {} using namespace std;

enum {
   // Those 'bits' are used in conjunction with CINT's bit to store the 'type'
   // info into one int
   R__BIT_ISSTRING   = 0x20000000,  // We can optimized a value operation when the content are strings
   R__BIT_ISTSTRING  = 0x40000000
};

//______________________________________________________________________________
TEmulatedVectorProxy::TEmulatedVectorProxy(const char* classname) :
   TVirtualCollectionProxy(gROOT->GetClass(classname)),
   fProxiedName(classname),
   fValueClass(0), fProxied(0), fSize(-1), fCase(0), fKind(kNoType_t), fNarr(0),
   fArr(0)
{
   // Build a proxy for an emulated vector whose type is 'classname'.

   Init();
}

//______________________________________________________________________________
TEmulatedVectorProxy::TEmulatedVectorProxy(TClass *collectionClass) :
   TVirtualCollectionProxy(collectionClass),
   fProxiedName(collectionClass->GetName()),
   fValueClass(0), fProxied(0), fSize(-1), fCase(0), fKind(kNoType_t), fNarr(0),
   fArr(0)
{
   // Build a proxy for an emulated vector whose type is described by
   // 'collectionClass'.
 
   Init();
}

//______________________________________________________________________________
void TEmulatedVectorProxy::Init() 
{
   // 'Parse' the typename to store easier to use descrition of the
   // vector class.

   string shortname = TClassEdit::ShortType(fProxiedName.Data(),
                                             TClassEdit::kDropAlloc);
   string inside = TClassEdit::ShortType(shortname.c_str(), 
                                           TClassEdit::kInnerClass);
   string insideTypename;

   fSize = -1;

   if ( inside == "string" ) {
      fCase = R__BIT_ISSTRING;
      insideTypename = "string";
      fValueClass = gROOT->GetClass("string");

   } else if ( inside == "string*" ) {

      fCase = R__BIT_ISSTRING|G__BIT_ISPOINTER;
      insideTypename = "string";
      fValueClass = gROOT->GetClass("string");
      fSize = sizeof(void*);

   } else {
      
      G__TypeInfo ti(inside.c_str());
      insideTypename = TClassEdit::ShortType( inside.c_str(),
                                              TClassEdit::kDropTrailStar );     
      if (!ti.IsValid()) {
         // CINT doesn't know anything about the content :(

         if (insideTypename != inside) {
            fCase |= G__BIT_ISPOINTER;
            fSize = sizeof(void*);
         }

         fValueClass = gROOT->GetClass(insideTypename.c_str());

         if (fValueClass) fCase |= G__BIT_ISCLASS;
         else {
            // either we have an Emulated enum or a really unknow class!
            // let's just claim its an enum :(
            fCase = G__BIT_ISENUM;
            fSize = sizeof(Int_t);
            fKind = kInt_t;
         }
         
      } else {

         long P = ti.Property();
         
         if(P&G__BIT_ISPOINTER) {
            fSize = sizeof(void*);
         }
         
         if(P&G__BIT_ISSTRUCT) {
            P |= G__BIT_ISCLASS;
         }
         if(P&G__BIT_ISCLASS) {
            fValueClass = gROOT->GetClass(insideTypename.c_str());
            Assert(fValueClass);
            
         } else if (P&G__BIT_ISFUNDAMENTAL) {
            
            
            TDataType *fundType = gROOT->GetType( insideTypename.c_str() );
            fKind = (EDataType)fundType->GetType();
            
            Assert(fKind>0 && fKind<20);

            fSize = ti.Size();
            
         } else if (P&G__BIT_ISENUM) {
            
            //          fValueClass = gROOT->GetClass("ROOT::INT_T");
            fSize = sizeof(Int_t);
            fKind = kInt_t;
         }
         
         fCase = P & (G__BIT_ISPOINTER|G__BIT_ISFUNDAMENTAL|G__BIT_ISENUM|G__BIT_ISCLASS);
         
         if (fValueClass==TString::Class() && (fCase&G__BIT_ISPOINTER)) {
            fCase |= R__BIT_ISTSTRING;
         }
      }
      
   }
   
   if (fSize==-1) {
      if (fValueClass==0) {
         Fatal("TEmulatedVectorProxy","Could not find %s!",inside.c_str());
      }
      fSize = fValueClass->Size();
   }
   
}

//______________________________________________________________________________
void   *TEmulatedVectorProxy::At(UInt_t idx) 
{
   // Return the address of the value at index 'idx'
   
   if (!fProxied) return 0;
   
   vector<char> &V = *((vector<char>*)fProxied);
   UInt_t shift = idx * fSize;
   if (shift >= V.size()) return 0;
   char *start = (char*)(&V[0]);
   return start+shift; // &(*(V.begin()+shift)); 
}

//______________________________________________________________________________
void TEmulatedVectorProxy::Clear(const char *opt)
{
   // Clear the emulated vector.  
   // If 'opt' contain 'f', force the running of the destructors.

   if (!Size()) return;

   Int_t force = 1;
   if (!opt || strstr(opt,"f")==0) force = 0;

   if (! ( fCase == G__BIT_ISFUNDAMENTAL
           || fCase == G__BIT_ISENUM ) ) {

      Destruct(0,99999999,force);
   }

   ((vector<char>*)fProxied)->clear();

   return;

}

void TEmulatedVectorProxy::Destruct(int first,int last,Int_t forceDelete) 
{
   // Helper fuction to run destructor and deletion when needed.

   if (fCase == G__BIT_ISFUNDAMENTAL) return;
   if (fCase == G__BIT_ISENUM)        return;

   void *obj = 0;
   for(int i=first; i<last; ++i) {
      obj = At(i);
      if (obj==0) break;
      
      switch(fCase) {
         case R__BIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:;
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:;
            if (forceDelete) fValueClass->Destructor(*((void**)obj),0);break;

         case R__BIT_ISSTRING:;
         case G__BIT_ISCLASS:;
            if (forceDelete) fValueClass->Destructor(obj,1);   break; 

         case R__BIT_ISSTRING|G__BIT_ISPOINTER:;
            if (forceDelete) delete *((string**)obj);   break; 

         default: Assert(0);
      } //end switch
   } //end for
}

//______________________________________________________________________________
TClass *TEmulatedVectorProxy::GetValueClass()
{
   //inner TClass

   return fValueClass;
}

//______________________________________________________________________________
EDataType TEmulatedVectorProxy::GetType() 
{
   // If the content is a simple numerical value, return its type (see TDataType)
   
   return fKind;
}

//______________________________________________________________________________
void  **TEmulatedVectorProxy::GetPtrArray()
{
   // Return a contiguous array of pointer to the values in the container.

   if (gDebug>1) Info("TEmulatedVectorProxy::GetPtrArray","called for %s at %p",fClass->GetName(),fProxied);

   if (HasPointers()) return (void**)At(0);

   unsigned int n = Size();
   if (n >= fNarr) {
      delete [] fArr;
      fNarr =  int(n*1.3) + 10;
      fArr  = new void*[fNarr+1];
   }

   fArr[0] = At(0);
   for (unsigned int i=1;i<n;i++)   { fArr[i] = (char*)(fArr[i-1]) + fSize;}

   fArr[n]=0;
   return fArr;
}

//______________________________________________________________________________
Bool_t TEmulatedVectorProxy::HasPointers() const 
{
   // Return true if the objects containers are pointers;
   
   return fCase&G__BIT_ISPOINTER;
}

//______________________________________________________________________________
void    *TEmulatedVectorProxy::New() const 
{
   // Return a new vector object

   return (void*)new  vector<char>;
}

//______________________________________________________________________________
void    *TEmulatedVectorProxy::New(void *arena) const 
{
   // Run the vector constructor

   return new( (vector<char>*)arena )  vector<char>;
}

//______________________________________________________________________________
void    TEmulatedVectorProxy::Resize(UInt_t n, Bool_t forceDelete)
{
   // Resize the container

   int    kase = fCase;

   UInt_t nold = Size();
   if (n==nold) return;

   if (n<nold) Destruct(n-1,nold,forceDelete);
   else        Clear(forceDelete?"f":"");

   ((vector<char>  *)fProxied)->resize(n*fSize);

   switch(kase) {
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:;
         case G__BIT_ISCLASS:;
         case R__BIT_ISSTRING:;
         case R__BIT_ISSTRING|G__BIT_ISPOINTER:;
            // We will loop for those case
            break;

         default:
            // for the other cases we don't have to initialized anything!
            return;
      }//end switch

   if (n < nold)  return;
   
   void *obj,**abj;
   for (UInt_t idx=0; idx<n; ++idx) {
      obj = At(idx);
      abj = (void **)obj;

      switch(kase) {
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:;
            *abj = 0;
            // NOTE: I don;t understand the logic here....
            // why is the object created?
            if (forceDelete) *abj = fValueClass->New();  break;

         case R__BIT_ISSTRING:;
         case G__BIT_ISCLASS:;
            fValueClass->New(obj);    
            break; 

         case R__BIT_ISSTRING|G__BIT_ISPOINTER:;
            *abj = 0;
            if (forceDelete) *abj = new string;  break; 

         default: Assert(0);
      }//end switch
   }//end for

}
 
//______________________________________________________________________________
UInt_t  TEmulatedVectorProxy::Size() const
{
   // Return the current size of the container
   
   if (!fProxied || fSize==0) return 0;

   vector<char> &vec = *((vector<char>*)fProxied);
   return (vec.size()/fSize);
}

//______________________________________________________________________________
UInt_t TEmulatedVectorProxy::Sizeof() const 
{
   // Return the sizeof the collection object. 

   return sizeof(vector<char>);
}


//______________________________________________________________________________
void   TEmulatedVectorProxy::Streamer(TBuffer &R__b)
{
#define DOLOOP for(idx=0, arr=At(idx); idx<R__n; ++idx,arr=At(idx))

   void *arr =0;
   Int_t idx;
   
   if (gDebug>1) Info("TEmulatedVectorProxy::Streamer","called for %s at %p",fClass->GetName(),fProxied);
   if (R__b.IsReading()) {  //Read mode
      
      Int_t R__n=0;   
      R__b >> R__n;
      Assert(R__n>=0 && R__n < 1000000);
      Resize(R__n, true);
      
      switch (fCase) {
         
         case G__BIT_ISFUNDAMENTAL:;
         case G__BIT_ISENUM:;
            arr = At(0); 
            switch(fKind) {
               case kChar_t:   {R__b.ReadFastArray((Char_t  *)arr,R__n);} break;
               case kShort_t:  {R__b.ReadFastArray((Short_t *)arr,R__n);} break;
               case kInt_t:    {R__b.ReadFastArray((Int_t   *)arr,R__n);} break;
               case kLong_t:   {R__b.ReadFastArray((Long_t  *)arr,R__n);} break;
               case kLong64_t: {R__b.ReadFastArray((Long_t  *)arr,R__n);} break;
               case kFloat_t:  {R__b.ReadFastArray((Float_t *)arr,R__n);} break;
               case kDouble_t: {R__b.ReadFastArray((Double_t*)arr,R__n);} break;

               // case kBool_t:   
               case kUChar_t:   {R__b.ReadFastArray((Char_t  *)arr,R__n);} break;

               case kUShort_t:  {R__b.ReadFastArray((Short_t *)arr,R__n);} break;
               case kUInt_t:    {R__b.ReadFastArray((Int_t   *)arr,R__n);} break;
               case kULong_t:   {R__b.ReadFastArray((Long_t  *)arr,R__n);} break;
               case kULong64_t: {R__b.ReadFastArray((Long_t  *)arr,R__n);} break;

               case kDouble32_t: {
                  Double_t *where = (Double_t*)arr;
                  for(Int_t k=0; k<R__n; ++k) {
                     Float_t afloat; R__b >> afloat; where[k] = (Double_t)afloat; 
                   }
               } break;
               case kchar:
               case kNoType_t:
               case kOther_t:
                  Error("TEmulatedVectorProxy","Type %d is not supported yet!\n",fKind);
                  break;
            }
            break;
            
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:;
            DOLOOP { *((void**)arr) = (void*)R__b.ReadObject(fValueClass);} break;
            
         case G__BIT_ISCLASS:;
            DOLOOP {
               R__b.StreamObject((void*)arr,fValueClass);
            } break;
            
         case R__BIT_ISSTRING:;
            DOLOOP {
               TString R__str;
               R__str.Streamer(R__b);
               *((string*)arr) = R__str.Data();
            } 
            break;
            
         case R__BIT_ISSTRING|G__BIT_ISPOINTER:;
            DOLOOP {
               TString R__str;
               R__str.Streamer(R__b);;
               *((void**)arr) = new string(R__str.Data());
            }  
            break;
            
         case R__BIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:;
            DOLOOP {
               TString **ptr = (TString**)arr;
               *ptr = new TString;
               (**ptr).Streamer(R__b);
            }  break;

         default: Assert(0);
      }//end of switch
      
      
   } else {     //Write case

      Int_t R__n = Size();
      R__b << R__n;
      switch (fCase) {
         
         case G__BIT_ISFUNDAMENTAL:;
         case G__BIT_ISENUM:;
            arr = At(0);
            switch(fKind) {
               case kChar_t:   {R__b.WriteFastArray((Char_t  *)arr,R__n);}  break;
               case kShort_t:  {R__b.WriteFastArray((Short_t *)arr,R__n);}  break;
               case kInt_t:    {R__b.WriteFastArray((Int_t   *)arr,R__n);}  break;
               case kLong_t:   {R__b.WriteFastArray((Long_t  *)arr,R__n);}  break;
               case kLong64_t: {R__b.WriteFastArray((Long_t  *)arr,R__n);}  break;
               case kFloat_t:  {R__b.WriteFastArray((Float_t *)arr,R__n);}  break;
               case kDouble_t: {R__b.WriteFastArray((Double_t*)arr,R__n);}  break;

               // case kBool_t:
               case kUChar_t:   {R__b.WriteFastArray((Char_t  *)arr,R__n);}  break;
               case kUShort_t:  {R__b.WriteFastArray((Short_t *)arr,R__n);}  break;
               case kUInt_t:    {R__b.WriteFastArray((Int_t   *)arr,R__n);}  break;
               case kULong_t:   {R__b.WriteFastArray((Long_t  *)arr,R__n);}  break;
               case kULong64_t: {R__b.WriteFastArray((Long_t  *)arr,R__n);}  break;
                  
               case kDouble32_t: {
                  Double_t *where = (Double_t*)arr;
                  for(int k=0; k<R__n; ++k) {
                     R__b << Float_t(where[k]); 
                  }
                  break;
               }
               case kchar:
               case kNoType_t:
               case kOther_t:
                  Error("TEmulatedVectorProxy","Type %d is not supported yet!\n",fKind);
                  break;
            }
            // For simple type we know that the memory is consecutive
            // and we save in only ONE iteration instead of R__n.
            // so let's break
            break;
            
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:;
            DOLOOP {R__b.WriteObjectAny(*((void**)arr),fValueClass);}   break;
            
         case G__BIT_ISCLASS:;
            DOLOOP {R__b.StreamObject((void*)arr,fValueClass);}  break;
            
         case R__BIT_ISSTRING:;
            DOLOOP {TString R__str(((string*)arr)->c_str());
            R__str.Streamer(R__b);}     break;
            
         case R__BIT_ISSTRING|G__BIT_ISPOINTER:;
            DOLOOP {TString R__str((*((string**)arr))->c_str());
            R__str.Streamer(R__b);}     break;
            
         case R__BIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:;
            DOLOOP {TString *R__str = *((TString**)arr);
            R__str->Streamer(R__b);
            }  break;
            
         default: {
            Assert(0);
         }
      }//end of switch
      
   }//end of write
      
#undef DOLOOP 

}
