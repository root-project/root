#include "TBuffer.h"
#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TRef.h"
#include "TProcessID.h"
#include "TStreamer.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TVirtualCollectionProxy.h"

//==========CPP macros

#define DOLOOP for(Int_t k=0; k<narr; ++k) 

#define ReadBasicTypeElem(name,index)           \
   {                                            \
      name *x=(name*)(arr[index]+ioffset);      \
      b >> *x;                                  \
   }

#define ReadBasicType(name)                     \
   {                                            \
      ReadBasicTypeElem(name,0);                \
   }

#define ReadBasicTypeLoop(name)                                \
   {                                                           \
      for(Int_t k=0; k<narr; ++k) ReadBasicTypeElem(name,k);   \
   }

#define ReadBasicArrayElem(name,index)          \
   {                                            \
      name *x=(name*)(arr[index]+ioffset);      \
      b.ReadFastArray(x,fLength[i]);            \
   }

#define ReadBasicArray(name)                     \
   {                                             \
      ReadBasicArrayElem(name,0);                \
   }

#define ReadBasicArrayLoop(name)                               \
   {                                                           \
      for(Int_t k=0; k<narr; ++k) ReadBasicArrayElem(name,k)   \
   } 

#define ReadBasicPointerElem(name,index)        \
   {                                            \
      Char_t isArray;                           \
      b >> isArray;                             \
      Int_t *l = (Int_t*)(arr[index]+imethod);  \
      name **f = (name**)(arr[index]+ioffset);  \
      int j;                                    \
      for(j=0;j<fLength[i];j++) {               \
         delete [] f[j];                        \
         f[j] = 0; if (*l <=0) continue;        \
         f[j] = new name[*l];                   \
         b.ReadFastArray(f[j],*l);              \
      }                                         \
   }

#define ReadBasicPointer(name)                  \
   {                                            \
      const int imethod = fMethod[i]+eoffset;   \
      ReadBasicPointerElem(name,0);             \
   }

#define ReadBasicPointerLoop(name)              \
   {                                            \
      int imethod = fMethod[i]+eoffset;         \
      for(int k=0; k<narr; ++k) {               \
         ReadBasicPointerElem(name,k);          \
      }                                         \
   }

#define SkipCBasicType(name)                    \
   {                                            \
      name dummy;                               \
      DOLOOP{ b >> dummy; }                     \
      break;                                    \
   }

#define SkipCBasicArray(name)                           \
   {  name dummy;                                       \
      DOLOOP{                                           \
         for (Int_t j=0;j<fLength[i];j++) b >> dummy;   \
      }                                                 \
      break;                                            \
   }

#define SkipCBasicPointer(name)                                         \
   {                                                                    \
      Int_t *n = (Int_t*)(arr[0]+imethod);                              \
      Int_t l = b.Length();                                             \
      int len = aElement->GetArrayDim()?aElement->GetArrayLength():1;   \
      b.SetBufferOffset(l+1+narr*(*n)*sizeof( name )*len);              \
      break;                                                            \
   }

//______________________________________________________________________________
template <class T>
Int_t TStreamerInfo::ReadBufferSkip(TBuffer &b, const T &arr, Int_t i, Int_t kase, 
                                    TStreamerElement *aElement, Int_t narr,
                                    Int_t eoffset)
{
   //  Skip elements in a TClonesArray

   UInt_t start, count;

//   Int_t ioffset = fOffset[i]+eoffset;
   Int_t imethod = fMethod[i]+eoffset;
   switch (kase) {

      // skip basic types
      case kSkip + kChar:      SkipCBasicType(Char_t);
      case kSkip + kShort:     SkipCBasicType(Short_t);
      case kSkip + kInt:       SkipCBasicType(Int_t);
      case kSkip + kLong:      SkipCBasicType(Long_t);
      case kSkip + kLong64:    SkipCBasicType(Long64_t);
      case kSkip + kFloat:     SkipCBasicType(Float_t);
      case kSkip + kDouble:    SkipCBasicType(Double_t);
      case kSkip + kDouble32:  SkipCBasicType(Float_t)
      case kSkip + kUChar:     SkipCBasicType(UChar_t);
      case kSkip + kUShort:    SkipCBasicType(UShort_t);
      case kSkip + kUInt:      SkipCBasicType(UInt_t);
      case kSkip + kULong:     SkipCBasicType(ULong_t);
      case kSkip + kULong64:   SkipCBasicType(ULong64_t);
      case kSkip + kBits:      SkipCBasicType(UInt_t);

         // skip array of basic types  array[8]
      case kSkipL + kChar:     SkipCBasicArray(Char_t);
      case kSkipL + kShort:    SkipCBasicArray(Short_t);
      case kSkipL + kInt:      SkipCBasicArray(Int_t);
      case kSkipL + kLong:     SkipCBasicArray(Long_t);
      case kSkipL + kLong64:   SkipCBasicArray(Long64_t);
      case kSkipL + kFloat:    SkipCBasicArray(Float_t);
      case kSkipL + kDouble32: SkipCBasicArray(Float_t)
      case kSkipL + kDouble:   SkipCBasicArray(Double_t);
      case kSkipL + kUChar:    SkipCBasicArray(UChar_t);
      case kSkipL + kUShort:   SkipCBasicArray(UShort_t);
      case kSkipL + kUInt:     SkipCBasicArray(UInt_t);
      case kSkipL + kULong:    SkipCBasicArray(ULong_t);
      case kSkipL + kULong64:  SkipCBasicArray(ULong64_t);

   // skip pointer to an array of basic types  array[n]
      case kSkipP + kChar:     SkipCBasicPointer(Char_t);
      case kSkipP + kShort:    SkipCBasicPointer(Short_t);
      case kSkipP + kInt:      SkipCBasicPointer(Int_t);
      case kSkipP + kLong:     SkipCBasicPointer(Long_t);
      case kSkipP + kLong64:   SkipCBasicPointer(Long64_t);
      case kSkipP + kFloat:    SkipCBasicPointer(Float_t);
      case kSkipP + kDouble:   SkipCBasicPointer(Double_t);
      case kSkipP + kDouble32: SkipCBasicPointer(Float_t)
      case kSkipP + kUChar:    SkipCBasicPointer(UChar_t);
      case kSkipP + kUShort:   SkipCBasicPointer(UShort_t);
      case kSkipP + kUInt:     SkipCBasicPointer(UInt_t);
      case kSkipP + kULong:    SkipCBasicPointer(ULong_t);
      case kSkipP + kULong64:  SkipCBasicPointer(ULong64_t);

         // skip char*
      case kSkip + kCharStar: {
         DOLOOP {
            Int_t nch; b >> nch;
            Int_t l = b.Length();
            b.SetBufferOffset(l+4+nch);
         }
         break;
      }
      
      // skip Class *  derived from TObject with comment field  //->
      case kSkip + kObjectp: {
         DOLOOP {
            b.ReadVersion(&start, &count);
            b.SetBufferOffset(start+count+sizeof(UInt_t));
         }
         break;
      }
     
      // skip Class*   derived from TObject
      case kSkip + kObjectP: {
         DOLOOP{
            for (Int_t j=0;j<fLength[i];j++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
         }
         break;
      }

      // skip array counter //[n]
      case kSkip + kCounter: {
         DOLOOP {
            //Int_t *counter = (Int_t*)fMethod[i];
            //b >> *counter;
            Int_t dummy; b >> dummy;
         }
         break;
      }

      // skip Class    derived from TObject
      case kSkip + kObject:  {
         if (fClass == TRef::Class()) {
            TRef refjunk;
            DOLOOP{ refjunk.Streamer(b);}
         } else {
            DOLOOP{
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
         }
         break;
      }

      // skip Special case for TString, TObject, TNamed
      case kSkip + kTString: {
         TString s;
         DOLOOP {
            s.Streamer(b);
         }
         break;
      }
      case kSkip + kTObject: {
         TObject x;
         DOLOOP {
            x.Streamer(b);
         }
         break;
      }
      case kSkip + kTNamed:  {
         TNamed n;
         DOLOOP {
            n.Streamer(b);
         }
      break;
      }

      // skip Class *  not derived from TObject with comment field  //->
      case kSkip + kAnyp: {
         DOLOOP {
            b.ReadVersion(&start, &count);
            b.SetBufferOffset(start+count+sizeof(UInt_t));
         }
         break;
      }

      // skip Class*   not derived from TObject
      case kSkip + kAnyP: {
         DOLOOP {
            for (Int_t j=0;j<fLength[i];j++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
         }
         break;
      }

      // skip Any Class not derived from TObject
      case kSkip + kAny:     {
         DOLOOP {
            b.ReadVersion(&start, &count);
            b.SetBufferOffset(start+count+sizeof(UInt_t));
         }
         break;
      }

      // skip Base Class
      case kSkip + kBase:    {
         DOLOOP {
            b.ReadVersion(&start, &count);
            b.SetBufferOffset(start+count+sizeof(UInt_t));
         }
         break;
      }

      case kSkip + kStreamLoop:
      case kSkip + kStreamer: {
         UInt_t start,count;
         DOLOOP {
            b.ReadVersion(&start,&count);
            b.SetBufferOffset(start + count + sizeof(UInt_t));}
         break;
      }
      default:
         //Error("ReadBufferClones","The element type %d is not supported yet\n",fType[i]);
         return -1;
   }
   return 0;
}

//______________________________________________________________________________
template <class T>
Int_t TStreamerInfo::ReadBufferConv(TBuffer &b, const T &arr,  Int_t i, Int_t kase, 
                                    TStreamerElement *aElement, Int_t narr, 
                                    Int_t eoffset)
{
   //  Convert elements of a TClonesArray

   Int_t ioffset = eoffset+fOffset[i];

#define ConvCBasicType(name) \
   { \
      DOLOOP { \
         name u; \
         b >> u; \
         switch(fNewType[i]) { \
            case kChar:    {Char_t   *x=(Char_t*)(arr[k]+ioffset);   *x = (Char_t)u;   break;} \
            case kShort:   {Short_t  *x=(Short_t*)(arr[k]+ioffset);  *x = (Short_t)u;  break;} \
            case kInt:     {Int_t    *x=(Int_t*)(arr[k]+ioffset);    *x = (Int_t)u;    break;} \
            case kLong:    {Long_t   *x=(Long_t*)(arr[k]+ioffset);   *x = (Long_t)u;   break;} \
            case kLong64:  {Long64_t *x=(Long64_t*)(arr[k]+ioffset); *x = (Long64_t)u;   break;} \
            case kFloat:   {Float_t  *x=(Float_t*)(arr[k]+ioffset);  *x = (Float_t)u;  break;} \
            case kDouble:  {Double_t *x=(Double_t*)(arr[k]+ioffset); *x = (Double_t)u; break;} \
            case kDouble32:{Double_t *x=(Double_t*)(arr[k]+ioffset); *x = (Double_t)u; break;} \
            case kUChar:   {UChar_t  *x=(UChar_t*)(arr[k]+ioffset);  *x = (UChar_t)u;  break;} \
            case kUShort:  {UShort_t *x=(UShort_t*)(arr[k]+ioffset); *x = (UShort_t)u; break;} \
            case kUInt:    {UInt_t   *x=(UInt_t*)(arr[k]+ioffset);   *x = (UInt_t)u;   break;} \
            case kULong:   {ULong_t  *x=(ULong_t*)(arr[k]+ioffset);  *x = (ULong_t)u;  break;} \
            case kULong64: {ULong64_t*x=(ULong64_t*)(arr[k]+ioffset);*x = (ULong64_t)u;  break;} \
         } \
      } break; \
   }

#define ConvCBasicArray(name) \
   { \
      name reader; \
      int len = fLength[i]; \
      int newtype = fNewType[i]%20; \
      DOLOOP { \
          switch(newtype) { \
             case kChar:   {Char_t *f=(Char_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Char_t)reader;} \
                            break; } \
             case kShort:  {Short_t *f=(Short_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Short_t)reader;} \
                            break; } \
             case kInt:    {Int_t *f=(Int_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Int_t)reader;} \
                            break; } \
             case kLong:   {Long_t *f=(Long_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Long_t)reader;} \
                            break; } \
             case kLong64: {Long64_t *f=(Long64_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Long64_t)reader;} \
                            break; } \
             case kFloat:  {Float_t *f=(Float_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Float_t)reader;} \
                            break; } \
             case kDouble: {Double_t *f=(Double_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Double_t)reader;} \
                            break; } \
             case kDouble32:{Double_t *f=(Double_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (Double_t)reader;} \
                            break; } \
             case kUChar:  {UChar_t *f=(UChar_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (UChar_t)reader;} \
                            break; } \
             case kUShort: {UShort_t *f=(UShort_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (UShort_t)reader;} \
                            break; } \
             case kUInt:   {UInt_t *f=(UInt_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (UInt_t)reader;} \
                            break; } \
             case kULong:  {ULong_t *f=(ULong_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (ULong_t)reader;} \
                            break; } \
             case kULong64:{ULong64_t *f=(ULong64_t*)(arr[k]+ioffset); \
                            for (Int_t j=0;j<len;j++) {b >> reader; f[j] = (ULong64_t)reader;} \
                            break; } \
          } \
      } break; \
   }

#define ConvCBasicPointer(name) \
   { \
   Char_t isArray; \
      int len = aElement->GetArrayDim()?aElement->GetArrayLength():1; \
      int j; \
      name u; \
      int newtype = fNewType[i] %20; \
      Int_t imethod = fMethod[i]+eoffset;\
      DOLOOP { \
         b >> isArray; \
         Int_t *l = (Int_t*)(arr[k]+imethod); \
         switch(newtype) { \
            case kChar:   {Char_t   **f=(Char_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Char_t[*l]; Char_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Char_t)u;} \
                       } break;} \
            case kShort:  {Short_t  **f=(Short_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Short_t[*l]; Short_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Short_t)u;} \
                       } break;} \
            case kInt:    {Int_t    **f=(Int_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Int_t[*l]; Int_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Int_t)u;} \
                       } break;} \
            case kLong:   {Long_t   **f=(Long_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Long_t[*l]; Long_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Long_t)u;} \
                       } break;} \
            case kLong64: {Long64_t   **f=(Long64_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Long64_t[*l]; Long64_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Long64_t)u;} \
                       } break;} \
            case kFloat:  {Float_t  **f=(Float_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Float_t[*l]; Float_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Float_t)u;} \
                       } break;} \
            case kDouble: {Double_t **f=(Double_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Double_t[*l]; Double_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Double_t)u;} \
                       } break;} \
            case kDouble32: {Double_t **f=(Double_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new Double_t[*l]; Double_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (Double_t)u;} \
                       } break;} \
            case kUChar:  {UChar_t  **f=(UChar_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new UChar_t[*l]; UChar_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (UChar_t)u;} \
                       } break;} \
            case kUShort: {UShort_t **f=(UShort_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new UShort_t[*l]; UShort_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (UShort_t)u;} \
                       } break;} \
            case kUInt:   {UInt_t   **f=(UInt_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new UInt_t[*l]; UInt_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (UInt_t)u;} \
                       } break;} \
            case kULong:  {ULong_t  **f=(ULong_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new ULong_t[*l]; ULong_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (ULong_t)u;} \
                       } break;} \
            case kULong64:{ULong64_t  **f=(ULong64_t**)(arr[k]+ioffset); \
                       for (j=0;j<len;j++) { \
                          delete [] f[j]; f[j] = 0; if (*l ==0) continue; \
                          f[j] = new ULong64_t[*l]; ULong64_t *af = f[j]; \
                          for (Int_t j=0;j<*l;j++) {b >> u; af[j] = (ULong64_t)u;} \
                       } break;} \
         } \
      } break; \
   }

   //============

   switch (kase) {

      // convert basic types
      case kConv + kChar:    ConvCBasicType(Char_t);
      case kConv + kShort:   ConvCBasicType(Short_t);
      case kConv + kInt:     ConvCBasicType(Int_t);
      case kConv + kLong:    ConvCBasicType(Long_t);
      case kConv + kLong64:  ConvCBasicType(Long64_t);
      case kConv + kFloat:   ConvCBasicType(Float_t);
      case kConv + kDouble:  ConvCBasicType(Double_t);
      case kConv + kDouble32:ConvCBasicType(Float_t);
      case kConv + kUChar:   ConvCBasicType(UChar_t);
      case kConv + kUShort:  ConvCBasicType(UShort_t);
      case kConv + kUInt:    ConvCBasicType(UInt_t);
      case kConv + kULong:   ConvCBasicType(ULong_t);
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case kConv + kULong64: ConvCBasicType(Long64_t)
#else
      case kConv + kULong64: ConvCBasicType(ULong64_t)
#endif
      case kConv + kBits:    ConvCBasicType(UInt_t);
         
         // convert array of basic types  array[8]
      case kConvL + kChar:    ConvCBasicArray(Char_t);
      case kConvL + kShort:   ConvCBasicArray(Short_t);
      case kConvL + kInt:     ConvCBasicArray(Int_t);
      case kConvL + kLong:    ConvCBasicArray(Long_t);
      case kConvL + kLong64:  ConvCBasicArray(Long64_t);
      case kConvL + kFloat:   ConvCBasicArray(Float_t);
      case kConvL + kDouble:  ConvCBasicArray(Double_t);
      case kConvL + kDouble32:ConvCBasicArray(Float_t);
      case kConvL + kUChar:   ConvCBasicArray(UChar_t);
      case kConvL + kUShort:  ConvCBasicArray(UShort_t);
      case kConvL + kUInt:    ConvCBasicArray(UInt_t);
      case kConvL + kULong:   ConvCBasicArray(ULong_t);
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case kConvL + kULong64: ConvCBasicArray(Long64_t)
#else
      case kConvL + kULong64: ConvCBasicArray(ULong64_t)
#endif

   // convert pointer to an array of basic types  array[n]
      case kConvP + kChar:    ConvCBasicPointer(Char_t);
      case kConvP + kShort:   ConvCBasicPointer(Short_t);
      case kConvP + kInt:     ConvCBasicPointer(Int_t);
      case kConvP + kLong:    ConvCBasicPointer(Long_t);
      case kConvP + kLong64:  ConvCBasicPointer(Long64_t);
      case kConvP + kFloat:   ConvCBasicPointer(Float_t);
      case kConvP + kDouble:  ConvCBasicPointer(Double_t);
      case kConvP + kDouble32:ConvCBasicPointer(Float_t);
      case kConvP + kUChar:   ConvCBasicPointer(UChar_t);
      case kConvP + kUShort:  ConvCBasicPointer(UShort_t);
      case kConvP + kUInt:    ConvCBasicPointer(UInt_t);
      case kConvP + kULong:   ConvCBasicPointer(ULong_t);
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case kConvP + kULong64: ConvCBasicPointer(Long64_t)
#else
      case kConvP + kULong64: ConvCBasicPointer(ULong64_t)
#endif

      default:
         //Error("ReadBufferClones","The element type %d is not supported yet\n",fType[i]);
         return -1;

   }

   return 0;
}

//______________________________________________________________________________
template <class T> 
Int_t TStreamerInfo::ReadBuffer(TBuffer &b, const T &arr, Int_t first, 
                                Int_t narr, Int_t eoffset, Int_t arrayMode)
{
   //  Deserialize information from buffer b into object at pointer
   //  if (arrayMode & 1) ptr is a pointer to array of pointers to the objects
   //  otherwise it is a pointer to a pointer to a single object.
   //  This also means that T is of a type such that arr[i] is a pointer to an
   //  object.  Currently the only anticipated instantiation are for T==char**
   //  and T==TVirtualCollectionProxy

   b.IncrementLevel(this);
   
   Int_t last;

   if (!fType) {
      char *ptr = (arrayMode&1)? 0:arr[0];
      fClass->BuildRealData(ptr);
      BuildOld();
   }
   
   //loop on all active members
   
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;
   
   // In order to speed up the case where the object being written is
   // not in a collection (i.e. arrayMode is false), we actually
   // duplicate the code for the elementary types using this typeOffset.
   static const int kHaveLoop = 1024;
   const Int_t typeOffset = arrayMode ? kHaveLoop : 0;

   TClass     *cle      =0;
   TMemberStreamer *pstreamer=0;
   Int_t isPreAlloc = 0;
   for (Int_t i=first;i<last;i++) {

      b.SetStreamerElementNumber(i);      
      TStreamerElement * aElement  = (TStreamerElement*)fElem[i];      
      fgElement = aElement;
      
      const Int_t ioffset = fOffset[i]+eoffset;   

      if (gDebug > 1) {
         printf("ReadBuffer, class:%s, name=%s, fType[%d]=%d,"
                " %s, bufpos=%d, arr=%p, offset=%d\n",
                fClass->GetName(),aElement->GetName(),i,fType[i],
                aElement->ClassName(),b.Length(),arr[0], ioffset);
      }

      Int_t kase = fType[i];

      switch (kase + typeOffset) {

         // read basic types
         case kChar:               ReadBasicType(Char_t);    continue;
         case kShort:              ReadBasicType(Short_t);   continue;
         case kInt:                ReadBasicType(Int_t);     continue;
         case kLong:               ReadBasicType(Long_t);    continue;
         case kLong64:             ReadBasicType(Long64_t);  continue;
         case kFloat:              ReadBasicType(Float_t);   continue;
         case kDouble:             ReadBasicType(Double_t);  continue;
         case kUChar:              ReadBasicType(UChar_t);   continue;
         case kUShort:             ReadBasicType(UShort_t);  continue;
         case kUInt:               ReadBasicType(UInt_t);    continue;
         case kULong:              ReadBasicType(ULong_t);   continue;
         case kULong64:            ReadBasicType(ULong64_t); continue;
         case kDouble32: {
            Double_t *x=(Double_t*)(arr[0]+ioffset);
            Float_t afloat; b >> afloat; *x = (Double_t)afloat; 
            continue; 
         }
      
         case kChar   + kHaveLoop: ReadBasicTypeLoop(Char_t);    continue;
         case kShort  + kHaveLoop: ReadBasicTypeLoop(Short_t);   continue;
         case kInt    + kHaveLoop: ReadBasicTypeLoop(Int_t);     continue;
         case kLong   + kHaveLoop: ReadBasicTypeLoop(Long_t);    continue;
         case kLong64 + kHaveLoop: ReadBasicTypeLoop(Long64_t);  continue;
         case kFloat  + kHaveLoop: ReadBasicTypeLoop(Float_t);   continue;
         case kDouble + kHaveLoop: ReadBasicTypeLoop(Double_t);  continue;
         case kUChar  + kHaveLoop: ReadBasicTypeLoop(UChar_t);   continue;
         case kUShort + kHaveLoop: ReadBasicTypeLoop(UShort_t);  continue;
         case kUInt   + kHaveLoop: ReadBasicTypeLoop(UInt_t);    continue;
         case kULong  + kHaveLoop: ReadBasicTypeLoop(ULong_t);   continue;
         case kULong64+ kHaveLoop: ReadBasicTypeLoop(ULong64_t); continue;
         case kDouble32 + kHaveLoop: {
            for(Int_t k=0; k<narr; ++k) {
               Double_t *x=(Double_t*)(arr[k]+ioffset);
               Float_t afloat; b >> afloat; *x = (Double_t)afloat; 
            }
            continue; 
         }
      
         // read array of basic types  like array[8]
         case kOffsetL + kChar:   ReadBasicArray(Char_t);    continue;
         case kOffsetL + kShort:  ReadBasicArray(Short_t);   continue;
         case kOffsetL + kInt:    ReadBasicArray(Int_t);     continue;
         case kOffsetL + kLong:   ReadBasicArray(Long_t);    continue;
         case kOffsetL + kLong64: ReadBasicArray(Long64_t);  continue;
         case kOffsetL + kFloat:  ReadBasicArray(Float_t);   continue;
         case kOffsetL + kDouble: ReadBasicArray(Double_t);  continue;
         case kOffsetL + kUChar:  ReadBasicArray(UChar_t);   continue;
         case kOffsetL + kUShort: ReadBasicArray(UShort_t);  continue;
         case kOffsetL + kUInt:   ReadBasicArray(UInt_t);    continue;
         case kOffsetL + kULong:  ReadBasicArray(ULong_t);   continue;
         case kOffsetL + kULong64:ReadBasicArray(ULong64_t); continue;
         case kOffsetL + kDouble32: {
            b.ReadFastArrayDouble32((Double_t*)(arr[0]+ioffset),fLength[i]);
            continue;
         }

         case kOffsetL + kChar    + kHaveLoop: ReadBasicArrayLoop(Char_t);    continue;
         case kOffsetL + kShort   + kHaveLoop: ReadBasicArrayLoop(Short_t);   continue;
         case kOffsetL + kInt     + kHaveLoop: ReadBasicArrayLoop(Int_t);     continue;
         case kOffsetL + kLong    + kHaveLoop: ReadBasicArrayLoop(Long_t);    continue;
         case kOffsetL + kLong64  + kHaveLoop: ReadBasicArrayLoop(Long64_t);  continue;
         case kOffsetL + kFloat   + kHaveLoop: ReadBasicArrayLoop(Float_t);   continue;
         case kOffsetL + kDouble  + kHaveLoop: ReadBasicArrayLoop(Double_t);  continue;
         case kOffsetL + kUChar   + kHaveLoop: ReadBasicArrayLoop(UChar_t);   continue;
         case kOffsetL + kUShort  + kHaveLoop: ReadBasicArrayLoop(UShort_t);  continue;
         case kOffsetL + kUInt    + kHaveLoop: ReadBasicArrayLoop(UInt_t);    continue;
         case kOffsetL + kULong   + kHaveLoop: ReadBasicArrayLoop(ULong_t);   continue;
         case kOffsetL + kULong64 + kHaveLoop: ReadBasicArrayLoop(ULong64_t); continue;
         case kOffsetL + kDouble32+ kHaveLoop: {
            for(Int_t k=0; k<narr; ++k) {
               b.ReadFastArrayDouble32((Double_t*)(arr[k]+ioffset),fLength[i]);
            }
            continue;
         }
      
         // read pointer to an array of basic types  array[n]
         case kOffsetP + kChar:   ReadBasicPointer(Char_t);  continue;
         case kOffsetP + kShort:  ReadBasicPointer(Short_t);  continue;
         case kOffsetP + kInt:    ReadBasicPointer(Int_t);  continue;
         case kOffsetP + kLong:   ReadBasicPointer(Long_t);  continue;
         case kOffsetP + kLong64: ReadBasicPointer(Long64_t);  continue;
         case kOffsetP + kFloat:  ReadBasicPointer(Float_t);  continue;
         case kOffsetP + kDouble: ReadBasicPointer(Double_t);  continue;
         case kOffsetP + kUChar:  ReadBasicPointer(UChar_t);  continue;
         case kOffsetP + kUShort: ReadBasicPointer(UShort_t);  continue;
         case kOffsetP + kUInt:   ReadBasicPointer(UInt_t);  continue;
         case kOffsetP + kULong:  ReadBasicPointer(ULong_t);  continue;
         case kOffsetP + kULong64:ReadBasicPointer(ULong64_t);  continue;
         case kOffsetP + kDouble32: {
            Char_t isArray; 
            b >> isArray; 
            const int imethod = fMethod[i]+eoffset;
            Int_t *l = (Int_t*)(arr[0]+imethod);
            Double_t **f = (Double_t**)(arr[0]+ioffset); 
            int j; 
            for(j=0;j<fLength[i];j++) { 
               delete [] f[j]; 
               f[j] = 0; if (*l <=0) continue; 
               f[j] = new Double_t[*l]; 
               b.ReadFastArrayDouble32(f[j],*l);
           } 
            continue; 
         }

         case kOffsetP + kChar    + kHaveLoop: ReadBasicPointerLoop(Char_t);    continue;
         case kOffsetP + kShort   + kHaveLoop: ReadBasicPointerLoop(Short_t);   continue;
         case kOffsetP + kInt     + kHaveLoop: ReadBasicPointerLoop(Int_t);     continue;
         case kOffsetP + kLong    + kHaveLoop: ReadBasicPointerLoop(Long_t);    continue;
         case kOffsetP + kLong64  + kHaveLoop: ReadBasicPointerLoop(Long64_t);  continue;
         case kOffsetP + kFloat   + kHaveLoop: ReadBasicPointerLoop(Float_t);   continue;
         case kOffsetP + kDouble  + kHaveLoop: ReadBasicPointerLoop(Double_t);  continue;
         case kOffsetP + kUChar   + kHaveLoop: ReadBasicPointerLoop(UChar_t);   continue;
         case kOffsetP + kUShort  + kHaveLoop: ReadBasicPointerLoop(UShort_t);  continue;
         case kOffsetP + kUInt    + kHaveLoop: ReadBasicPointerLoop(UInt_t);    continue;
         case kOffsetP + kULong   + kHaveLoop: ReadBasicPointerLoop(ULong_t);   continue;
         case kOffsetP + kULong64 + kHaveLoop: ReadBasicPointerLoop(ULong64_t); continue;
         case kOffsetP + kDouble32+ kHaveLoop: {
            const int imethod = fMethod[i]+eoffset;
            for(int k=0; k<narr; ++k) {
               Char_t isArray; 
               b >> isArray; 
               Int_t *l = (Int_t*)(arr[k]+imethod);
               Double_t **f = (Double_t**)(arr[k]+ioffset); 
               int j; 
               for(j=0;j<fLength[i];j++) { 
                  delete [] f[j]; 
                  f[j] = 0; if (*l <=0) continue; 
                  f[j] = new Double_t[*l]; 
                  b.ReadFastArrayDouble32(f[j],*l);
               } 
            }
            continue; 
         }
      }

      switch (kase) {

         // char*
         case kCharStar: {
            DOLOOP {
               Int_t nch; b >> nch;
               char **f = (char**)(arr[k]+ioffset);
               delete [] *f;
               *f = 0; if (nch <=0) continue;
               *f = new char[nch+1];
               b.ReadFastArray(*f,nch); (*f)[nch] = 0;
            }
         }
         continue;
         
         // special case for TObject::fBits in case of a referenced object
         case kBits: {
            DOLOOP { 
               UInt_t *x=(UInt_t*)(arr[k]+ioffset); b >> *x;
               if ((*x & kIsReferenced) != 0) {
                  UShort_t pidf;
                  b >> pidf;
                  TFile* file = (TFile*)b.GetParent();
                  TProcessID *pid = TProcessID::ReadProcessID(pidf,file);
                  if (pid!=0) {
                     TObject *obj = (TObject*)(arr[k]+eoffset); 
                     UInt_t gpid = pid->GetUniqueID();
                     UInt_t uid = (obj->GetUniqueID() & 0xffffff) + (gpid<<24);
                     obj->SetUniqueID(uid);
                     pid->PutObjectWithID(obj);
                  }
               }
            }
         }
         continue;
         
         // array counter //[n]
         case kCounter: {
            DOLOOP {               
               Int_t *x=(Int_t*)(arr[k]+ioffset);
               b >> *x;
            }
         }
         continue;


         // Special case for TString, TObject, TNamed
         case kTString: { DOLOOP { ((TString*)(arr[k]+ioffset))->Streamer(b);         } } continue;
         case kTObject: { DOLOOP { ((TObject*)(arr[k]+ioffset))->TObject::Streamer(b);} } continue;
         case kTNamed:  { DOLOOP { ((TNamed*) (arr[k]+ioffset))->TNamed::Streamer(b) ;} } continue;

      }
      
   SWIT: 
      isPreAlloc= 0; 
      cle       = fComp[i].fClass;
      pstreamer = fComp[i].fStreamer;
      
      switch (kase) {
         
         case kAnyp:    // Class*  not derived from TObject with    comment field //->
         case kAnyp+kOffsetL:
         case kObjectp: // Class*      derived from TObject with    comment field  //->
         case kObjectp+kOffsetL:
            isPreAlloc = 1;
            
         case kObjectP: // Class* derived from TObject with no comment field NOTE: Re-added by Phil
         case kObjectP+kOffsetL:
         case kAnyP:    // Class*  not derived from TObject with no comment field NOTE:: Re-added by Phil
         case kAnyP+kOffsetL: {
            DOLOOP {
               b.ReadFastArray((void**)(arr[k]+ioffset),cle,fLength[i],isPreAlloc,pstreamer);
            }
         }
         continue;

//        case kSTLvarp:           // Variable size array of STL containers.
//             {
//                TMemberStreamer *pstreamer = fComp[i].fStreamer;
//                TClass *cl                 = fComp[i].fClass;
//                ROOT::NewArrFunc_t arraynew = cl->GetNewArray();
//                ROOT::DelArrFunc_t arraydel = cl->GetDeleteArray();
//                UInt_t start,count;
//                // Version_t v = 
//                b.ReadVersion(&start, &count, cle);
//                if (pstreamer == 0) {
//                   Int_t size = cl->Size();
//                   Int_t imethod = fMethod[i]+eoffset;
//                   DOLOOP {
//                      char **contp = (char**)(arr[k]+ioffset);
//                      const Int_t *counter = (Int_t*)(arr[k]+imethod);
//                      const Int_t sublen = (*counter);

//                      for(int j=0;j<fLength[i];++j) {
//                         if (arraydel) arraydel(contp[j]);
//                         contp[j] = 0;
//                         if (sublen<=0) continue;
//                         if (arraynew) {
//                            contp[j] = (char*)arraynew(sublen);
//                            char *cont = contp[j];
//                            for(int k=0;k<sublen;++k) {
//                               cl->Streamer( cont, b );
//                               cont += size;
//                            }
//                         } else {
//                            // Can't create an array of object
//                            Error("ReadBuffer","The element %s::%s type %d (%s) can be read because of the class does not have access to new %s[..]\n",
//                                  GetName(),aElement->GetFullName(),kase,aElement->GetTypeName(),GetName());
//                            void *cont = cl->New();
//                            for(int k=0;k<sublen;++k) {
//                               cl->Streamer( cont, b );
//                            }
//                         }
//                      }
//                   }
//                } else {
//                   DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
//                }
//                b.CheckByteCount(start,count,aElement->GetFullName());               
//             }
//             continue;

         case kSTLp:            // Pointer to Container with no virtual table (stl) and no comment
         case kSTLp + kOffsetL: // array of pointers to Container with no virtual table (stl) and no comment
            {
               UInt_t start,count;
               // Version_t v = 
               b.ReadVersion(&start, &count, cle);
               if (pstreamer == 0) {
                  DOLOOP {
                     void **contp = (void**)(arr[k]+ioffset);
                     int j;
                     for(j=0;j<fLength[i];j++) {
                        void *cont = contp[j];
                        if (cont==0) {
                           // int R__n;
                           // b >> R__n;
                           // b.SetOffset(b.GetOffset()-4); // rewind to the start of the int
                           // if (R__n) continue;
                           contp[j] = cle->New();
                           cont = contp[j];
                        }
                        cle->Streamer( cont, b );
                     }
                  }
               } else {
                  DOLOOP {(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
               }
               b.CheckByteCount(start,count,aElement->GetFullName());               
            }
            continue;

         case kSTL:                // Container with no virtual table (stl) and no comment
         case kSTL + kOffsetL:     // array of Container with no virtual table (stl) and no comment
            {
               UInt_t start,count;
               // Version_t v = 
               b.ReadVersion(&start, &count, cle);
               if (fOldVersion<3){   // case of old TStreamerInfo
                  //  Backward compatibility. Some TStreamerElement's where without
                  //  Streamer but were not removed from element list
                  if (aElement->IsBase() && aElement->IsA()!=TStreamerBase::Class()) {    
                     b.SetBufferOffset(start);  //thre is no byte count
                  }
               }
               if (pstreamer == 0) {
                  DOLOOP {
                     b.ReadFastArray((void*)(arr[k]+ioffset),cle,fLength[i],(TMemberStreamer*)0);
                  }
               } else {
                  DOLOOP {(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
               }
               b.CheckByteCount(start,count,aElement->GetFullName());               
            }
            continue;
            
         case kObject: // Class derived from TObject 
            if (cle->IsStartingWithTObject() && cle->GetClassInfo()) {
               DOLOOP {((TObject*)(arr[k]+ioffset))->Streamer(b);}
               continue; // intentionally inside the if statement.
                      // if the class does not start with its TObject part (or does
                      // not have one), we use the generic case.
            }
         case kAny:    // Class not derived from TObject 
            if (pstreamer) {
               DOLOOP {(*pstreamer)(b,arr[k]+ioffset,0);} 
            } else {
               DOLOOP { cle->Streamer(arr[k]+ioffset,b);}}
            continue;

         case kObject+kOffsetL:  {
            TFile *file = (TFile*)b.GetParent();
            if (file && file->GetVersion() < 30208) {
               // For older ROOT file we use a totally different case to treat
               // this situation, so we change 'kase' and restart.
               kase = kStreamer; 
               goto SWIT;
            }
            // there is intentionally no break/continue statement here.
            // For newer ROOT file, we always use the generic case for kOffsetL(s)
         }

         case kAny+kOffsetL: {
            DOLOOP {
               b.ReadFastArray((void*)(arr[k]+ioffset),cle,fLength[i],pstreamer);
            }
            continue;
         }
            
         // Base Class
         case kBase:     
            if (!(arrayMode&1)) {
               if(pstreamer)  {kase = kStreamer; goto SWIT;}
               DOLOOP { ((TStreamerBase*)aElement)->ReadBuffer(b,arr[k]);}
            } else {
               
               Int_t clversion = ((TStreamerBase*)aElement)->GetBaseVersion();
               cle->GetStreamerInfo(clversion)->ReadBuffer(b,arr,-1,narr,ioffset,arrayMode);
            }
            continue;
            
         case kOffsetL + kTString:
         case kOffsetL + kTObject:
         case kOffsetL + kTNamed:
         {
            //  Backward compatibility. Some TStreamerElement's where without
            //  Streamer but were not removed from element list
            UInt_t start,count;
            Version_t v = b.ReadVersion(&start, &count, cle);
            if (fOldVersion<3){   // case of old TStreamerInfo
               if (count<= 0    || v   !=  fOldVersion) {
                  b.SetBufferOffset(start);
                  continue;
               }
            }
            DOLOOP {
               b.ReadFastArray((void*)(arr[k]+ioffset),cle,fLength[i],pstreamer);
            }
            b.CheckByteCount(start,count,aElement->GetFullName());
            continue;
         }


         case kStreamer:{
            //  Backward compatibility. Some TStreamerElement's where without
            //  Streamer but were not removed from element list
            UInt_t start,count;
            Version_t v = b.ReadVersion(&start, &count, cle);
            if (fOldVersion<3){   // case of old TStreamerInfo
               if (aElement->IsBase() && aElement->IsA()!=TStreamerBase::Class()) {
                  b.SetBufferOffset(start);  //it was no byte count
               } else if (kase == kSTL || kase == kSTL+kOffsetL ||
                          count<= 0    || v   !=  fOldVersion) {
                  b.SetBufferOffset(start);
                  continue;
               } 
            }
            if (pstreamer == 0) {
               if (1 || gDebug > 0) {
                  printf("ERROR, Streamer is null\n");
                  aElement->ls(); continue;
               }
            } else {
               DOLOOP {(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
            }
            b.CheckByteCount(start,count,aElement->GetFullName());
         }
         continue;
         
         case kStreamLoop:{
            UInt_t start,count;
            b.ReadVersion(&start, &count, cle);
            if (pstreamer == 0) {
               if (1 || gDebug > 0) {
                  printf("ERROR, Streamer is null\n");
                  aElement->ls();
               }
            } else {
               int imethod = fMethod[i]+eoffset;
               DOLOOP {
                  Int_t *counter = (Int_t*)(arr[k]+imethod);
                  (*pstreamer)(b,arr[k]+ioffset,*counter);
               }  
            }
            b.CheckByteCount(start,count,aElement->GetFullName());
         }
         continue;
         
         
         
         default: {
            int ans = -1;
            if (kase >= kConv) 
               ans = ReadBufferConv(b,arr,i,kase,aElement,narr,eoffset);
            if (ans==0) continue;
            
            if (kase >= kSkip) 
               ans = ReadBufferSkip(b,arr,i,kase,aElement,narr,eoffset);
            if (ans==0) continue;
         }
         Error("ReadBuffer","The element %s::%s type %d (%s) is not supported yet\n",
               GetName(),aElement->GetFullName(),kase,aElement->GetTypeName());
         continue;
      }
   }
   b.DecrementLevel(this);

   return 0;
}

//______________________________________________________________________________
Int_t TStreamerInfo::ReadBufferSTL(TBuffer &b, TVirtualCollectionProxy *cont, 
                                   Int_t nc, Int_t first, Int_t eoffset)
{
   //  The STL vector/list is deserialized from the buffer b

   int ret = ReadBuffer(b, *cont, first,nc,eoffset,1);
   return ret;
}

//______________________________________________________________________________
Int_t TStreamerInfo::ReadBufferClones(TBuffer &b, TClonesArray *clones, 
                                      Int_t nc, Int_t first, Int_t eoffset)
{
   char **arr = (char **)clones->GetObjectRef(0); 
   return ReadBuffer(b,arr,first,nc,eoffset,1);
}

