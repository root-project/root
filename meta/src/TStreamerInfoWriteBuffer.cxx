#include "TBuffer.h"
#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TProcessID.h"
#include "TStreamer.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TVirtualCollectionProxy.h"

//==========CPP macros

#define DOLOOP for(int k=0; k<narr; ++k) 

#define WriteBasicTypeElem(name,index)          \
   {                                            \
      name *x=(name*)(arr[index]+ioffset);      \
      b << *x;                                  \
   }

#define WriteBasicType(name)                    \
   {                                            \
      WriteBasicTypeElem(name,0);               \
   }

#define WriteBasicTypeLoop(name)                            \
   {                                                        \
      for(int k=0; k<narr; ++k) WriteBasicTypeElem(name,k); \
   }

#define WriteBasicArrayElem(name,index)         \
   {                                            \
      name *x=(name*)(arr[index]+ioffset);      \
      b.WriteFastArray(x,fLength[i]);           \
   }

#define WriteBasicArray(name)                   \
   {                                            \
      WriteBasicArrayElem(name,0);              \
   }

#define WriteBasicArrayLoop(name)                              \
   {                                                           \
      for(int k=0; k<narr; ++k) WriteBasicArrayElem(name,k);   \
   }

#define WriteBasicPointerElem(name,index)       \
   {                                            \
      Int_t *l = (Int_t*)(arr[index]+imethod);  \
      name **f = (name**)(arr[index]+ioffset);  \
      name *af = *f;                            \
      if (af && *l)  b << Char_t(1);            \
      else          {b << Char_t(0); continue;} \
      int j;                                    \
      for(j=0;j<fLength[i];j++) {               \
         b.WriteFastArray(f[j],*l);             \
      }                                         \
   }

#define WriteBasicPointer(name)                 \
   {                                            \
      int imethod = fMethod[i]+eoffset;         \
      WriteBasicPointerElem(name,0);            \
   }

#define WriteBasicPointerLoop(name)             \
   {                                            \
      int imethod = fMethod[i]+eoffset;         \
      for(int k=0; k<narr; ++k) {               \
         WriteBasicPointerElem(name,k);         \
      }                                         \
   }


//______________________________________________________________________________
Int_t TStreamerInfo::WriteBufferAux(TBuffer &b, char **arr, Int_t first, 
                                    Int_t narr, Int_t eoffset, Int_t arrayMode)
{
   //  The object at pointer is serialized to the buffer b
   //  if (arrayMode & 1) ptr is a pointer to array of pointers to the objects
   //  otherwise it is a pointer to the object

   //mark this class as being used in the current file
   TagFile((TFile *)b.GetParent());

//============

   //loop on all active members
   Int_t last;
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;

   // In order to speed up the case where the object being written is not in a collection (i.e. arrayMode is false),
   // we actually duplicate the elementary types using this typeOffset.
   static const int kHaveLoop = 1024;
   const Int_t typeOffset = arrayMode ? kHaveLoop : 0;

   for (Int_t i=first;i<last;i++) {

      TStreamerElement *aElement = (TStreamerElement*)fElem[i];
      const Int_t ioffset = eoffset+fOffset[i];

      if (gDebug > 1) {
         printf("WriteBuffer, class:%s, name=%s, fType[%d]=%d, %s, bufpos=%d, arr=%p, offset=%d\n",fClass->GetName(),aElement->GetName(),i,fType[i],aElement->ClassName(),b.Length(),arr,ioffset);
      }

      switch (fType[i]+typeOffset) { 

         case kChar:                WriteBasicType(Char_t);    continue; // use 'continue' instead of 'break' to avoid running the 2nd switch
         case kShort:               WriteBasicType(Short_t);   continue;
         case kInt:                 WriteBasicType(Int_t);     continue;
         case kLong:                WriteBasicType(Long_t);    continue;
         case kLong64:              WriteBasicType(Long64_t);  continue;
         case kFloat:               WriteBasicType(Float_t);   continue;
         case kDouble:              WriteBasicType(Double_t);  continue;
         case kUChar:               WriteBasicType(UChar_t);   continue;
         case kUShort:              WriteBasicType(UShort_t);  continue;
         case kUInt:                WriteBasicType(UInt_t);    continue;
         case kULong:               WriteBasicType(ULong_t);   continue;
         case kULong64:             WriteBasicType(ULong64_t); continue;
         case kDouble32: {
            Double_t *x=(Double_t*)(arr[0]+ioffset); 
            b << Float_t(*x); 
            continue; 
         }

         case kChar    + kHaveLoop: WriteBasicTypeLoop(Char_t);    continue;
         case kShort   + kHaveLoop: WriteBasicTypeLoop(Short_t);   continue;
         case kInt     + kHaveLoop: WriteBasicTypeLoop(Int_t);     continue;
         case kLong    + kHaveLoop: WriteBasicTypeLoop(Long_t);    continue;
         case kLong64  + kHaveLoop: WriteBasicTypeLoop(Long64_t);  continue;
         case kFloat   + kHaveLoop: WriteBasicTypeLoop(Float_t);   continue;
         case kDouble  + kHaveLoop: WriteBasicTypeLoop(Double_t);  continue;
         case kUChar   + kHaveLoop: WriteBasicTypeLoop(UChar_t);   continue;
         case kUShort  + kHaveLoop: WriteBasicTypeLoop(UShort_t);  continue;
         case kUInt    + kHaveLoop: WriteBasicTypeLoop(UInt_t);    continue;
         case kULong   + kHaveLoop: WriteBasicTypeLoop(ULong_t);   continue;
         case kULong64 + kHaveLoop: WriteBasicTypeLoop(ULong64_t); continue;
         case kDouble32+ kHaveLoop: {
            for(int k=0; k<narr; ++k) {
               Double_t *x=(Double_t*)(arr[k]+ioffset); 
               b << Float_t(*x); 
            }
            continue; 
         }

         // write array of basic types  array[8]
         case kOffsetL + kChar:   WriteBasicArray(Char_t);    continue;
         case kOffsetL + kShort:  WriteBasicArray(Short_t);   continue;
         case kOffsetL + kInt:    WriteBasicArray(Int_t);     continue;
         case kOffsetL + kLong:   WriteBasicArray(Long_t);    continue;
         case kOffsetL + kLong64: WriteBasicArray(Long64_t);  continue;
         case kOffsetL + kFloat:  WriteBasicArray(Float_t);   continue;
         case kOffsetL + kDouble: WriteBasicArray(Double_t);  continue;
         case kOffsetL + kUChar:  WriteBasicArray(UChar_t);   continue;
         case kOffsetL + kUShort: WriteBasicArray(UShort_t);  continue;
         case kOffsetL + kUInt:   WriteBasicArray(UInt_t);    continue;
         case kOffsetL + kULong:  WriteBasicArray(ULong_t);   continue;
         case kOffsetL + kULong64:WriteBasicArray(ULong64_t); continue;
         case kOffsetL + kDouble32: {
            b.WriteFastArrayDouble32((Double_t*)(arr[0]+ioffset),fLength[i]);
            continue; 
         }

         case kOffsetL + kChar    + kHaveLoop: WriteBasicArrayLoop(Char_t);    continue;
         case kOffsetL + kShort   + kHaveLoop: WriteBasicArrayLoop(Short_t);   continue;
         case kOffsetL + kInt     + kHaveLoop: WriteBasicArrayLoop(Int_t);     continue;
         case kOffsetL + kLong    + kHaveLoop: WriteBasicArrayLoop(Long_t);    continue;
         case kOffsetL + kLong64  + kHaveLoop: WriteBasicArrayLoop(Long64_t);  continue;
         case kOffsetL + kFloat   + kHaveLoop: WriteBasicArrayLoop(Float_t);   continue;
         case kOffsetL + kDouble  + kHaveLoop: WriteBasicArrayLoop(Double_t);  continue;
         case kOffsetL + kUChar   + kHaveLoop: WriteBasicArrayLoop(UChar_t);   continue;
         case kOffsetL + kUShort  + kHaveLoop: WriteBasicArrayLoop(UShort_t);  continue;
         case kOffsetL + kUInt    + kHaveLoop: WriteBasicArrayLoop(UInt_t);    continue;
         case kOffsetL + kULong   + kHaveLoop: WriteBasicArrayLoop(ULong_t);   continue;
         case kOffsetL + kULong64 + kHaveLoop: WriteBasicArrayLoop(ULong64_t); continue;
         case kOffsetL + kDouble32+ kHaveLoop: {
            for(int k=0; k<narr; ++k) {
               b.WriteFastArrayDouble32((Double_t*)(arr[k]+ioffset),fLength[i]);
            }
            continue; 
         }

         // write pointer to an array of basic types  array[n]
         case kOffsetP + kChar:   WriteBasicPointer(Char_t);    continue;
         case kOffsetP + kShort:  WriteBasicPointer(Short_t);   continue;
         case kOffsetP + kInt:    WriteBasicPointer(Int_t);     continue;
         case kOffsetP + kLong:   WriteBasicPointer(Long_t);    continue;
         case kOffsetP + kLong64: WriteBasicPointer(Long64_t);  continue;
         case kOffsetP + kFloat:  WriteBasicPointer(Float_t);   continue;
         case kOffsetP + kDouble: WriteBasicPointer(Double_t);  continue;
         case kOffsetP + kUChar:  WriteBasicPointer(UChar_t);   continue;
         case kOffsetP + kUShort: WriteBasicPointer(UShort_t);  continue;
         case kOffsetP + kUInt:   WriteBasicPointer(UInt_t);    continue;
         case kOffsetP + kULong:  WriteBasicPointer(ULong_t);   continue;
         case kOffsetP + kULong64:WriteBasicPointer(ULong64_t); continue;
         case kOffsetP + kDouble32: {
            int imethod = fMethod[i]+eoffset;
            Int_t *l = (Int_t*)(arr[0]+imethod); 
            Double_t **f = (Double_t**)(arr[0]+ioffset); 
            Double_t *af = *f; 
            if (af && *l)  b << Char_t(1); 
            else          {b << Char_t(0); continue;}
            int j;
            for(j=0;j<fLength[i];j++) { 
               b.WriteFastArrayDouble32(f[j],*l);
            }  
            continue; 
         }
 
         case kOffsetP + kChar    + kHaveLoop: WriteBasicPointerLoop(Char_t);    continue;
         case kOffsetP + kShort   + kHaveLoop: WriteBasicPointerLoop(Short_t);   continue;
         case kOffsetP + kInt     + kHaveLoop: WriteBasicPointerLoop(Int_t);     continue;
         case kOffsetP + kLong    + kHaveLoop: WriteBasicPointerLoop(Long_t);    continue;
         case kOffsetP + kLong64  + kHaveLoop: WriteBasicPointerLoop(Long64_t);  continue;
         case kOffsetP + kFloat   + kHaveLoop: WriteBasicPointerLoop(Float_t);   continue;
         case kOffsetP + kDouble  + kHaveLoop: WriteBasicPointerLoop(Double_t);  continue;
         case kOffsetP + kUChar   + kHaveLoop: WriteBasicPointerLoop(UChar_t);   continue;
         case kOffsetP + kUShort  + kHaveLoop: WriteBasicPointerLoop(UShort_t);  continue;
         case kOffsetP + kUInt    + kHaveLoop: WriteBasicPointerLoop(UInt_t);    continue;
         case kOffsetP + kULong   + kHaveLoop: WriteBasicPointerLoop(ULong_t);   continue;
         case kOffsetP + kULong64 + kHaveLoop: WriteBasicPointerLoop(ULong64_t); continue;
         case kOffsetP + kDouble32+ kHaveLoop: {
            int imethod = fMethod[i]+eoffset;
            for(int k=0; k<narr; ++k) {
               Int_t *l = (Int_t*)(arr[k]+imethod); 
               Double_t **f = (Double_t**)(arr[k]+ioffset); 
               Double_t *af = *f; 
               if (af && *l)  b << Char_t(1); 
               else          {b << Char_t(0); continue;}
               int j; 
               for(j=0;j<fLength[i];j++) { 
                  b.WriteFastArrayDouble32(f[j],*l);
               }  
            }
            continue; 
         }
         
         case kCounter: { 
            Int_t *x=(Int_t*)(arr[0]+ioffset);
            b << *x;
            if (i == last-1) return x[0]; // info used by TBranchElement::FillLeaves
            continue; 
         }

         case kCounter + kHaveLoop : { 
            DOLOOP { 
               Int_t *x=(Int_t*)(arr[k]+ioffset);
               b << x[0];
            }
            continue; 
         }

         
      };
      Bool_t isPreAlloc = 0;

      switch (fType[i]) { 

         // char*
         case kCharStar: { DOLOOP {
            Int_t nch = 0;
            char **f = (char**)(arr[k]+ioffset);
            char *af = *f;
            if (af) {
               nch = strlen(af);
               b  << nch;
               b.WriteFastArray(af,nch);
            } else {
               b << nch;
            }
         } 
         continue; }
         
         // special case for TObject::fBits in case of a referenced object
         case kBits: { DOLOOP { 
            UInt_t *x=(UInt_t*)(arr[k]+ioffset); b << *x;
            if ((*x & kIsReferenced) != 0) {
               TObject *obj = (TObject*)(arr[k]+eoffset);
               TProcessID *pid = TProcessID::GetProcessWithUID(obj->GetUniqueID());
               UShort_t pidf = TProcessID::WriteProcessID(pid,(TFile *)b.GetParent());
               b << pidf;
            }
         }
         continue; }

         // Special case for TString, TObject, TNamed
         case kTString: { DOLOOP{ ((TString*)(arr[k]+ioffset))->Streamer(b);         }; continue; }
         case kTObject: { DOLOOP{ ((TObject*)(arr[k]+ioffset))->TObject::Streamer(b);}; continue; }
         case kTNamed:  { DOLOOP{ ((TNamed *)(arr[k]+ioffset))->TNamed::Streamer(b); }; continue; }

         case kAnyp:     // Class*   Class not derived from TObject and with comment field //->
         case kAnyp    + kOffsetL:            

         case kObjectp:  // Class *  Class     derived from TObject and with comment field //->
         case kObjectp + kOffsetL:

            isPreAlloc = kTRUE;

         case kAnyP:         // Class*   Class not derived from TObject and no comment
         case kAnyP    + kOffsetL:

         case kObjectP:  // Class*   Class derived from TObject
         case kObjectP + kOffsetL: {
            TClass *cl                 = fComp[i].fClass;
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            DOLOOP {
               Int_t res = b.WriteFastArray((void**)(arr[k]+ioffset),cl,fLength[i],isPreAlloc,pstreamer);
               if (res==2) {
                  Warning("WriteBuffer",
                          "The actual class of %s::%s is not available. Only the \"%s\" part will be written\n",
                          GetName(),aElement->GetName(),cl->GetName());
               }
             }
            continue;
         }

         case kAnyPnoVT:     // Class*   Class not derived from TObject and no virtual table and no comment
         case kAnyPnoVT    + kOffsetL: {
            TClass *cl                 = fComp[i].fClass;
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            DOLOOP {
               void **f = (void**)(arr[k]+ioffset);
               int j;
               for(j=0;j<fLength[i];j++) {
                  void *af = f[j];
                  if (af)  b << Char_t(1);
                  else    {b << Char_t(0); continue;}
                  if (pstreamer) (*pstreamer)(b, af, 0);
                  else cl->Streamer( af, b );
               }
            }
            continue;
         }

//          case kSTLvarp:           // Variable size array of STL containers.
//             {
//                TMemberStreamer *pstreamer = fComp[i].fStreamer;
//                TClass *cl                 = fComp[i].fClass;
//                UInt_t pos = b.WriteVersion(IsA(),kTRUE);
//                if (pstreamer == 0) {
//                   Int_t size = cl->Size();
//                   Int_t imethod = fMethod[i]+eoffset;
//                   DOLOOP {
//                      char **contp = (char**)(arr[k]+ioffset);
//                      const Int_t *counter = (Int_t*)(arr[k]+imethod);
//                      const Int_t sublen = (*counter);
                    
//                      for(int j=0;j<fLength[i];++j) {
//                         char *cont = contp[j];
//                         for(int k=0;k<sublen;++k) {
//                            cl->Streamer( cont, b );
//                            cont += size;
//                         }
//                      }
//                   }
//                } else {
//                   DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
//                }
//                b.SetByteCount(pos,kTRUE);
//             }
//             continue;


         case kSTLp:                // Pointer to container with no virtual table (stl) and no comment
         case kSTLp + kOffsetL:     // array of pointers to container with no virtual table (stl) and no comment
            {
               TClass *cl                 = fComp[i].fClass;
               TMemberStreamer *pstreamer = fComp[i].fStreamer;
               UInt_t pos = b.WriteVersion(IsA(),kTRUE);
               if (pstreamer == 0) {
                  DOLOOP {
                     char **contp = (char**)(arr[k]+ioffset);
                     for(int j=0;j<fLength[i];++j) {
                        char *cont = contp[j];
                        cl->Streamer( cont, b );
                     }
                  }
               } else {
                  DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
               }
               b.SetByteCount(pos,kTRUE);
            }
            continue;

         case kSTL:             // container with no virtual table (stl) and no comment
         case kSTL + kOffsetL:  // array of containers with no virtual table (stl) and no comment
            {
               TClass *cl                 = fComp[i].fClass;
               TMemberStreamer *pstreamer = fComp[i].fStreamer;
               UInt_t pos = b.WriteVersion(IsA(),kTRUE);
               if (pstreamer == 0) {
                  DOLOOP {
                     b.WriteFastArray((void*)(arr[k]+ioffset),cl,fLength[i],0);
                  }
               } else {
                  DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
               }
               b.SetByteCount(pos,kTRUE);

               continue;
            }
         
         case kObject:   // Class      derived from TObject
         case kOffsetL + kObject: 
         case kAny:   // Class  NOT derived from TObject
         case kAny     + kOffsetL: {
            TClass *cl                 = fComp[i].fClass;
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            DOLOOP
               {b.WriteFastArray((void*)(arr[k]+ioffset),cl,fLength[i],pstreamer);}
            continue;
         }
            
            // Base Class
         case kBase:    
            if (!(arrayMode&1)) {
               TMemberStreamer *pstreamer = fComp[i].fStreamer;
               if(pstreamer) {
                  // See kStreamer case (similar code)
                  UInt_t pos = b.WriteVersion(IsA(),kTRUE);
                  DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
                  b.SetByteCount(pos,kTRUE);
               } else {
                  DOLOOP { ((TStreamerBase*)aElement)->WriteBuffer(b,arr[k]);}
               }
            } else { 
               TClass *cl                 = fComp[i].fClass;
               cl->GetStreamerInfo()->WriteBufferAux(b,arr,-1,narr,ioffset,arrayMode);
            }
            continue;
            
         case kOffsetL + kTString:
         case kOffsetL + kTObject:
         case kOffsetL + kTNamed:
         case kStreamer: 
         {
            TMemberStreamer *pstreamer = fComp[i].fStreamer;

            UInt_t pos = b.WriteVersion(IsA(),kTRUE);
            if (pstreamer == 0) {
               printf("ERROR, Streamer is null\n");
               aElement->ls();continue;
            } else {
               DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
            }
            b.SetByteCount(pos,kTRUE);
         }
         continue;
         
         
         case kStreamLoop:{
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            UInt_t pos = b.WriteVersion(IsA(),kTRUE);
            if (pstreamer == 0) {
               printf("ERROR, Streamer is null\n");
               aElement->ls(); continue;
            }
            Int_t imethod = fMethod[i]+eoffset;
            DOLOOP {
               Int_t *counter = (Int_t*)(arr[k]+imethod);
               (*pstreamer)(b,arr[k]+ioffset,*counter);
            }
            b.SetByteCount(pos,kTRUE);
         }
         continue;

         default:
            Error("WriteBuffer","The element %s::%s type %d (%s) is not supported yet\n",GetName(),aElement->GetFullName(),fType[i],aElement->GetTypeName());
            continue;
      }
   }
   return 0;
}













//______________________________________________________________________________
Int_t TStreamerInfo::WriteBufferSTL(TBuffer &b, TVirtualCollectionProxy *cont, Int_t nc, Int_t first, Int_t eoffset)
{

   if (!nc) return 0;
   Assert((unsigned int)nc==cont->Size());
   char **arr = (char**) cont->GetPtrArray();

   int ret = WriteBufferAux(b, arr,first,nc,eoffset,1);
   return ret;
}

//______________________________________________________________________________
Int_t TStreamerInfo::WriteBuffer(TBuffer &b, char *ipointer, Int_t first)
{  
   return WriteBufferAux(b,&ipointer,first,1,0,0);
}

//______________________________________________________________________________
Int_t TStreamerInfo::WriteBufferClones(TBuffer &b, TClonesArray *clones, 
                                       Int_t nc, Int_t first, Int_t eoffset)
{
   char **arr = reinterpret_cast<char**>(clones->GetObjectRef(0)); // (char **)&((*clones)[0]);
   return WriteBufferAux(b,arr,first,nc,eoffset,1);
}


