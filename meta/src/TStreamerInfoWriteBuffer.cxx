#include "TBuffer.h"
#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TProcessID.h"
#include "TStreamer.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TVirtualCollectionProxy.h"
#include "TRefTable.h"

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
#ifdef R__BROKEN_FUNCTION_TEMPLATES
template <class T>
Int_t TStreamerInfo__WriteBufferAuxImp(TStreamerInfo *This,
                                       TBuffer &b, const T &arr, Int_t first,
                                       Int_t narr, Int_t eoffset, Int_t arrayMode,
                                       ULong_t *fMethod, ULong_t *fElem,Int_t *fLength,
                                       TClass *fClass, Int_t *fOffset, Int_t * /*fNewType*/,
                                       Int_t fNdata, Int_t *fType, TStreamerElement *& /*fgElement*/,
                                       TStreamerInfo::CompInfo *fComp)
{
#else
template <class T>
Int_t TStreamerInfo::WriteBufferAux(TBuffer &b, const T &arr, Int_t first,
				    Int_t narr, Int_t eoffset, Int_t arrayMode)
{
   TStreamerInfo *This = this;
#endif
   //  The object at pointer is serialized to the buffer b
   //  if (arrayMode & 1) ptr is a pointer to array of pointers to the objects
   //  otherwise it is a pointer to a pointer to a single object.
   //  This also means that T is of a type such that arr[i] is a pointer to an
   //  object.  Currently the only anticipated instantiation are for T==char**
   //  and T==TVirtualCollectionProxy

   b.IncrementLevel(This);

   //mark this class as being used in the current file
   This->TagFile((TFile *)b.GetParent());

   //============

   //loop on all active members
   Int_t last;
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;

   // In order to speed up the case where the object being written is
   // not in a collection (i.e. arrayMode is false), we actually
   // duplicate the elementary types using this typeOffset.
   static const int kHaveLoop = 1024;
   const Int_t typeOffset = arrayMode ? kHaveLoop : 0;

   for (Int_t i=first;i<last;i++) {

      b.SetStreamerElementNumber(i);
      TStreamerElement *aElement = (TStreamerElement*)fElem[i];

      const Int_t ioffset = eoffset+fOffset[i];

      if (gDebug > 1) {
         printf("WriteBuffer, class:%s, name=%s, fType[%d]=%d, %s, "
               "bufpos=%d, arr=%p, offset=%d\n",
                fClass->GetName(),aElement->GetName(),i,fType[i],aElement->ClassName(),
                b.Length(),arr[0],ioffset);
      }

      switch (fType[i]+typeOffset) {
         // In this switch we intentionally use 'continue' instead of
         // 'break' to avoid running the 2nd switch (see later in this
         // function).

         case TStreamerInfo::kBool:                WriteBasicType(Bool_t);    continue;
         case TStreamerInfo::kChar:                WriteBasicType(Char_t);    continue;
         case TStreamerInfo::kShort:               WriteBasicType(Short_t);   continue;
         case TStreamerInfo::kInt:                 WriteBasicType(Int_t);     continue;
         case TStreamerInfo::kLong:                WriteBasicType(Long_t);    continue;
         case TStreamerInfo::kLong64:              WriteBasicType(Long64_t);  continue;
         case TStreamerInfo::kFloat:               WriteBasicType(Float_t);   continue;
         case TStreamerInfo::kDouble:              WriteBasicType(Double_t);  continue;
         case TStreamerInfo::kUChar:               WriteBasicType(UChar_t);   continue;
         case TStreamerInfo::kUShort:              WriteBasicType(UShort_t);  continue;
         case TStreamerInfo::kUInt:                WriteBasicType(UInt_t);    continue;
         case TStreamerInfo::kULong:               WriteBasicType(ULong_t);   continue;
         case TStreamerInfo::kULong64:             WriteBasicType(ULong64_t); continue;
         case TStreamerInfo::kDouble32: {
            Double_t *x=(Double_t*)(arr[0]+ioffset);
            b.WriteDouble32(x,aElement);
            continue;
         }

         case TStreamerInfo::kBool    + kHaveLoop: WriteBasicTypeLoop(Bool_t);    continue;
         case TStreamerInfo::kChar    + kHaveLoop: WriteBasicTypeLoop(Char_t);    continue;
         case TStreamerInfo::kShort   + kHaveLoop: WriteBasicTypeLoop(Short_t);   continue;
         case TStreamerInfo::kInt     + kHaveLoop: WriteBasicTypeLoop(Int_t);     continue;
         case TStreamerInfo::kLong    + kHaveLoop: WriteBasicTypeLoop(Long_t);    continue;
         case TStreamerInfo::kLong64  + kHaveLoop: WriteBasicTypeLoop(Long64_t);  continue;
         case TStreamerInfo::kFloat   + kHaveLoop: WriteBasicTypeLoop(Float_t);   continue;
         case TStreamerInfo::kDouble  + kHaveLoop: WriteBasicTypeLoop(Double_t);  continue;
         case TStreamerInfo::kUChar   + kHaveLoop: WriteBasicTypeLoop(UChar_t);   continue;
         case TStreamerInfo::kUShort  + kHaveLoop: WriteBasicTypeLoop(UShort_t);  continue;
         case TStreamerInfo::kUInt    + kHaveLoop: WriteBasicTypeLoop(UInt_t);    continue;
         case TStreamerInfo::kULong   + kHaveLoop: WriteBasicTypeLoop(ULong_t);   continue;
         case TStreamerInfo::kULong64 + kHaveLoop: WriteBasicTypeLoop(ULong64_t); continue;
         case TStreamerInfo::kDouble32+ kHaveLoop: {
            for(int k=0; k<narr; ++k) {
               Double_t *x=(Double_t*)(arr[k]+ioffset);
               b.WriteDouble32(x,aElement);
            }
            continue;
         }

         // write array of basic types  array[8]
         case TStreamerInfo::kOffsetL + TStreamerInfo::kBool:   WriteBasicArray(Bool_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:   WriteBasicArray(Char_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:  WriteBasicArray(Short_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:    WriteBasicArray(Int_t);     continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:   WriteBasicArray(Long_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64: WriteBasicArray(Long64_t);  continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:  WriteBasicArray(Float_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble: WriteBasicArray(Double_t);  continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:  WriteBasicArray(UChar_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort: WriteBasicArray(UShort_t);  continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:   WriteBasicArray(UInt_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:  WriteBasicArray(ULong_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64:WriteBasicArray(ULong64_t); continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32: {
            b.WriteFastArrayDouble32((Double_t*)(arr[0]+ioffset),fLength[i],aElement);
            continue;
         }

         case TStreamerInfo::kOffsetL + TStreamerInfo::kBool    + kHaveLoop: WriteBasicArrayLoop(Bool_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kChar    + kHaveLoop: WriteBasicArrayLoop(Char_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kShort   + kHaveLoop: WriteBasicArrayLoop(Short_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kInt     + kHaveLoop: WriteBasicArrayLoop(Int_t);     continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong    + kHaveLoop: WriteBasicArrayLoop(Long_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64  + kHaveLoop: WriteBasicArrayLoop(Long64_t);  continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat   + kHaveLoop: WriteBasicArrayLoop(Float_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble  + kHaveLoop: WriteBasicArrayLoop(Double_t);  continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar   + kHaveLoop: WriteBasicArrayLoop(UChar_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort  + kHaveLoop: WriteBasicArrayLoop(UShort_t);  continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt    + kHaveLoop: WriteBasicArrayLoop(UInt_t);    continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong   + kHaveLoop: WriteBasicArrayLoop(ULong_t);   continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64 + kHaveLoop: WriteBasicArrayLoop(ULong64_t); continue;
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32+ kHaveLoop: {
            for(int k=0; k<narr; ++k) {
               b.WriteFastArrayDouble32((Double_t*)(arr[k]+ioffset),fLength[i],aElement);
            }
            continue;
         }

         // write pointer to an array of basic types  array[n]
         case TStreamerInfo::kOffsetP + TStreamerInfo::kBool:   WriteBasicPointer(Bool_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:   WriteBasicPointer(Char_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:  WriteBasicPointer(Short_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:    WriteBasicPointer(Int_t);     continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:   WriteBasicPointer(Long_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64: WriteBasicPointer(Long64_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:  WriteBasicPointer(Float_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble: WriteBasicPointer(Double_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:  WriteBasicPointer(UChar_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort: WriteBasicPointer(UShort_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:   WriteBasicPointer(UInt_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:  WriteBasicPointer(ULong_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64:WriteBasicPointer(ULong64_t); continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32: {
            int imethod = fMethod[i]+eoffset;
            Int_t *l = (Int_t*)(arr[0]+imethod);
            Double_t **f = (Double_t**)(arr[0]+ioffset);
            Double_t *af = *f;
            if (af && *l)  b << Char_t(1);
            else          {b << Char_t(0); continue;}
            int j;
            for(j=0;j<fLength[i];j++) {
               b.WriteFastArrayDouble32(f[j],*l,aElement);
            }
            continue;
         }

         case TStreamerInfo::kOffsetP + TStreamerInfo::kBool    + kHaveLoop: WriteBasicPointerLoop(Bool_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kChar    + kHaveLoop: WriteBasicPointerLoop(Char_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kShort   + kHaveLoop: WriteBasicPointerLoop(Short_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kInt     + kHaveLoop: WriteBasicPointerLoop(Int_t);     continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong    + kHaveLoop: WriteBasicPointerLoop(Long_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64  + kHaveLoop: WriteBasicPointerLoop(Long64_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat   + kHaveLoop: WriteBasicPointerLoop(Float_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble  + kHaveLoop: WriteBasicPointerLoop(Double_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar   + kHaveLoop: WriteBasicPointerLoop(UChar_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort  + kHaveLoop: WriteBasicPointerLoop(UShort_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt    + kHaveLoop: WriteBasicPointerLoop(UInt_t);    continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong   + kHaveLoop: WriteBasicPointerLoop(ULong_t);   continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64 + kHaveLoop: WriteBasicPointerLoop(ULong64_t); continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32+ kHaveLoop: {
            int imethod = fMethod[i]+eoffset;
            for(int k=0; k<narr; ++k) {
               Int_t *l = (Int_t*)(arr[k]+imethod);
               Double_t **f = (Double_t**)(arr[k]+ioffset);
               Double_t *af = *f;
               if (af && *l)  b << Char_t(1);
               else          {b << Char_t(0); continue;}
               int j;
               for(j=0;j<fLength[i];j++) {
                  b.WriteFastArrayDouble32(f[j],*l,aElement);
               }
            }
            continue;
         }

         case TStreamerInfo::kCounter: {
            Int_t *x=(Int_t*)(arr[0]+ioffset);
            b << *x;
            if (i == last-1) {
               b.DecrementLevel(This);
               return x[0]; // info used by TBranchElement::FillLeaves
            }
            continue;
         }

         case TStreamerInfo::kCounter + kHaveLoop : {
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
         case TStreamerInfo::kCharStar: { DOLOOP {
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
         case TStreamerInfo::kBits: { DOLOOP {
            UInt_t *x=(UInt_t*)(arr[k]+ioffset); b << *x;
            if ((*x & kIsReferenced) != 0) {
               TObject *obj = (TObject*)(arr[k]+eoffset);
               TRefTable *table = TRefTable::GetRefTable();
               if(table) table->Add(obj->GetUniqueID() & 0xffffff);
               TProcessID *pid = TProcessID::GetProcessWithUID(obj->GetUniqueID(),obj);
               UShort_t pidf = TProcessID::WriteProcessID(pid,(TFile *)b.GetParent());
               b << pidf;
            }
         }
         continue; }

         // Special case for TString, TObject, TNamed
         case TStreamerInfo::kTString: { DOLOOP{ ((TString*)(arr[k]+ioffset))->Streamer(b);         }; continue; }
         case TStreamerInfo::kTObject: { DOLOOP{ ((TObject*)(arr[k]+ioffset))->TObject::Streamer(b);}; continue; }
         case TStreamerInfo::kTNamed:  { DOLOOP{ ((TNamed *)(arr[k]+ioffset))->TNamed::Streamer(b); }; continue; }

         case TStreamerInfo::kAnyp:     // Class*   Class not derived from TObject and with comment field //->
         case TStreamerInfo::kAnyp    + TStreamerInfo::kOffsetL:

         case TStreamerInfo::kObjectp:  // Class *  Class     derived from TObject and with comment field //->
         case TStreamerInfo::kObjectp + TStreamerInfo::kOffsetL:

            isPreAlloc = kTRUE;

         case TStreamerInfo::kAnyP:         // Class*   Class not derived from TObject and no comment
         case TStreamerInfo::kAnyP    + TStreamerInfo::kOffsetL:

         case TStreamerInfo::kObjectP:  // Class*   Class derived from TObject
         case TStreamerInfo::kObjectP + TStreamerInfo::kOffsetL: {
            TClass *cl                 = fComp[i].fClass;
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            DOLOOP {
               Int_t res = b.WriteFastArray((void**)(arr[k]+ioffset),cl,fLength[i],isPreAlloc,pstreamer);
               if (res==2) {
                  Warning("WriteBuffer",
                          "The actual class of %s::%s is not available. Only the \"%s\" part will be written\n",
                          This->GetName(),aElement->GetName(),cl->GetName());
               }
             }
            continue;
         }

         case TStreamerInfo::kAnyPnoVT:     // Class*   Class not derived from TObject and no virtual table and no comment
         case TStreamerInfo::kAnyPnoVT    + TStreamerInfo::kOffsetL: {
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

//          case TStreamerInfo::kSTLvarp:           // Variable size array of STL containers.
//             {
//                TMemberStreamer *pstreamer = fComp[i].fStreamer;
//                TClass *cl                 = fComp[i].fClass;
//                UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
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


         case TStreamerInfo::kSTLp:                // Pointer to container with no virtual table (stl) and no comment
         case TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL:     // array of pointers to container with no virtual table (stl) and no comment
            {
               TClass *cl                 = fComp[i].fClass;
               TMemberStreamer *pstreamer = fComp[i].fStreamer;

               if (This->GetStreamMemberWise() && cl->CanSplit()) {
                  // Let's save the collection member-wise.

                  UInt_t pos = b.WriteVersionMemberWise(This->IsA(),kTRUE);
                  TVirtualCollectionProxy *proxy = cl->GetCollectionProxy();
                  TStreamerInfo *subinfo = proxy->GetValueClass()->GetStreamerInfo();
                  DOLOOP {
                     char **contp = (char**)(arr[k]+ioffset);
                     for(int j=0;j<fLength[i];++j) {
                        char *cont = contp[j];
                        TVirtualCollectionProxy::TPushPop helper( proxy, cont );
                        Int_t nobjects = proxy->Size();
                        b << nobjects;
                        subinfo->WriteBufferAux(b,*(proxy),-1,nobjects,0,1);
                     }
                   }
                   b.SetByteCount(pos,kTRUE);
                   continue;
               }
               UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
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

         case TStreamerInfo::kSTL:             // container with no virtual table (stl) and no comment
         case TStreamerInfo::kSTL + TStreamerInfo::kOffsetL:  // array of containers with no virtual table (stl) and no comment
            {
               TClass *cl                 = fComp[i].fClass;
               TMemberStreamer *pstreamer = fComp[i].fStreamer;
               if (This->GetStreamMemberWise() && cl->CanSplit()) {
                  // Let's save the collection in member-wise order.

                  UInt_t pos = b.WriteVersionMemberWise(This->IsA(),kTRUE);
                  TVirtualCollectionProxy *proxy = cl->GetCollectionProxy();
                  TStreamerInfo *subinfo = proxy->GetValueClass()->GetStreamerInfo();
                  DOLOOP {
                     char *obj = (char*)(arr[k]+ioffset);
                     Int_t n = fLength[i];
                     if (!n) n=1;
                     int size = cl->Size();

                     for(Int_t j=0; j<n; j++,obj+=size) {
                        TVirtualCollectionProxy::TPushPop helper( proxy, obj );
                        Int_t nobjects = proxy->Size();
                        b << nobjects;
                        subinfo->WriteBufferAux(b,*(proxy),-1,nobjects,0,1);
                     }
                  }
                  b.SetByteCount(pos,kTRUE);
                  continue;
               }
               UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
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

         case TStreamerInfo::kObject:   // Class      derived from TObject
         case TStreamerInfo::kAny:   // Class  NOT derived from TObject
         case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
         case TStreamerInfo::kAny     + TStreamerInfo::kOffsetL: {
            TClass *cl                 = fComp[i].fClass;
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            DOLOOP
               {b.WriteFastArray((void*)(arr[k]+ioffset),cl,fLength[i],pstreamer);}
            continue;
         }

         case TStreamerInfo::kOffsetL + TStreamerInfo::kTString:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kTObject:
         case TStreamerInfo::kOffsetL + TStreamerInfo::kTNamed:
         {
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            TClass *cl                 = fComp[i].fClass;

            UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
            DOLOOP {b.WriteFastArray((void*)(arr[k]+ioffset),cl,fLength[i],pstreamer);}
            b.SetByteCount(pos,kTRUE);
            continue;
         }

            // Base Class
         case TStreamerInfo::kBase:
            if (!(arrayMode&1)) {
               TMemberStreamer *pstreamer = fComp[i].fStreamer;
               if(pstreamer) {
                  // See kStreamer case (similar code)
                  UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
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

         case TStreamerInfo::kStreamer:
         {
            TMemberStreamer *pstreamer = fComp[i].fStreamer;

            UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
            if (pstreamer == 0) {
               printf("ERROR, Streamer is null\n");
               aElement->ls();continue;
            } else {
               DOLOOP{(*pstreamer)(b,arr[k]+ioffset,fLength[i]);}
            }
            b.SetByteCount(pos,kTRUE);
         }
         continue;


         case TStreamerInfo::kStreamLoop:{
            TMemberStreamer *pstreamer = fComp[i].fStreamer;
            UInt_t pos = b.WriteVersion(This->IsA(),kTRUE);
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
            Error("WriteBuffer","The element %s::%s type %d (%s) is not supported yet\n",This->GetName(),aElement->GetFullName(),fType[i],aElement->GetTypeName());
            continue;
      }
   }

   b.DecrementLevel(This);

   return 0;
}

#ifdef R__BROKEN_FUNCTION_TEMPLATES
// Support for non standard compilers

Int_t TStreamerInfo::WriteBufferAux(TBuffer &b, char ** const &arr, Int_t first,Int_t narr,Int_t eoffset,Int_t mode)
{
  return TStreamerInfo__WriteBufferAuxImp(this,b,arr,first,narr,eoffset,mode,
                                          fMethod,fElem,fLength,fClass,fOffset,fNewType,
                                          fNdata,fType,fgElement,fComp);
}

Int_t TStreamerInfo::WriteBufferAux(TBuffer &b, const TVirtualCollectionProxy &arr, Int_t first,Int_t narr,Int_t eoffset,Int_t mode)
{
  return TStreamerInfo__WriteBufferAuxImp(this,b,arr,first,narr,eoffset,mode,
                                          fMethod,fElem,fLength,fClass,fOffset,fNewType,
                                          fNdata,fType,fgElement,fComp);
}

#endif

//______________________________________________________________________________
Int_t TStreamerInfo::WriteBufferSTL(TBuffer &b, TVirtualCollectionProxy *cont, Int_t nc, Int_t first, Int_t eoffset)
{

   if (!nc) return 0;
   Assert((unsigned int)nc==cont->Size());

   int ret = WriteBufferAux(b, *cont,first,nc,eoffset,1);
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
   char **arr = reinterpret_cast<char**>(clones->GetObjectRef(0));
   return WriteBufferAux(b,arr,first,nc,eoffset,1);
}


