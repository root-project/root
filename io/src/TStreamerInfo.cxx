// @(#)root/meta:$Name:  $:$Id: TStreamerInfo.cxx,v 1.72 2001/05/20 13:50:54 brun Exp $
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TMethodCall.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TBaseClass.h"
#include "TBuffer.h"
#include "TFile.h"
#include "TArrayC.h"
#include "TArrayI.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TArrayS.h"
#include "TArrayL.h"
#include "TError.h"

Int_t   TStreamerInfo::fgCount = 0;
Bool_t  TStreamerInfo::fgOptimize = kTRUE;

const Int_t kRegrouped = TStreamerInfo::kOffsetL;

ClassImp(TStreamerInfo) 

//______________________________________________________________________________
TStreamerInfo::TStreamerInfo()
{
   // Default ctor.

   fNumber   = fgCount;
   fClass    = 0;
   fElements = 0;
   fType     = 0;
   fNewType  = 0;
   fOffset   = 0;
   fLength   = 0;
   fElem     = 0;
   fMethod   = 0;
   fCheckSum = 0;
   fNdata    = 0;
   fSize     = 0;
   fClassVersion = 0;
   fOptimized = kFALSE;
}

//______________________________________________________________________________
TStreamerInfo::TStreamerInfo(TClass *cl, const char *info)
        : TNamed(cl->GetName(),info)
{
   // Create a TStreamerInfo object.

   fgCount++;
   fNumber   = fgCount;
   fClass    = cl;
   fElements = new TObjArray();
   fType     = 0;
   fNewType  = 0;
   fOffset   = 0;
   fLength   = 0;
   fElem     = 0;
   fMethod   = 0;
   fCheckSum = 0;
   fNdata    = 0;
   fSize     = 0;
   fOptimized = kFALSE;
   fClassVersion = fClass->GetClassVersion();

   if (info) BuildUserInfo(info);

}

//______________________________________________________________________________
TStreamerInfo::~TStreamerInfo()
{
   // TStreamerInfo dtor.

   if (fNdata) {
      delete [] fType;
      delete [] fNewType;
      delete [] fOffset;
      delete [] fLength;
      delete [] fElem;
      delete [] fMethod;
   }
   if (!fElements) return;
   fElements->Delete();
   delete fElements;
}

//______________________________________________________________________________
void TStreamerInfo::Build()
{
   // Build the I/O data structure for the current class version
   // A list of TStreamerElement derived classes is built by scanning
   // one by one the list of data members of the analyzed class.

   TStreamerElement::Class()->IgnoreTObjectStreamer();

   fClass->BuildRealData();

   fCheckSum = fClass->GetCheckSum();
   Int_t i, ndim, offset;
   TClass *clm;
   TDataType *dt;
   TDataMember *dm;
   TBaseClass *base;
   TStreamerElement *element;
   TIter nextb(fClass->GetListOfBases());

   //iterate on list of base classes
   while((base = (TBaseClass*)nextb())) {
      clm = gROOT->GetClass(base->GetName());
      if (!clm) {
#ifdef MAYBEINFUTURE
         // support sor STL collections as base class cannot be implemented
         // It requires access to the base class offset in the class
         // try STL container or string
         Streamer_t streamer = 0;
         offset = GetDataMemberOffset(dm,streamer);
         if (offset == kMissing) continue;
         if (strcmp(base->GetName(),"string") == 0) {
            TStreamerSTLstring *stls = new TStreamerSTLstring(base->GetName(),base->GetTitle(),offset,base->GetName());
            fElements->Add(stls);
            stls->SetStreamer(streamer);
            continue;
         }
         if (strchr(base->GetName(),'<') && strchr(base->GetName(),'>')) {
            TStreamerSTL *stl = new TStreamerSTL(base->GetName(),base->GetTitle(),offset,base->GetName(),0);
            if (stl->GetSTLtype()) {
               fElements->Add(stl);
               stl->SetStreamer(streamer);
            }
            else delete stl;
            continue;
         }
         Error("Build","%s, unknow type: %s %s\n",GetName(),base->GetName(),base->GetTitle());
#endif
         continue;
      }
      clm->GetStreamerInfo();
      offset = fClass->GetBaseClassOffset(clm);
      element = new TStreamerBase(base->GetName(),base->GetTitle(),offset);
      if (clm == TObject::Class() && fClass->CanIgnoreTObjectStreamer()) {
         SetBit(TClass::kIgnoreTObjectStreamer);
         element->SetType(-1);
      }
      fElements->Add(element);
   }

   //iterate on list of data members
   TIter nextd(fClass->GetListOfDataMembers());

   while((dm=(TDataMember*)nextd())) {
      if (fClass->GetClassVersion() == 0) continue;
      if (!dm->IsPersistent()) continue;
      Streamer_t streamer = 0;
      offset = GetDataMemberOffset(dm,streamer);
      if (offset == kMissing) continue;

      //look for a data member with a counter in the comment string [n]
      TRealData *refcount = 0;
      TDataMember *dmref = 0;
      if (dm->IsaPointer()) {
         const char *title = (char*)dm->GetTitle();
         const char *lbracket = strchr(title,'[');
         const char *rbracket = strchr(title,']');
         if (lbracket && rbracket) {
            refcount = (TRealData*)fClass->GetListOfRealData()->FindObject(dm->GetArrayIndex());
            if (!refcount) {
               Error("Build","%s, discarding: %s %s, illegal %s\n",GetName(),dm->GetFullTypeName(),dm->GetName(),title);
               continue;
            }
            dmref = refcount->GetDataMember();
            TDataType *reftype = dmref->GetDataType();
            if (!reftype || reftype->GetType() != 3) {
               Error("Build","%s, discarding: %s %s, illegal [%s] (must be Int_t)\n",GetName(),dm->GetFullTypeName(),dm->GetName(),dm->GetArrayIndex());
               continue;
            }
            TStreamerBasicType *bt = TStreamerInfo::GetElementCounter(dm->GetArrayIndex(),dmref->GetClass(),dmref->GetClass()->GetClassVersion());
            if (!bt) {
               if (dmref->GetClass()->Property() & kIsAbstract) continue;
               Error("Build","%s, discarding: %s %s, illegal [%s] must be placed before \n",GetName(),dm->GetFullTypeName(),dm->GetName(),dm->GetArrayIndex());
               continue;
            }
         }
      }

      dt=dm->GetDataType();
      ndim = dm->GetArrayDim();

      if (dt) {  // found a basic type
         Int_t dtype = dt->GetType();
         Int_t dsize = dt->Size();
         if (dm->IsaPointer()) {
            if (refcount) {
               // data member is pointer to an array of basic types
               element = new TStreamerBasicPointer(dm->GetName(),dm->GetTitle(),offset,dtype,
                                                   dm->GetArrayIndex(),
                                                   dmref->GetClass()->GetName(),
                                                   dmref->GetClass()->GetClassVersion(),
                                                   dm->GetFullTypeName());
               for (i=0;i<ndim;i++) element->SetMaxIndex(i,dm->GetMaxIndex(i));
               element->SetArrayDim(ndim);
               element->SetSize(dsize);
               fElements->Add(element);
               continue;
            } else {
               if (fName == "TString" || fName == "TClass") continue;
               Error("Build","%s, discarding: %s %s, no [dimension]\n",GetName(),dm->GetFullTypeName(),dm->GetName());
               continue;
            }
         }
         // data member is a basic type
         element = new TStreamerBasicType(dm->GetName(),dm->GetTitle(),offset,dtype,dm->GetFullTypeName());
         Int_t ndim = dm->GetArrayDim();
         for (i=0;i<ndim;i++) element->SetMaxIndex(i,dm->GetMaxIndex(i));
         element->SetArrayDim(ndim);
         element->SetSize(dsize);
         fElements->Add(element);
         continue;

      } else {
         clm = gROOT->GetClass(dm->GetTypeName());
         if (!clm) {
            // try STL container or string
            static const char *full_string_name = "basic_string<char,char_traits<char>,allocator<char> >";
            if (strcmp(dm->GetTypeName(),"string") == 0
                ||strcmp(dm->GetTypeName(),full_string_name)==0 ) {
               TStreamerSTLstring *stls = new TStreamerSTLstring(dm->GetName(),dm->GetTitle(),offset,dm->GetFullTypeName());
               fElements->Add(stls);
               for (i=0;i<ndim;i++) stls->SetMaxIndex(i,dm->GetMaxIndex(i));
               stls->SetArrayDim(ndim);
               stls->SetStreamer(streamer);
               continue;
            }
            if (dm->IsSTLContainer()) {
               TStreamerSTL *stl = new TStreamerSTL(dm->GetName(),dm->GetTitle(),offset,dm->GetFullTypeName(),dm->IsaPointer());
               if (stl->GetSTLtype()) {
                  fElements->Add(stl);
                  for (i=0;i<ndim;i++) stl->SetMaxIndex(i,dm->GetMaxIndex(i));
                  stl->SetArrayDim(ndim);
                  stl->SetStreamer(streamer);
               }
               else delete stl;
               continue;
            }
            Error("Build","%s, unknow type: %s %s\n",GetName(),dm->GetFullTypeName(),dm->GetName());
            continue;
         }
         // a pointer to a class
         if (dm->IsaPointer()) {
            if(refcount) {
               element = new TStreamerLoop(dm->GetName(),
                                           dm->GetTitle(),offset,
                                           dm->GetArrayIndex(),
                                           dmref->GetClass()->GetName(),
                                           dmref->GetClass()->GetClassVersion(),
                                           dm->GetFullTypeName());
               fElements->Add(element);
               element->SetStreamer(streamer);
               continue;
            } else {
               if (clm->InheritsFrom(TObject::Class())) {
                  element = new TStreamerObjectPointer(dm->GetName(),dm->GetTitle(),offset,dm->GetFullTypeName());
                  fElements->Add(element);
                  for (i=0;i<ndim;i++) element->SetMaxIndex(i,dm->GetMaxIndex(i));
                  element->SetArrayDim(ndim);
                  element->SetStreamer(streamer);
                  continue;
               }
            }
         }
         // a class
         if (clm->InheritsFrom(TObject::Class())) {
            element = new TStreamerObject(dm->GetName(),dm->GetTitle(),offset,dm->GetFullTypeName());
            fElements->Add(element);
            for (i=0;i<ndim;i++) element->SetMaxIndex(i,dm->GetMaxIndex(i));
            element->SetArrayDim(ndim);
            element->SetStreamer(streamer);
            continue;
         } else if(clm == TString::Class() && !dm->IsaPointer()) {
            element = new TStreamerString(dm->GetName(),dm->GetTitle(),offset);
            fElements->Add(element);
            for (i=0;i<ndim;i++) element->SetMaxIndex(i,dm->GetMaxIndex(i));
            element->SetArrayDim(ndim);
            element->SetStreamer(streamer);
            continue;
         } else {
            element = new TStreamerObjectAny(dm->GetName(),dm->GetTitle(),offset,dm->GetFullTypeName());
            fElements->Add(element);
            for (i=0;i<ndim;i++) element->SetMaxIndex(i,dm->GetMaxIndex(i));
            element->SetArrayDim(ndim);
            element->SetStreamer(streamer);
            continue;
         }
      }
   }

   Compile();

}


//______________________________________________________________________________
void TStreamerInfo::BuildCheck()
{
   // check if the TStreamerInfo structure is already created
   // called by TFile::ReadStreamerInfo

   fClass = gROOT->GetClass(GetName());
   TObjArray *array;
   if (fClass) {
      array = fClass->GetStreamerInfos();
      if (fClassVersion == fClass->GetClassVersion()) {
         TStreamerInfo *info = (TStreamerInfo *)array->At(fClassVersion);
         if (info) {fNumber = info->GetNumber(); SetBit(kCanDelete); return;}
         if (fClass->GetListOfDataMembers() && (fCheckSum != fClass->GetCheckSum())) {
            printf("\nWARNING, class:%s StreamerInfo read from file:%s\n",GetName(),gDirectory->GetFile()->GetName());
            printf("        has the same version:%d than the active class\n",fClassVersion);
            printf("        but a different checksum.\n");
            printf("        You should update the version to ClassDef(%s,%d).\n",GetName(),fClassVersion+1);
            printf("        Do not try to write objects with the current class definition,\n");
            printf("        the files will not be readable.\n\n");
            //array->RemoveAt(fClassVersion);
         } else {
            if (info) {printf("ERROR\n"); SetBit(kCanDelete); return;}
         }
      }
   } else {
      fClass = new TClass(GetName(),fClassVersion,0,0,-1,-1);
      array = fClass->GetStreamerInfos();
   }
   if (TestBit(TClass::kIgnoreTObjectStreamer)) fClass->IgnoreTObjectStreamer();
   if (fClassVersion < 0 || fClassVersion > 65000) {
      printf("ERROR reading TStreamerInfo: %s fClassVersion=%d\n",GetName(),fClassVersion);
      SetBit(kCanDelete);
      fNumber = -1;
      return;
   }
   array->AddAtAndExpand(this,fClassVersion);
   fgCount++;
   fNumber = fgCount;

   //add to the global list of StreamerInfo
   TObjArray *infos = (TObjArray*)gROOT->GetListOfStreamerInfo();
   infos->AddAtAndExpand(this,fNumber);
}

//______________________________________________________________________________
void TStreamerInfo::BuildFake()
{
   // Create a Fake TStreamerInfo object.
   char duName[100];
   Assert(gFile);
   Int_t fv = gFile->GetVersion()%100000;
   Assert(fv < 30000);
   fClassVersion = -1;
   fCheckSum = 2001;
   TObjArray *elements = GetElements();
   if (!elements) return;
   Int_t ndata = elements->GetEntries();
   if (ndata == 0) return;
   TStreamerElement *element;
   Int_t i;
   for (i=0;i<ndata;i++) {
      element = (TStreamerElement*)elements->At(i);
      if (!element) break;
      int ty = element->GetType();
      if (ty < kChar || ty >kULong+kOffsetL)    continue;
      if (ty == kLong )                         element->SetType(kInt          );
      if (ty == kULong)                         element->SetType(kUInt         );
      if (ty == kLong +kOffsetL)                element->SetType(kInt +kOffsetL);
      if (ty == kULong+kOffsetL)                element->SetType(kUInt+kOffsetL);
      if (ty <= kULong)                         continue;
      strcpy(duName,element->GetName());
      strcat(duName,"QWERTY");
      TStreamerBasicType *bt = new TStreamerBasicType(duName, "", 0, kInt,"Int_t");
      {for (int j=ndata-1;j>=i;j--) {elements->AddAtAndExpand(elements->At(j),j+1);}}
      elements->AddAt(bt,i);
      ndata++;
      i++;
   }
   BuildOld();
}


//______________________________________________________________________________
void TStreamerInfo::BuildOld()
{
   // rebuild the TStreamerInfo structure

   if (gDebug > 0) printf("\n====>Rebuilding TStreamerInfo for class:%s, version:%d\n",GetName(),fClassVersion);
   TIter next(fElements);
   TStreamerElement *element;
   Int_t offset = 0;
   Streamer_t streamer = 0;
   while ((element = (TStreamerElement*)next())) {
      element->SetNewType(element->GetType());
      if (element->IsA() == TStreamerBase::Class()) {
         TStreamerBase *base = (TStreamerBase*)element;
         TClass *baseclass = base->GetClassPointer();
         baseclass->BuildRealData();
         Int_t version = base->GetBaseVersion();
         TStreamerInfo *infobase = baseclass->GetStreamerInfo(version);
         //if (infobase->GetNdata() == 0) infobase->BuildOld();
         if (infobase->GetTypes() == 0) infobase->BuildOld();
         element->Init();
         Int_t baseOffset = fClass->GetBaseClassOffset(baseclass);
         if (baseOffset < 0) baseOffset = 0;
         element->SetOffset(baseOffset);
         offset += baseclass->Size();
         continue;
      }
      //in principle, we should look rather into TRealData to support the
      //case where a member has been moved to a base class
      TDataMember *dm = (TDataMember*)fClass->GetListOfDataMembers()->FindObject(element->GetName());
      // may be a fake class
      if (!dm && fClass->GetDeclFileLine() < 0) {
         streamer = 0;
         element->SetOffset(offset);
         element->Init(fClass);
         element->SetStreamer(streamer);
         Int_t alength = element->GetArrayLength();
         if (alength == 0) alength = 1;
         Int_t asize = element->GetSize();
         //if (asize == 0) asize = 8;
         //offset += asize*alength;
         offset += asize;
         if (element->GetType() == kObject || element->GetType() == kTString) {
            //element->SetOffset(kMissing);
            //element->SetNewType(-1);
         }
      } else if (dm && dm->IsPersistent()) {
         TDataType *dt = dm->GetDataType();
         fClass->BuildRealData();
         streamer = 0;
         offset = GetDataMemberOffset(dm,streamer);
         element->SetOffset(offset);
         element->Init(fClass);
         element->SetStreamer(streamer);
         // in case, element is an array check array dimension(s)
         // check if data type has been changed
         if (strcmp(element->GetTypeName(),dm->GetFullTypeName())) {
            if (element->IsOldFormat(dm->GetFullTypeName())) continue;
            if (dt) {
               element->SetNewType(dt->GetType());
               printf("element: %s %s has new type: %s\n",element->GetTypeName(),element->GetName(),dm->GetFullTypeName());
            } else {
               element->SetNewType(-2);
               printf("Cannot convert %s from type:%s to type:%s, skip element\n",
                  element->GetName(),element->GetTypeName(),dm->GetFullTypeName());
            }
         }
      } else {
         element->SetOffset(kMissing);
         element->SetNewType(-1);
      }
   }

   Compile();
}


//______________________________________________________________________________
void TStreamerInfo::BuildUserInfo(const char *info)
{
   // Build the I/O data structure for the current class version

#ifdef TOBEIMPLEMENTED

   Int_t nch = strlen(title);
   char *info = new char[nch+1];
   strcpy(info,title);
   char *pos = info;
   TDataType *dt;
   TDataMember *dm;
   TRealData *rdm, *rdm1;
   TIter nextrdm(rmembers);

   // search tokens separated by a semicolon
   //tokens can be of the following types
   // -baseclass
   // -class     membername
   // -basictype membername
   // -classpointer     membername
   // -basictypepointer membername
   while(1) {
      Bool_t isPointer = kFALSE;
      while (*pos == ' ') pos++;
      if (*pos == 0) break;
      char *colon = strchr(pos,';');
      char *col = colon;
      if (colon) {
         *col = 0; col--;
         while (*col == ' ') {*col = 0; col--;}
         if (pos > col) break;
         char *star = strchr(pos,'*');
         if (star) {*star = ' '; isPointer = kTRUE;}
         char *blank;
         while(1) {
            blank = strstr(pos,"  ");
            if (blank == 0) break;
            strcpy(blank,blank+1);
         }
         blank = strrchr(pos,' '); //start in reverse order (case like unsigned short xx)
         if (blank) {
            *blank = 0;
            //check that this is a known data member
            dm  = (TDataMember*)members->FindObject(blank+1);
            rdm1 = 0;
            nextrdm.Reset();
            while((rdm = (TRealData*)nextrdm())) {
               if (rdm->GetDataMember() != dm) continue;
               rdm1 = rdm;
               break;
            }
            if (!dm || !rdm1) {
                printf("Unknown data member:%s %s in class:%s\n",pos,blank+1,fClass->GetName());
                pos = colon+1;
                continue;
            }
            // Is type a class name?
            TClass *clt = gROOT->GetClass(pos);
            if (clt) {
               //checks that the class matches with the data member declaration
               if (strcmp(pos,dm->GetTypeName()) == 0) {
                  newtype[fNdata] = 20;
                  fNdata++;
                  printf("class: %s member=%s\n",pos,blank+1);
               } else {
                  printf("Mismatch between class:%s %s and data member:%s %s in class:%s\n",
                       pos,blank+1,dm->GetTypeName(),blank+1,fClass->GetName());
               }
            // Is type a basic type?
            } else {
              dt = (TDataType*)gROOT->GetListOfTypes()->FindObject(pos);
              if (dt) {
                 Int_t dtype = dt->GetType();
                 //check that this is a valid data member and that the
                 //declared type matches with the data member type
                 if (dm->GetDataType()->GetType() == dtype) {
                    //store type and offset
                    newtype[fNdata]   = dtype;
                    newoffset[fNdata] = rdm->GetThisOffset();
                    fNdata++;
                    printf("type=%s, basic type=%s dtype=%d, member=%s\n",pos,dt->GetFullTypeName(),dtype,blank+1);
                 } else {
                    printf("Mismatch between type:%s %s and data member:%s %s in class:%s\n",
                       pos,blank+1,dm->GetTypeName(),blank+1,fClass->GetName());
                 }

              } else {
                 printf("found unknown type:%s, member=%s\n",pos,blank+1);
              }
           }
         } else {
            // very likely a base class
            TClass *base = gROOT->GetClass(pos);
            if (base && fClass->GetBaseClass(pos)) {
               printf("base class:%s\n",pos);
               //get pointer to method baseclass::Streamer
               TMethodCall *methodcall = new TMethodCall(base,"Streamer","");
               newtype[fNdata]   = 0;
               newmethod[fNdata] = (ULong_t)methodcall;
               fNdata++;
            }
         }
         pos = colon+1;
      }
   }
   fType     = new Int_t[fNdata+1];
   fOffset   = new Int_t[fNdata+1];
   fMethod   = new ULong_t[fNdata+1];
   for (Int_t i=0;i<fNdata;i++) {
      fType[i]   = newtype[i];
      fOffset[i] = newoffset[i];
      fMethod[i] = newmethod[i];
   }
   delete [] info;
   delete [] newtype;
   delete [] newoffset;
   delete [] newmethod;
#endif
}

//______________________________________________________________________________
Bool_t TStreamerInfo::CanOptimize()
{
// static function returning true if optimization can be on
   return fgOptimize;
}

//______________________________________________________________________________
void TStreamerInfo::Compile()
{
// loop on the TStreamerElement list
// regroup members with same type
// Store predigested information into local arrays. This saves a huge amount
// of time compared to an explicit iteration on all elements.

   TObjArray *infos = (TObjArray*)gROOT->GetListOfStreamerInfo();
   if (fNumber >= infos->GetSize()) {
      infos->AddAtAndExpand(this,fNumber);
   } else {
      if (!infos->At(fNumber)) {
         infos->AddAt(this,fNumber);
      }
   }

   if (fNdata) {
      delete [] fType;
      delete [] fNewType;
      delete [] fOffset;
      delete [] fLength;
      delete [] fElem;
      delete [] fMethod;
   }
   fOptimized = kFALSE;
   fNdata = 0;
   Int_t ndata = fElements->GetEntries();
   fType   = new Int_t[ndata+1];
   if (ndata == 0) return;
   fNewType= new Int_t[ndata];
   fOffset = new Int_t[ndata];
   fLength = new Int_t[ndata];
   fElem   = new ULong_t[ndata];
   fMethod = new ULong_t[ndata];
   TStreamerElement *element;
   Int_t keep = -1;
   Int_t i;
   if (!fgOptimize) SetBit(kCannotOptimize);
   for (i=0;i<ndata;i++) {
      element = (TStreamerElement*)fElements->At(i);
      if (!element) break;
      if (element->GetType() < 0) continue;
      Int_t asize = element->GetSize();
      if (element->GetArrayLength()) asize /= element->GetArrayLength();
      fType[fNdata]   = element->GetType();
      fNewType[fNdata]= element->GetNewType();
      fOffset[fNdata] = element->GetOffset();
      fLength[fNdata] = element->GetArrayLength();
      fElem[fNdata]   = (ULong_t)element;
      fMethod[fNdata] = element->GetMethod();
      // try to group consecutive members of the same type
      if (!TestBit(kCannotOptimize) && keep>=0 && (element->GetType() < 10)
                  && (fType[fNdata] == fNewType[fNdata])
                  && (fMethod[keep] == 0)
                  && (element->GetType() > 0)
                  && (element->GetArrayDim() == 0)
                  && (element->GetType() == (fType[keep]%kRegrouped))
                  && ((element->GetOffset()-fOffset[keep]) == (fLength[keep])*asize)) {
         if (fLength[keep] == 0) fLength[keep]++;
         fLength[keep]++;
         fType[keep] = element->GetType() + kRegrouped;
         fOptimized = kTRUE;
      } else {
         if (fType[fNdata] != kCounter) {
            if (fNewType[fNdata] != fType[fNdata]) {
               if (fNewType[fNdata] > 0) fType[fNdata] += kConv;
               else                      fType[fNdata] += kSkip;
            }
         }
         keep = fNdata;
         if (fLength[keep] == 0) fLength[keep] = 1;
         fNdata++;
      }
   }

   ComputeSize();

   if (gDebug > 0) ls();
}


//______________________________________________________________________________
void TStreamerInfo::ComputeSize()
{
// Compute total size of all persistent elements of the class

   TIter next(fElements);
   TStreamerElement *element;
   fSize = 0;
   while ((element = (TStreamerElement*)next())) {
      fSize += element->GetSize();
   }
}

//______________________________________________________________________________
void TStreamerInfo::ForceWriteInfo()
{
   // will force this TStreamerInfo to the file and also
   // all the dependencies.
   // This function is called when streaming a class that contains
   // a null pointer. In this case, the TStreamerInfo for the class
   // with the null pointer must be written to the file and also all the
   // TStreamerInfo of all the classes referenced by the class.

   // flag this class
   //if (!gDirectory->GetFile()) return;
   //TArrayC *cindex = gDirectory->GetFile()->GetClassIndex();
   if (!gFile) return;
   TArrayC *cindex = gFile->GetClassIndex();
   if (cindex->fArray[fNumber]) return;
   cindex->fArray[fNumber] = 1;
   cindex->fArray[0] = 1;

   // flag all its dependencies
   TIter next(fElements);
   TStreamerElement *element;
   while ((element = (TStreamerElement*)next())) {
      TClass *cl = element->GetClassPointer();
      if (cl) {
         cl->BuildRealData();
         cl->GetStreamerInfo()->ForceWriteInfo();
      }
   }
}


//______________________________________________________________________________
Int_t TStreamerInfo::GenerateHeaderFile(const char *dirname)
{
   // Generate header file for the class described by this TStreamerInfo
   // the function is called by TFile::MakeProject for each class in the file

   TClass *cl = gROOT->GetClass(GetName());
   if (cl) {
      if (cl->GetClassInfo()) return 0; // skip known classes
   }
   if (gDebug) printf("generating code for class %s\n",GetName());

   //open the file
   Int_t nch = strlen(dirname) + strlen(GetName()) + 4;
   char *filename = new char[nch];
   sprintf(filename,"%s/%s.h",dirname,GetName());
   FILE *fp = fopen(filename,"w");
   if (!fp) {
      printf("Cannot open output file:%s\n",filename);
      delete [] filename;
      return 0;
   }

   // generate class header
   TDatime td;
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"//   This class has been generated by TFile::MakeProject\n");
   fprintf(fp,"//     (%s by ROOT version %s)\n",td.AsString(),gROOT->GetVersion());
   fprintf(fp,"//      from the StreamerInfo in file %s\n",gDirectory->GetFile()->GetName());
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"\n");
   fprintf(fp,"\n");
   fprintf(fp,"#ifndef %s_h\n",GetName());
   fprintf(fp,"#define %s_h\n",GetName());
   fprintf(fp,"\n");

   // compute longest typename and member name
   // in the same loop, generate list of include files
   Int_t ltype = 10;
   Int_t ldata = 10;
   Int_t i,lt,ld;

   char *line = new char[512];
   char name[128];
   char cdim[8];
   char *inclist = new char[1000];
   inclist[0] = 0;

   TIter next(fElements);
   TStreamerElement *element;
   Int_t ninc = 0;
   while ((element = (TStreamerElement*)next())) {
      //if (element->IsA() == TStreamerBase::Class()) continue;
      sprintf(name,element->GetName());
      for (i=0;i<element->GetArrayDim();i++) {
         sprintf(cdim,"[%d]",element->GetMaxIndex(i));
         strcat(name,cdim);
      }
      ld = strlen(name);
      lt = strlen(element->GetTypeName());
      if (ltype < lt) ltype = lt;
      if (ldata < ld) ldata = ld;
      //get include file name if any
      const char *include = element->GetInclude();
      if (strlen(include) == 0) continue;
      // do not generate the include if already done
      if (strstr(inclist,include)) continue;
      ninc++;
      strcat(inclist,include);
      if (strstr(include,"include/") || strstr(include,"include\\"))
           fprintf(fp,"#include \"%s\n",include+9);
      else fprintf(fp,"#include %s\n",include);
   }
   ltype += 2;
   ldata++; // to take into account the semi colon
   if (ninc == 0) fprintf(fp,"#include \"TNamed.h\"\n");

   // generate class statement with base classes
   fprintf(fp,"\nclass %s",GetName());
   next.Reset();
   Int_t nbase = 0;
   while ((element = (TStreamerElement*)next())) {
      if (element->IsA() != TStreamerBase::Class()) continue;
      nbase++;
      if (nbase == 1) fprintf(fp," : public %s",element->GetName());
      else            fprintf(fp," , public %s",element->GetName());
   }
   fprintf(fp," {\n");

   // generate data members
   fprintf(fp,"\npublic:\n");
   next.Reset();
   while ((element = (TStreamerElement*)next())) {
      for (i=0;i<512;i++) line[i] = ' ';
      if (element->IsA() == TStreamerBase::Class()) continue;
      sprintf(name,element->GetName());
      for (Int_t i=0;i<element->GetArrayDim();i++) {
         sprintf(cdim,"[%d]",element->GetMaxIndex(i));
         strcat(name,cdim);
      }
      strcat(name,";");
      ld = strlen(name);
      lt = strlen(element->GetTypeNameBasic());
      strncpy(line+3,element->GetTypeNameBasic(),lt);
      strncpy(line+3+ltype,name,ld);
      if (element->IsaPointer() && !strchr(line,'*')) line[2+ltype] = '*';
      sprintf(line+3+ltype+ldata,"   //%s",element->GetTitle());
      fprintf(fp,"%s\n",line);
   }

   // generate default functions, ClassDef and trailer
   fprintf(fp,"\n   %s() {;}\n",GetName());
   fprintf(fp,"   virtual ~%s() {;}\n\n",GetName());
   fprintf(fp,"   ClassDef(%s,%d) //\n",GetName(),fClassVersion);
   fprintf(fp,"};\n");
   fprintf(fp,"\n   ClassImp(%s)\n",GetName());
   fprintf(fp,"#endif\n");

   fclose(fp);
   delete [] filename;
   delete [] inclist;
   delete [] line;
   return 1;
}

//______________________________________________________________________________
Int_t TStreamerInfo::GetDataMemberOffset(TDataMember *dm, Streamer_t &streamer) const
{
   // Compute data member offset
   // return pointer to the Streamer function if one exists

//printf("GetDataMemberOffset, class:%s, dm=%s\n",fClass->GetName(),dm->GetName());
   TIter nextr(fClass->GetListOfRealData());
   char dmbracket[256];
   sprintf(dmbracket,"%s[",dm->GetName());
   Int_t offset = kMissing;
   if (fClass->GetDeclFileLine() < 0) offset = dm->GetOffset();
   TRealData *rdm;
   while ((rdm = (TRealData*)nextr())) {
      char *rdmc = (char*)rdm->GetName();
//printf("rdmc=%s\n",rdmc);
      if (dm->IsaPointer() && rdmc[0] == '*') rdmc++;
      if (strcmp(rdmc,dm->GetName()) == 0) {
         offset   = rdm->GetThisOffset();
         streamer = rdm->GetStreamer();
         break;
      }
      if (strcmp(rdm->GetName(),dm->GetName()) == 0) {
         if (rdm->IsObject()) {
            offset = rdm->GetThisOffset();
            streamer = rdm->GetStreamer();
            break;
         }
      }
      if (strstr(rdm->GetName(),dmbracket)) {
         offset   = rdm->GetThisOffset();
         streamer = rdm->GetStreamer();
         break;
      }
  }
  return offset;
}

//______________________________________________________________________________
TStreamerBasicType *TStreamerInfo::GetElementCounter(const char *countName, TClass *cl, Int_t version)
{
   // Get pointer to a TStreamerBasicType in TClass *cl
   //static function

   TObjArray *sinfos = cl->GetStreamerInfos();
   TStreamerInfo *info = (TStreamerInfo *)sinfos->At(version);
   if (!info) return 0;
   TStreamerElement *element = (TStreamerElement *)info->fElements->FindObject(countName);
   if (!element) return 0;
   if (element->IsA() == TStreamerBasicType::Class()) return (TStreamerBasicType*)element;
   return 0;
}


//______________________________________________________________________________
Int_t TStreamerInfo::GetSize() const
{
//  return total size of all persistent elements of the class

   return fSize;
}

//______________________________________________________________________________
TStreamerElement* TStreamerInfo::GetStreamerElement(const char* datamember, Int_t &offset) const
{
//  Return the StreamerElement of "datamember" inside this class of any of its
//  base class.  The offset information contained in the StreamerElement is related
//  to its immediate containing class, so we return in 'offset' the offset inside
//  the class of this streamerInfo.

   if (!fElements) return 0;
   TStreamerElement *element = (TStreamerElement*)fElements->FindObject(datamember);
   if (element) {
      offset = element->GetOffset();
      return element;
   }

   TStreamerElement *base_element;
   TBaseClass *base;
   TClass *base_cl;
   Int_t base_offset = 0;
   Int_t local_offset = 0;
   TIter nextb(fClass->GetListOfBases());
   //iterate on list of base classes
   while((base = (TBaseClass*)nextb())) {
      base_cl = gROOT->GetClass(base->GetName());
      base_element = (TStreamerElement*)fElements->FindObject(base->GetName());
      base_offset = base_element->GetOffset();

      element = base_cl->GetStreamerInfo()->GetStreamerElement(datamember,local_offset);
      if (element) {
        offset = base_offset + local_offset;
        return element;
      }
   }
   return 0;
}

//______________________________________________________________________________
Double_t TStreamerInfo::GetValue(char *pointer, Int_t i, Int_t j, Int_t len) const
{
//  return value of element i in object at pointer.
//  The function may be called in two ways:
//    -method1  len < 0
//           i is assumed to be the TStreamerElement number i in StreamerInfo
//    -method2  len >= 0
//           i is the type
//           address of variable is directly pointer.

   char *ladd;
   Int_t atype;
   if (len >= 0) {
      ladd  = pointer;
      atype = i;
   } else {
      ladd  = pointer + fOffset[i];
      atype = fType[i];
   }
   switch (atype) {
         // basic types
      case kChar:              {Char_t *val   = (Char_t*)ladd;   return Double_t(*val);}
      case kShort:             {Short_t *val  = (Short_t*)ladd;  return Double_t(*val);}
      case kInt:               {Int_t *val    = (Int_t*)ladd;    return Double_t(*val);}
      case kLong:              {Long_t *val   = (Long_t*)ladd;   return Double_t(*val);}
      case kFloat:             {Float_t *val  = (Float_t*)ladd;  return Double_t(*val);}
      case kDouble:            {Double_t *val = (Double_t*)ladd; return Double_t(*val);}
      case kUChar:             {UChar_t *val  = (UChar_t*)ladd;  return Double_t(*val);}
      case kUShort:            {UShort_t *val = (UShort_t*)ladd; return Double_t(*val);}
      case kUInt:              {UInt_t *val   = (UInt_t*)ladd;   return Double_t(*val);}
      case kULong:             {ULong_t *val  = (ULong_t*)ladd;  return Double_t(*val);}

         // array of basic types  array[8]
      case kOffsetL + kChar:   {Char_t *val   = (Char_t*)ladd;   return Double_t(val[j]);}
      case kOffsetL + kShort:  {Short_t *val  = (Short_t*)ladd;  return Double_t(val[j]);}
      case kOffsetL + kInt:    {Int_t *val    = (Int_t*)ladd;    return Double_t(val[j]);}
      case kOffsetL + kLong:   {Long_t *val   = (Long_t*)ladd;   return Double_t(val[j]);}
      case kOffsetL + kFloat:  {Float_t *val  = (Float_t*)ladd;  return Double_t(val[j]);}
      case kOffsetL + kDouble: {Double_t *val = (Double_t*)ladd; return Double_t(val[j]);}
      case kOffsetL + kUChar:  {UChar_t *val  = (UChar_t*)ladd;  return Double_t(val[j]);}
      case kOffsetL + kUShort: {UShort_t *val = (UShort_t*)ladd; return Double_t(val[j]);}
      case kOffsetL + kUInt:   {UInt_t *val   = (UInt_t*)ladd;   return Double_t(val[j]);}
      case kOffsetL + kULong:  {ULong_t *val  = (ULong_t*)ladd;  return Double_t(val[j]);}

         // pointer to an array of basic types  array[n]
      case kOffsetP + kChar:   {Char_t **val   = (Char_t**)ladd;   return Double_t((*val)[j]);}
      case kOffsetP + kShort:  {Short_t **val  = (Short_t**)ladd;  return Double_t((*val)[j]);}
      case kOffsetP + kInt:    {Int_t **val    = (Int_t**)ladd;    return Double_t((*val)[j]);}
      case kOffsetP + kLong:   {Long_t **val   = (Long_t**)ladd;   return Double_t((*val)[j]);}
      case kOffsetP + kFloat:  {Float_t **val  = (Float_t**)ladd;  return Double_t((*val)[j]);}
      case kOffsetP + kDouble: {Double_t **val = (Double_t**)ladd; return Double_t((*val)[j]);}
      case kOffsetP + kUChar:  {UChar_t **val  = (UChar_t**)ladd;  return Double_t((*val)[j]);}
      case kOffsetP + kUShort: {UShort_t **val = (UShort_t**)ladd; return Double_t((*val)[j]);}
      case kOffsetP + kUInt:   {UInt_t **val   = (UInt_t**)ladd;   return Double_t((*val)[j]);}
      case kOffsetP + kULong:  {ULong_t **val  = (ULong_t**)ladd;  return Double_t((*val)[j]);}
         // array counter //[n]
      case kCounter:           {Int_t *val    = (Int_t*)ladd;    return Double_t(*val);}
   }
   return 0;
}

//______________________________________________________________________________
Double_t TStreamerInfo::GetValueClones(TClonesArray *clones, Int_t i, Int_t j, int k, Int_t eoffset) const
{
//  return value of element i in object number j in a TClonesArray and eventually
// element k in a sub-array.

   Int_t nc = clones->GetEntriesFast();
   if (j >= nc) return 0;

   char *pointer = (char*)clones->UncheckedAt(j);
   char *ladd    = pointer + eoffset + fOffset[i];
   switch (fType[i]) {
         // basic types
      case kChar:              {Char_t *val   = (Char_t*)ladd;   return Double_t(*val);}
      case kShort:             {Short_t *val  = (Short_t*)ladd;  return Double_t(*val);}
      case kInt:               {Int_t *val    = (Int_t*)ladd;    return Double_t(*val);}
      case kLong:              {Long_t *val   = (Long_t*)ladd;   return Double_t(*val);}
      case kFloat:             {Float_t *val  = (Float_t*)ladd;  return Double_t(*val);}
      case kDouble:            {Double_t *val = (Double_t*)ladd; return Double_t(*val);}
      case kUChar:             {UChar_t *val  = (UChar_t*)ladd;  return Double_t(*val);}
      case kUShort:            {UShort_t *val = (UShort_t*)ladd; return Double_t(*val);}
      case kUInt:              {UInt_t *val   = (UInt_t*)ladd;   return Double_t(*val);}
      case kULong:             {ULong_t *val  = (ULong_t*)ladd;  return Double_t(*val);}

         // array of basic types  array[8]
      case kOffsetL + kChar:   {Char_t *val   = (Char_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kShort:  {Short_t *val  = (Short_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kInt:    {Int_t *val    = (Int_t*)ladd;    return Double_t(val[k]);}
      case kOffsetL + kLong:   {Long_t *val   = (Long_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kFloat:  {Float_t *val  = (Float_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kDouble: {Double_t *val = (Double_t*)ladd; return Double_t(val[k]);}
      case kOffsetL + kUChar:  {UChar_t *val  = (UChar_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kUShort: {UShort_t *val = (UShort_t*)ladd; return Double_t(val[k]);}
      case kOffsetL + kUInt:   {UInt_t *val   = (UInt_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kULong:  {ULong_t *val  = (ULong_t*)ladd;  return Double_t(val[k]);}

         // pointer to an array of basic types  array[n]
      case kOffsetP + kChar:   {Char_t **val   = (Char_t**)ladd;   return Double_t((*val)[k]);}
      case kOffsetP + kShort:  {Short_t **val  = (Short_t**)ladd;  return Double_t((*val)[k]);}
      case kOffsetP + kInt:    {Int_t **val    = (Int_t**)ladd;    return Double_t((*val)[k]);}
      case kOffsetP + kLong:   {Long_t **val   = (Long_t**)ladd;   return Double_t((*val)[k]);}
      case kOffsetP + kFloat:  {Float_t **val  = (Float_t**)ladd;  return Double_t((*val)[k]);}
      case kOffsetP + kDouble: {Double_t **val = (Double_t**)ladd; return Double_t((*val)[k]);}
      case kOffsetP + kUChar:  {UChar_t **val  = (UChar_t**)ladd;  return Double_t((*val)[k]);}
      case kOffsetP + kUShort: {UShort_t **val = (UShort_t**)ladd; return Double_t((*val)[k]);}
      case kOffsetP + kUInt:   {UInt_t **val   = (UInt_t**)ladd;   return Double_t((*val)[k]);}
      case kOffsetP + kULong:  {ULong_t **val  = (ULong_t**)ladd;  return Double_t((*val)[k]);}
         // array counter //[n] 
      case kCounter:           {Int_t *val    = (Int_t*)ladd;    return Double_t(*val);}
   }
   return 0;
}

//______________________________________________________________________________
void TStreamerInfo::ls(Option_t *option) const
{
//  List the TStreamerElement list and also the precomputed tables
  printf("\nStreamerInfo for class: %s, version=%d\n",GetName(),fClassVersion);

   if (fElements) fElements->ls(option);
   for (Int_t i=0;i<fNdata;i++) {
      TStreamerElement *element = (TStreamerElement*)fElem[i];
      printf("   i=%2d, %-15s type=%3d, offset=%3d, len=%d, method=%ld\n",i,element->GetName(),fType[i],fOffset[i],fLength[i],fMethod[i]);
   }
}

//______________________________________________________________________________
Int_t TStreamerInfo::New(const char *p)
{
//  fake constructor for this class.
//  A fake object is created at address p

   TIter next(fElements);
   TStreamerElement *element;
   while ((element = (TStreamerElement*)next())) {
      Int_t etype = element->GetType();
      if (element->GetOffset() == kMissing) continue;
      //cle->GetStreamerInfo(); //necessary in case "->" is not specified
      if (etype == kObjectp) {
         // if the option "->" is given in the data member comment field
         // it is assumed that the object exist before reading data in.
         // In this case an object must be created
         if (strstr(element->GetTitle(),"->") == element->GetTitle()) {
            char line[200];
            char pname[100];
            char clonesClass[40];
            // in case of a TClonesArray, the class name of the contained objects
            // must be specified
            sprintf(clonesClass,"%s"," ");
            if (element->GetClassPointer() == TClonesArray::Class()) {
               char *bracket1 = (char*)strchr(element->GetTitle(),'(');
               char *bracket2 = (char*)strchr(element->GetTitle(),')');
               if (bracket1 && bracket2) {
                  clonesClass[0] = '"';
                  strncat(clonesClass,bracket1+1,bracket2-bracket1-1);
                  strcat(clonesClass,"\"");
               }
            }
            // object is created via the interpreter
            sprintf(pname,"R__%s_%s",GetName(),element->GetName());
            sprintf(line,"%s* %s = (%s*)0x%lx; *%s = new %s(%s);",
               element->GetTypeName(),pname,element->GetTypeName(),
               (Long_t)((char*)p + element->GetOffset()),pname,
               element->GetClassPointer()->GetName(),clonesClass);
               gROOT->ProcessLine(line);
         }
      }
      if (etype == kObject || etype == kAny || etype == kBase ||
          etype == kTObject || etype == kTString || etype == kTNamed) {
         TClass *cle = element->GetClassPointer();
         if (!cle) continue;
         cle->New((char*)p + element->GetOffset());
      }
   }
   return 0;
}

//______________________________________________________________________________
void TStreamerInfo::Optimize(Bool_t opt)
{
//  This is a static function.
//  Set optimization option.
//  When this option is activated (default), consecutive data members
//  of the same type are merged into an array (faster).
//  Optimization must be off in TTree split mode.

   fgOptimize = opt;
}


//______________________________________________________________________________
void TStreamerInfo::PrintValue(const char *name, char *pointer, Int_t i, Int_t len) const
{
//  print value of element i in object at pointer
//  The function may be called in two ways:
//    -method1  len < 0
//           i is assumed to be the TStreamerElement number i in StreamerInfo
//    -method2  len >= 0
//           i is the type
//           address of variable is directly pointer.
//           len is the number of elements to be printed starting at pointer.

   printf(" %-15s = ",name);
   Int_t j;
   char *ladd;
   Int_t atype,aleng;
   if (len >= 0) {
      ladd  = pointer;
      atype = i;
      aleng = len;
   } else {
      if (i < 0) {printf("NULL\n"); return;}
      ladd  = pointer + fOffset[i];
      atype = fType[i];
      aleng = fLength[i];
   }
   switch (atype) {
         // basic types
      case kChar:              {Char_t *val   = (Char_t*)ladd;   printf("%d",*val);  break;}
      case kShort:             {Short_t *val  = (Short_t*)ladd;  printf("%d",*val);  break;}
      case kInt:               {Int_t *val    = (Int_t*)ladd;    printf("%d",*val);  break;}
      case kLong:              {Long_t *val   = (Long_t*)ladd;   printf("%ld",*val); break;}
      case kFloat:             {Float_t *val  = (Float_t*)ladd;  printf("%f",*val);  break;}
      case kDouble:            {Double_t *val = (Double_t*)ladd; printf("%g",*val);  break;}
      case kUChar:             {UChar_t *val  = (UChar_t*)ladd;  printf("%d",*val);  break;}
      case kUShort:            {UShort_t *val = (UShort_t*)ladd; printf("%d",*val);  break;}
      case kUInt:              {UInt_t *val   = (UInt_t*)ladd;   printf("%d",*val);  break;}
      case kULong:             {ULong_t *val  = (ULong_t*)ladd;  printf("%ld",*val); break;}

         // array of basic types  array[8]
      case kOffsetL + kChar:   {Char_t *val   = (Char_t*)ladd;   for(j=0;j<aleng;j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kShort:  {Short_t *val  = (Short_t*)ladd;  for(j=0;j<aleng;j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kInt:    {Int_t *val    = (Int_t*)ladd;    for(j=0;j<aleng;j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kLong:   {Long_t *val   = (Long_t*)ladd;   for(j=0;j<aleng;j++) printf("%ld ",val[j]); break;}
      case kOffsetL + kFloat:  {Float_t *val  = (Float_t*)ladd;  for(j=0;j<aleng;j++) printf("%f ",val[j]);  break;}
      case kOffsetL + kDouble: {Double_t *val = (Double_t*)ladd; for(j=0;j<aleng;j++) printf("%g ",val[j]);  break;}
      case kOffsetL + kUChar:  {UChar_t *val  = (UChar_t*)ladd;  for(j=0;j<aleng;j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kUShort: {UShort_t *val = (UShort_t*)ladd; for(j=0;j<aleng;j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kUInt:   {UInt_t *val   = (UInt_t*)ladd;   for(j=0;j<aleng;j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kULong:  {ULong_t *val  = (ULong_t*)ladd;  for(j=0;j<aleng;j++) printf("%ld ",val[j]); break;}

         // pointer to an array of basic types  array[n]
      case kOffsetP + kChar:   {Char_t **val   = (Char_t**)ladd;   Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kShort:  {Short_t **val  = (Short_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kInt:    {Int_t **val    = (Int_t**)ladd;    Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kLong:   {Long_t **val   = (Long_t**)ladd;   Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%ld ",(*val)[j]); break;}
      case kOffsetP + kFloat:  {Float_t **val  = (Float_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%f ",(*val)[j]);  break;}
      case kOffsetP + kDouble: {Double_t **val = (Double_t**)ladd; Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%g ",(*val)[j]);  break;}
      case kOffsetP + kUChar:  {UChar_t **val  = (UChar_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kUShort: {UShort_t **val = (UShort_t**)ladd; Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kUInt:   {UInt_t **val   = (UInt_t**)ladd;   Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kULong:  {ULong_t **val  = (ULong_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%ld ",(*val)[j]); break;}
         // array counter //[n]
      case kCounter:           {Int_t *val    = (Int_t*)ladd;    printf("%d",*val);  break;}
         // Class *  derived from TObject with comment field  //->
      case kObjectp: {
                      TObject **obj = (TObject**)(pointer+fOffset[i]);
                      TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                      printf("(%s*)%lx",el->GetClass()->GetName(),(Long_t)(*obj));
                      break;
                     }

         // Class*   derived from TObject
      case kObjectP: {
                      TObject **obj = (TObject**)(pointer+fOffset[i]);
                      TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                      printf("(%s*)%lx",el->GetClass()->GetName(),(Long_t)(*obj));
                      break;
                     }

         // Class    derived from TObject
      case kObject:  {
                      TObject *obj = (TObject*)(pointer+fOffset[i]);
                      printf("%s",obj->GetName());
                      break;
                     }

         // Special case for TString, TObject, TNamed
      case kTString: {
                      TString *st = (TString*)(pointer+fOffset[i]);
                      printf("%s",st->Data());
                      break;
                     }
      case kTObject: {
                      TObject *obj = (TObject*)(pointer+fOffset[i]);
                      printf("%s",obj->GetName());
                      break;
                     }
      case kTNamed:  {
                      TNamed *named = (TNamed*) (pointer+fOffset[i]);
                      printf("%s/%s",named->GetName(),named->GetTitle());
                      break;
                     }

         // Any Class not derived from TObject
      case kOffsetL + kObjectp:
      case kOffsetL + kObjectP:
      case kAny:     {
                      printf("printing kAny case (%d)",fType[i]);
                      TStreamerElement *element = (TStreamerElement*)fElem[i];
                      //TClass *clany = element->GetClassPointer();
                      //TStreamerInfo *infoany = clany->GetStreamerInfo();
                      //infoany->PrintValue();
                      Streamer_t pstreamer = element->GetStreamer();
                      if (pstreamer == 0) {
                         //printf("ERROR, Streamer is null\n");
                         //element->ls();
                         break;
                      }
                      //(*pstreamer)(b,pointer+fOffset[i],0);
                      break;
                     }
         // Base Class
      case kBase:    {
                      printf("printing kBase case (%d)",fType[i]);
                       //TStreamerBase *element = (TStreamerBase*)fElem[i];
                       //element->ReadBuffer(b,pointer);
                       break;
                     }

      case kStreamer: {
                      printf("printing kStreamer case (%d)",fType[i]);
                      TStreamerElement *element = (TStreamerElement*)fElem[i];
                      Streamer_t pstreamer = element->GetStreamer();
                      if (pstreamer == 0) {
                         //printf("ERROR, Streamer is null\n");
                         //element->ls();
                         break;
                      }
                      //UInt_t start,count;
                      //b.ReadVersion(&start, &count);
                      //(*pstreamer)(b,pointer+fOffset[i],0);
                      //b.CheckByteCount(start,count,IsA());
                      break;
                     }

      case kStreamLoop: {
                      printf("printing kStreamLoop case (%d)",fType[i]);
                      TStreamerElement *element = (TStreamerElement*)fElem[i];
                      Streamer_t pstreamer = element->GetStreamer();
                      if (pstreamer == 0) {
                         //printf("ERROR, Streamer is null\n");
                         //element->ls();
                         break;
                      }
                      //Int_t *counter = (Int_t*)(pointer+fMethod[i]);
                      //UInt_t start,count;
                      ///b.ReadVersion(&start, &count);
                      //(*pstreamer)(b,pointer+fOffset[i],*counter);
                      //b.CheckByteCount(start,count,IsA());
                      break;
                     }
   }
   printf("\n");
}


//______________________________________________________________________________
void TStreamerInfo::PrintValueClones(const char *name, TClonesArray *clones, Int_t i, Int_t eoffset) const
{
//  print value of element i in a TClonesArray

   printf(" %-15s = ",name);
   Int_t nc = clones->GetEntriesFast();
   const Int_t kMaxPrint = 10;
   if (nc > kMaxPrint) nc = kMaxPrint;

   Int_t offset = eoffset + fOffset[i];
   Int_t j;
   for (Int_t k=0;k<nc;k++) {
      char *pointer = (char*)clones->UncheckedAt(k);
      char *ladd = pointer+offset;
      switch (fType[i]) {
         // basic types
      case kChar:              {Char_t *val   = (Char_t*)ladd;   printf("%d",*val);  break;}
      case kShort:             {Short_t *val  = (Short_t*)ladd;  printf("%d",*val);  break;}
      case kInt:               {Int_t *val    = (Int_t*)ladd;    printf("%d",*val);  break;}
      case kLong:              {Long_t *val   = (Long_t*)ladd;   printf("%ld",*val); break;}
      case kFloat:             {Float_t *val  = (Float_t*)ladd;  printf("%f",*val);  break;}
      case kDouble:            {Double_t *val = (Double_t*)ladd; printf("%g",*val);  break;}
      case kUChar:             {UChar_t *val  = (UChar_t*)ladd;  printf("%d",*val);  break;}
      case kUShort:            {UShort_t *val = (UShort_t*)ladd; printf("%d",*val);  break;}
      case kUInt:              {UInt_t *val   = (UInt_t*)ladd;   printf("%d",*val);  break;}
      case kULong:             {ULong_t *val  = (ULong_t*)ladd;  printf("%ld",*val); break;}

         // array of basic types  array[8]
      case kOffsetL + kChar:   {Char_t *val   = (Char_t*)ladd;   for(j=0;j<fLength[i];j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kShort:  {Short_t *val  = (Short_t*)ladd;  for(j=0;j<fLength[i];j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kInt:    {Int_t *val    = (Int_t*)ladd;    for(j=0;j<fLength[i];j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kLong:   {Long_t *val   = (Long_t*)ladd;   for(j=0;j<fLength[i];j++) printf("%ld ",val[j]); break;}
      case kOffsetL + kFloat:  {Float_t *val  = (Float_t*)ladd;  for(j=0;j<fLength[i];j++) printf("%f ",val[j]);  break;}
      case kOffsetL + kDouble: {Double_t *val = (Double_t*)ladd; for(j=0;j<fLength[i];j++) printf("%g ",val[j]);  break;}
      case kOffsetL + kUChar:  {UChar_t *val  = (UChar_t*)ladd;  for(j=0;j<fLength[i];j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kUShort: {UShort_t *val = (UShort_t*)ladd; for(j=0;j<fLength[i];j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kUInt:   {UInt_t *val   = (UInt_t*)ladd;   for(j=0;j<fLength[i];j++) printf("%d ",val[j]);  break;}
      case kOffsetL + kULong:  {ULong_t *val  = (ULong_t*)ladd;  for(j=0;j<fLength[i];j++) printf("%ld ",val[j]); break;}

         // pointer to an array of basic types  array[n]
      case kOffsetP + kChar:   {Char_t **val   = (Char_t**)ladd;   Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kShort:  {Short_t **val  = (Short_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kInt:    {Int_t **val    = (Int_t**)ladd;    Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kLong:   {Long_t **val   = (Long_t**)ladd;   Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%ld ",(*val)[j]); break;}
      case kOffsetP + kFloat:  {Float_t **val  = (Float_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%f ",(*val)[j]);  break;}
      case kOffsetP + kDouble: {Double_t **val = (Double_t**)ladd; Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%g ",(*val)[j]);  break;}
      case kOffsetP + kUChar:  {UChar_t **val  = (UChar_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kUShort: {UShort_t **val = (UShort_t**)ladd; Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kUInt:   {UInt_t **val   = (UInt_t**)ladd;   Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%d ",(*val)[j]);  break;}
      case kOffsetP + kULong:  {ULong_t **val  = (ULong_t**)ladd;  Int_t *l = (Int_t*)(pointer+fMethod[i]); for(j=0;j<*l;j++) printf("%ld ",(*val)[j]); break;}
         // array counter //[n]
      case kCounter:           {Int_t *val    = (Int_t*)ladd;    printf("%d",*val);  break;}
         // Class *  derived from TObject with comment field  //->
      case kObjectp: {
                      TObject **obj = (TObject**)ladd;
                      TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                      printf("(%s*)%lx",el->GetClass()->GetName(),(Long_t)(*obj));
                      break;
                     }

         // Class*   derived from TObject
      case kObjectP: {
                      TObject **obj = (TObject**)ladd;
                      TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                      printf("(%s*)%lx",el->GetClass()->GetName(),(Long_t)(*obj));
                      break;
                     }

         // Class    derived from TObject
      case kObject:  {
                      TObject *obj = (TObject*)ladd;
                      printf("%s",obj->GetName());
                      break;
                     }

         // Special case for TString, TObject, TNamed
      case kTString: {
                      TString *st = (TString*)ladd;
                      printf("%s",st->Data());
                      break;
                     }
      case kTObject: {
                      TObject *obj = (TObject*)ladd;
                      printf("%s",obj->GetName());
                      break;
                     }
      case kTNamed:  {
                      TNamed *named = (TNamed*)ladd;
                      printf("%s/%s",named->GetName(),named->GetTitle());
                      break;
                     }

         // Any Class not derived from TObject
      case kOffsetL + kObjectp:
      case kOffsetL + kObjectP:
      case kAny:     {
                      TStreamerElement *element = (TStreamerElement*)fElem[i];
                      printf("%s",element->GetTypeName());
                      //TClass *clany = element->GetClassPointer();
                      //TStreamerInfo *infoany = clany->GetStreamerInfo();
                      //infoany->PrintValue();
                      Streamer_t pstreamer = element->GetStreamer();
                      if (pstreamer == 0) {
                         //printf("ERROR, Streamer is null\n");
                         //element->ls();
                         break;
                      }
                      //(*pstreamer)(b,pointer+fOffset[i],0);
                      break;
                     }
         // Base Class
      case kBase:    {
                      printf("printing kBase case");
                       //TStreamerBase *element = (TStreamerBase*)fElem[i];
                       //element->ReadBuffer(b,pointer);
                       break;
                     }

      case kStreamer: {
                      printf("printing kStreamer case");
                      TStreamerElement *element = (TStreamerElement*)fElem[i];
                      Streamer_t pstreamer = element->GetStreamer();
                      if (pstreamer == 0) {
                         //printf("ERROR, Streamer is null\n");
                         element->ls();
                         break;
                      }
                      //UInt_t start,count;
                      //b.ReadVersion(&start, &count);
                      //(*pstreamer)(b,pointer+fOffset[i],0);
                      //b.CheckByteCount(start,count,IsA());
                      break;
                     }

      case kStreamLoop: {
                      printf("printing kStreamLoop case");
                      TStreamerElement *element = (TStreamerElement*)fElem[i];
                      Streamer_t pstreamer = element->GetStreamer();
                      if (pstreamer == 0) {
                         printf("ERROR, Streamer is null\n");
                         element->ls();
                         break;
                      }
                      //Int_t *counter = (Int_t*)(pointer+fMethod[i]);
                      //UInt_t start,count;
                      ///b.ReadVersion(&start, &count);
                      //(*pstreamer)(b,pointer+fOffset[i],*counter);
                      //b.CheckByteCount(start,count,IsA());
                      break;
                     }
      }
      if (k < nc-1) printf(", ");
   }
   printf("\n");

}

//______________________________________________________________________________
Int_t TStreamerInfo::ReadBuffer(TBuffer &b, char *pointer, Int_t first)
{
//  Deserialize information from buffer b into object at pointer


//==========CPP macros
#define ReadBasicType(name) \
{ \
   name *x=(name*)(pointer+fOffset[i]); \
   b >> *x; \
   break; \
}

#define ReadBasicArray(name) \
{ \
   name *x=(name*)(pointer+fOffset[i]); \
   b.ReadFastArray(x,fLength[i]); \
   break; \
}

#define ReadBasicPointer(name) \
{ \
   Char_t isArray; \
   b >> isArray; \
   if (isArray == 0) break; \
   Int_t *l = (Int_t*)(pointer+fMethod[i]); \
   name **f = (name**)(pointer+fOffset[i]); \
   delete [] *f; \
   *f = 0; if (*l ==0) break; \
   *f = new name[*l]; \
   b.ReadFastArray(*f,*l); break; \
}

#define SkipBasicType(name) \
{ \
   name dummy; \
   b >> dummy; \
   break; \
}

#define SkipBasicArray(name) \
{ \
   name dummy; \
   for (Int_t j=0;j<fLength[i];j++) b >> dummy; \
   break; \
}

#define SkipBasicPointer(name) \
{ \
   Int_t *n = (Int_t*)(pointer+fMethod[i]); \
   Int_t l = b.Length(); \
   b.SetBufferOffset(l+1+(*n)*sizeof( name )); \
   break; \
}

#define ConvBasicType(name) \
{ \
   name  dummy;   b >> dummy; \
   switch(fNewType[i]) { \
      case kChar:   {Char_t   *x=(Char_t*)(pointer+fOffset[i]);   *x = (Char_t)dummy;   break;} \
      case kShort:  {Short_t  *x=(Short_t*)(pointer+fOffset[i]);  *x = (Short_t)dummy;  break;} \
      case kInt:    {Int_t    *x=(Int_t*)(pointer+fOffset[i]);    *x = (Int_t)dummy;    break;} \
      case kLong:   {Long_t   *x=(Long_t*)(pointer+fOffset[i]);   *x = (Long_t)dummy;   break;} \
      case kFloat:  {Float_t  *x=(Float_t*)(pointer+fOffset[i]);  *x = (Float_t)dummy;  break;} \
      case kDouble: {Double_t *x=(Double_t*)(pointer+fOffset[i]); *x = (Double_t)dummy; break;} \
      case kUChar:  {UChar_t  *x=(UChar_t*)(pointer+fOffset[i]);  *x = (UChar_t)dummy;  break;} \
      case kUShort: {UShort_t *x=(UShort_t*)(pointer+fOffset[i]); *x = (UShort_t)dummy; break;} \
      case kUInt:   {UInt_t   *x=(UInt_t*)(pointer+fOffset[i]);   *x = (UInt_t)dummy;   break;} \
      case kULong:  {ULong_t  *x=(ULong_t*)(pointer+fOffset[i]);  *x = (ULong_t)dummy;  break;} \
   } break; \
}

#define ConvBasicArray(name) \
{ \
   name dummy; \
   for (Int_t j=0;j<fLength[i];j++) { \
      b >> dummy; \
      switch(fNewType[i]) { \
         case kChar:   {Char_t   *x=(Char_t*)(pointer+fOffset[i]);   x[j] = (Char_t)dummy;   break;} \
         case kShort:  {Short_t  *x=(Short_t*)(pointer+fOffset[i]);  x[j] = (Short_t)dummy;  break;} \
         case kInt:    {Int_t    *x=(Int_t*)(pointer+fOffset[i]);    x[j] = (Int_t)dummy;    break;} \
         case kLong:   {Long_t   *x=(Long_t*)(pointer+fOffset[i]);   x[j] = (Long_t)dummy;   break;} \
         case kFloat:  {Float_t  *x=(Float_t*)(pointer+fOffset[i]);  x[j] = (Float_t)dummy;  break;} \
         case kDouble: {Double_t *x=(Double_t*)(pointer+fOffset[i]); x[j] = (Double_t)dummy; break;} \
         case kUChar:  {UChar_t  *x=(UChar_t*)(pointer+fOffset[i]);  x[j] = (UChar_t)dummy;  break;} \
         case kUShort: {UShort_t *x=(UShort_t*)(pointer+fOffset[i]); x[j] = (UShort_t)dummy; break;} \
         case kUInt:   {UInt_t   *x=(UInt_t*)(pointer+fOffset[i]);   x[j] = (UInt_t)dummy;   break;} \
         case kULong:  {ULong_t  *x=(ULong_t*)(pointer+fOffset[i]);  x[j] = (ULong_t)dummy;  break;} \
      } \
   } break; \
}

#define ConvBasicPointer(name) \
{ \
   Char_t isArray; \
   b >> isArray; \
   if (isArray == 0) break; \
   Int_t *l = (Int_t*)(pointer+fMethod[i]); name dummy; \
   switch(fNewType[i]) { \
      case kChar:   {Char_t   **f=(Char_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new Char_t[*l]; Char_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (Char_t)dummy;} break;} \
      case kShort:  {Short_t  **f=(Short_t**)(pointer+fOffset[i]); \
                    *f = 0; if (*l ==0) continue; \
                    *f = new Short_t[*l]; Short_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (Short_t)dummy;} break;} \
      case kInt:    {Int_t    **f=(Int_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new Int_t[*l]; Int_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (Int_t)dummy;} break;} \
      case kLong:   {Long_t   **f=(Long_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new Long_t[*l]; Long_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (Long_t)dummy;} break;} \
      case kFloat:  {Float_t  **f=(Float_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new Float_t[*l]; Float_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (Float_t)dummy;} break;} \
      case kDouble: {Double_t **f=(Double_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new Double_t[*l]; Double_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (Double_t)dummy;} break;} \
      case kUChar:  {UChar_t  **f=(UChar_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new UChar_t[*l]; UChar_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (UChar_t)dummy;} break;} \
      case kUShort: {UShort_t **f=(UShort_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new UShort_t[*l]; UShort_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (UShort_t)dummy;} break;} \
      case kUInt:   {UInt_t   **f=(UInt_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new UInt_t[*l]; UInt_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (UInt_t)dummy;} break;} \
      case kULong:  {ULong_t  **f=(ULong_t**)(pointer+fOffset[i]); delete [] *f; \
                    *f = 0; if (*l ==0) continue; \
                    *f = new ULong_t[*l]; ULong_t *af = *f; \
                    for (Int_t j=0;j<*l;j++) {b >> dummy; af[j] = (ULong_t)dummy;} break;} \
   } break; \
}

//============

   if (!fType) {
      fClass->BuildRealData(pointer);
      BuildOld();
   }

   //loop on all active members
   Int_t last;
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;
   for (Int_t i=first;i<last;i++) {
      TStreamerElement *element = (TStreamerElement*)fElem[i];
//#ifdef DEBUG
      if (gDebug > 1) {
         printf("ReadBuffer, class:%s, name=%s, fType[%d]=%d, %s, bufpos=%d, pointer=%lx, offset=%d\n",fClass->GetName(),element->GetName(),i,fType[i],element->ClassName(),b.Length(),(Long_t)pointer, fOffset[i]);
      }
//#endif
      switch (fType[i]) {
         // read basic types
         case kChar:              ReadBasicType(Char_t)
         case kShort:             ReadBasicType(Short_t)
         case kInt:               ReadBasicType(Int_t)
         case kLong:              ReadBasicType(Long_t)
         case kFloat:             ReadBasicType(Float_t)
         case kDouble:            ReadBasicType(Double_t)
         case kUChar:             ReadBasicType(UChar_t)
         case kUShort:            ReadBasicType(UShort_t)
         case kUInt:              ReadBasicType(UInt_t)
         case kULong:             ReadBasicType(ULong_t)

         // read array of basic types  array[8]
         case kOffsetL + kChar:   ReadBasicArray(Char_t)
         case kOffsetL + kShort:  ReadBasicArray(Short_t)
         case kOffsetL + kInt:    ReadBasicArray(Int_t)
         case kOffsetL + kLong:   ReadBasicArray(Long_t)
         case kOffsetL + kFloat:  ReadBasicArray(Float_t)
         case kOffsetL + kDouble: ReadBasicArray(Double_t)
         case kOffsetL + kUChar:  ReadBasicArray(UChar_t)
         case kOffsetL + kUShort: ReadBasicArray(UShort_t)
         case kOffsetL + kUInt:   ReadBasicArray(UInt_t)
         case kOffsetL + kULong:  ReadBasicArray(ULong_t)

         // read pointer to an array of basic types  array[n]
         case kOffsetP + kChar:   ReadBasicPointer(Char_t)
         case kOffsetP + kShort:  ReadBasicPointer(Short_t)
         case kOffsetP + kInt:    ReadBasicPointer(Int_t)
         case kOffsetP + kLong:   ReadBasicPointer(Long_t)
         case kOffsetP + kFloat:  ReadBasicPointer(Float_t)
         case kOffsetP + kDouble: ReadBasicPointer(Double_t)
         case kOffsetP + kUChar:  ReadBasicPointer(UChar_t)
         case kOffsetP + kUShort: ReadBasicPointer(UShort_t)
         case kOffsetP + kUInt:   ReadBasicPointer(UInt_t)
         case kOffsetP + kULong:  ReadBasicPointer(ULong_t)

         // Class *  derived from TObject with comment field  //->
         case kObjectp: {
                         TObject **obj = (TObject**)(pointer+fOffset[i]);
                         if (!(*obj)) {
                            TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                            *obj = (TObject*)el->GetClass()->New();
                         }
                         Bool_t old = gROOT->ReadingObject();
                         gROOT->SetReadingObject(kTRUE);
                         (*obj)->Streamer(b);
                         gROOT->SetReadingObject(old);
                         if ((*obj)->IsZombie()) {
                            delete (*obj);
                            *obj = 0;
                         }
                         break;
                        }

         // Class*   derived from TObject
         case kObjectP: { TObject **obj = (TObject**)(pointer+fOffset[i]);
                          b >> *obj;
                          break;
                        }

         // array counter //[n]
         case kCounter: { Int_t *x=(Int_t*)(pointer+fOffset[i]);
                          b >> *x;
                          break;
                        }

         // Class    derived from TObject
         case kObject:  {
                         TClass *cl = element->GetClassPointer();
                         if (cl->GetClassInfo()) {
                            ((TObject*)(pointer+fOffset[i]))->Streamer(b);
                            break;
                         } else {
                            cl->GetStreamerInfo()->ReadBuffer(b,pointer+fOffset[i],0);
                         }
                         break;
                        }

         // Special case for TString, TObject, TNamed
         case kTString: { ((TString*)(pointer+fOffset[i]))->Streamer(b); break;}
         case kTObject: { ((TObject*)(pointer+fOffset[i]))->TObject::Streamer(b); break;}
         case kTNamed:  { ((TNamed*) (pointer+fOffset[i]))->TNamed::Streamer(b); break;}

         // Any Class not derived from TObject
         case kOffsetL + kObjectp:
         case kOffsetL + kObjectP:
         case kAny:     {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            if (gDebug > 0) printf("WARNING, Streamer is null\n");
                            //Note that this does not work if the class has a custom Streamer
                            //with no bytecount
                            TClass *cle = element->GetClassPointer();
                            if (cle->InheritsFrom(TArray::Class())) {
                               //special case (frequent) with TArray classes
                               //The TArray Streamers not compatible with ReadBuffer
                               // (no byte count)
                               if (strchr(element->GetTypeName(),'*')) {
                                  if (cle == TArrayI::Class()) {TArrayI **ar = (TArrayI**)(pointer+fOffset[i]); b >> *ar; break;}
                                  if (cle == TArrayF::Class()) {TArrayF **ar = (TArrayF**)(pointer+fOffset[i]); b >> *ar; break;}
                                  if (cle == TArrayC::Class()) {TArrayC **ar = (TArrayC**)(pointer+fOffset[i]); b >> *ar; break;}
                                  if (cle == TArrayD::Class()) {TArrayD **ar = (TArrayD**)(pointer+fOffset[i]); b >> *ar; break;}
                                  if (cle == TArrayS::Class()) {TArrayS **ar = (TArrayS**)(pointer+fOffset[i]); b >> *ar; break;}
                                  if (cle == TArrayL::Class()) {TArrayL **ar = (TArrayL**)(pointer+fOffset[i]); b >> *ar; break;}
                               } else {
                                  if (cle == TArrayI::Class()) {TArrayI *ar = (TArrayI*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayF::Class()) {TArrayF *ar = (TArrayF*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayC::Class()) {TArrayC *ar = (TArrayC*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayD::Class()) {TArrayD *ar = (TArrayD*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayS::Class()) {TArrayS *ar = (TArrayS*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayL::Class()) {TArrayL *ar = (TArrayL*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                               }
                            }
                           cle->ReadBuffer(b,pointer+fOffset[i]);
                            break;
                         }
                         (*pstreamer)(b,pointer+fOffset[i],0);
                         break;
                        }
         // Base Class
         case kBase:    { TStreamerBase *element = (TStreamerBase*)fElem[i];
                          element->ReadBuffer(b,pointer);
                          break;
                        }

         case kStreamer: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            if (gDebug > 0) {
                               printf("ERROR, Streamer is null\n");
                               element->ls();
                            }
                            break;
                         }
                         UInt_t start,count;
                         if (fClassVersion != -1) b.ReadVersion(&start, &count);
                         (*pstreamer)(b,pointer+fOffset[i],0);
                         if (fClassVersion != -1) b.CheckByteCount(start,count,IsA());
                         break;
                        }

         case kStreamLoop: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            if (gDebug > 0) {
                               printf("ERROR, Streamer is null\n");
                               element->ls();
                            }
                            break;
                         }
                         Int_t *counter = (Int_t*)(pointer+fMethod[i]);
                         UInt_t start,count;
                         b.ReadVersion(&start, &count);
                         (*pstreamer)(b,pointer+fOffset[i],*counter);
                         b.CheckByteCount(start,count,IsA());
                         break;
                        }


         // skip basic types
         case kSkip + kChar:    SkipBasicType(Char_t)
         case kSkip + kShort:   SkipBasicType(Short_t)
         case kSkip + kInt:     SkipBasicType(Int_t)
         case kSkip + kLong:    SkipBasicType(Long_t)
         case kSkip + kFloat:   SkipBasicType(Float_t)
         case kSkip + kDouble:  SkipBasicType(Double_t)
         case kSkip + kUChar:   SkipBasicType(UChar_t)
         case kSkip + kUShort:  SkipBasicType(UShort_t)
         case kSkip + kUInt:    SkipBasicType(UInt_t)
         case kSkip + kULong:   SkipBasicType(ULong_t)

         // skip array of basic types  array[8]
         case kSkipL + kChar:   SkipBasicArray(Char_t)
         case kSkipL + kShort:  SkipBasicArray(Short_t)
         case kSkipL + kInt:    SkipBasicArray(Int_t)
         case kSkipL + kLong:   SkipBasicArray(Long_t)
         case kSkipL + kFloat:  SkipBasicArray(Float_t)
         case kSkipL + kDouble: SkipBasicArray(Double_t)
         case kSkipL + kUChar:  SkipBasicArray(UChar_t)
         case kSkipL + kUShort: SkipBasicArray(UShort_t)
         case kSkipL + kUInt:   SkipBasicArray(UInt_t)
         case kSkipL + kULong:  SkipBasicArray(ULong_t)

         // skip pointer to an array of basic types  array[n]
         case kSkipP + kChar:   SkipBasicPointer(Char_t)
         case kSkipP + kShort:  SkipBasicPointer(Short_t)
         case kSkipP + kInt:    SkipBasicPointer(Int_t)
         case kSkipP + kLong:   SkipBasicPointer(Long_t)
         case kSkipP + kFloat:  SkipBasicPointer(Float_t)
         case kSkipP + kDouble: SkipBasicPointer(Double_t)
         case kSkipP + kUChar:  SkipBasicPointer(UChar_t)
         case kSkipP + kUShort: SkipBasicPointer(UShort_t)
         case kSkipP + kUInt:   SkipBasicPointer(UInt_t)
         case kSkipP + kULong:  SkipBasicPointer(ULong_t)

         // skip Class *  derived from TObject with comment field  //->
         case kSkip + kObjectp: {
                                 UInt_t start, count;
                                 b.ReadVersion(&start, &count);
                                 b.SetBufferOffset(start+count+sizeof(UInt_t));
                                 break;
                                }

         // skip Class*   derived from TObject
         case kSkip + kObjectP: {
                                 UInt_t start, count;
                                 b.ReadVersion(&start, &count);
                                 b.SetBufferOffset(start+count+sizeof(UInt_t));
                                 break;
                                }

         // skip array counter //[n]
         case kSkip + kCounter: { Int_t dummy; b >> dummy; break;}

         // skip Class    derived from TObject
         case kSkip + kObject:  {
                                 UInt_t start, count;
                                 b.ReadVersion(&start, &count);
                                 b.SetBufferOffset(start+count+sizeof(UInt_t));
                                 break;
                                }

         // skip Special case for TString, TObject, TNamed
         case kSkip + kTString: { TString s; s.Streamer(b); break;}
         case kSkip + kTObject: { TObject x; x.Streamer(b); break;}
         case kSkip + kTNamed:  { TNamed  n; n.Streamer(b); break;}

         // skip Any Class not derived from TObject
         case kSkip + kAny:     {
                                 UInt_t start, count;
                                 b.ReadVersion(&start, &count);
                                 b.SetBufferOffset(start+count+sizeof(UInt_t));
                                 break;
                                }
         // skip Base Class
         case kSkip + kBase:    {
                                 UInt_t start, count;
                                 b.ReadVersion(&start, &count);
                                 b.SetBufferOffset(start+count+sizeof(UInt_t));
                                 break;
                                }

         case kSkip + kStreamLoop:
         case kSkip + kStreamer: {
                         UInt_t start,count;
                         b.ReadVersion(&start,&count);
                         //printf("Skipping Streamer start=%d, count=%d\n",start,count);
                         b.SetBufferOffset(start + count + sizeof(UInt_t));
                         break;
                        }

         // convert basic types
         case kConv + kChar:    ConvBasicType(Char_t)
         case kConv + kShort:   ConvBasicType(Short_t)
         case kConv + kInt:     ConvBasicType(Int_t)
         case kConv + kLong:    ConvBasicType(Long_t)
         case kConv + kFloat:   ConvBasicType(Float_t)
         case kConv + kDouble:  ConvBasicType(Double_t)
         case kConv + kUChar:   ConvBasicType(UChar_t)
         case kConv + kUShort:  ConvBasicType(UShort_t)
         case kConv + kUInt:    ConvBasicType(UInt_t)
         case kConv + kULong:   ConvBasicType(ULong_t)

         // convert array of basic types  array[8]
         case kConvL + kChar:   ConvBasicArray(Char_t)
         case kConvL + kShort:  ConvBasicArray(Short_t)
         case kConvL + kInt:    ConvBasicArray(Int_t)
         case kConvL + kLong:   ConvBasicArray(Long_t)
         case kConvL + kFloat:  ConvBasicArray(Float_t)
         case kConvL + kDouble: ConvBasicArray(Double_t)
         case kConvL + kUChar:  ConvBasicArray(UChar_t)
         case kConvL + kUShort: ConvBasicArray(UShort_t)
         case kConvL + kUInt:   ConvBasicArray(UInt_t)
         case kConvL + kULong:  ConvBasicArray(ULong_t)

         // convert pointer to an array of basic types  array[n]
         case kConvP + kChar:   ConvBasicPointer(Char_t)
         case kConvP + kShort:  ConvBasicPointer(Short_t)
         case kConvP + kInt:    ConvBasicPointer(Int_t)
         case kConvP + kLong:   ConvBasicPointer(Long_t)
         case kConvP + kFloat:  ConvBasicPointer(Float_t)
         case kConvP + kDouble: ConvBasicPointer(Double_t)
         case kConvP + kUChar:  ConvBasicPointer(UChar_t)
         case kConvP + kUShort: ConvBasicPointer(UShort_t)
         case kConvP + kUInt:   ConvBasicPointer(UInt_t)
         case kConvP + kULong:  ConvBasicPointer(ULong_t)
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TStreamerInfo::ReadBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc, Int_t first, Int_t eoffset)
{
//  The TClonesArray clones is deserialized from the buffer b


   char *pointer=0;
   UInt_t start, count;
   Int_t leng,offset;

//==========CPP macros
#define ReadCBasicType(name) \
{ \
   for (Int_t k=0;k<nc;k++) { \
      name *x=(name*)((char*)clones->UncheckedAt(k)+offset); \
      b >> *x; \
   } break; \
}

#define ReadCBasicArray(name) \
{ \
   for (Int_t k=0;k<nc;k++) { \
      name *x=(name*)((char*)clones->UncheckedAt(k)+offset); \
      b.ReadFastArray(x,leng); \
   } break; \
}

#define ReadCBasicPointer(name) \
{ \
   Char_t isArray; \
   for (Int_t k=0;k<nc;k++) { \
      b >> isArray; \
      if (isArray == 0) continue; \
      pointer = (char*)clones->UncheckedAt(k)+eoffset; \
      Int_t *l = (Int_t*)(pointer+fMethod[i]); \
      name **f = (name**)(pointer+fOffset[i]); \
      delete [] *f; \
      *f = 0; if (*l ==0) continue; \
      *f = new name[*l]; \
      b.ReadFastArray(*f,*l); \
   } break; \
}

#define SkipCBasicType(name) \
{ \
   name dummy; \
   for (Int_t k=0;k<nc;k++) b >> dummy; \
   break; \
}

#define SkipCBasicArray(name) \
{  name dummy; \
   for (Int_t k=0;k<nc;k++) { \
      for (Int_t j=0;j<fLength[i];j++) b >> dummy; \
   } \
   break; \
}

#define SkipCBasicPointer(name) \
{ \
   Int_t *n = (Int_t*)(pointer+fMethod[i]); \
   Int_t l = b.Length(); \
   b.SetBufferOffset(l+1+nc*(*n)*sizeof( name )); \
   break; \
}
//==========

   if (!fType) {
      fClass->BuildRealData();
      BuildOld();
   }
   //loop on all active members
   Int_t last;
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;
   for (Int_t i=first;i<last;i++) {
      leng   = fLength[i];
      offset = eoffset + fOffset[i];
      if (gDebug > 1) {
         TStreamerElement *element = (TStreamerElement*)fElem[i];
         printf("ReadBufferClones, class:%s, name=%s, fType[%d]=%d, offset=%d,  %s, bufpos=%d, nc=%d\n",fClass->GetName(),element->GetName(),i,fType[i],offset,element->ClassName(),b.Length(),nc);
      }
      switch (fType[i]) {
         // write basic types
         case kChar:              ReadCBasicType(Char_t)
         case kShort:             ReadCBasicType(Short_t)
         case kInt:               ReadCBasicType(Int_t)
         case kLong:              ReadCBasicType(Long_t)
         case kFloat:             ReadCBasicType(Float_t)
         case kDouble:            ReadCBasicType(Double_t)
         case kUChar:             ReadCBasicType(UChar_t)
         case kUShort:            ReadCBasicType(UShort_t)
         case kUInt:              ReadCBasicType(UInt_t)
         case kULong:             ReadCBasicType(ULong_t)

         // write array of basic types  array[8]
         case kOffsetL + kChar:   ReadCBasicArray(Char_t)
         case kOffsetL + kShort:  ReadCBasicArray(Short_t)
         case kOffsetL + kInt:    ReadCBasicArray(Int_t)
         case kOffsetL + kLong:   ReadCBasicArray(Long_t)
         case kOffsetL + kFloat:  ReadCBasicArray(Float_t)
         case kOffsetL + kDouble: ReadCBasicArray(Double_t)
         case kOffsetL + kUChar:  ReadCBasicArray(UChar_t)
         case kOffsetL + kUShort: ReadCBasicArray(UShort_t)
         case kOffsetL + kUInt:   ReadCBasicArray(UInt_t)
         case kOffsetL + kULong:  ReadCBasicArray(ULong_t)

         // write pointer to an array of basic types  array[n]
         case kOffsetP + kChar:   ReadCBasicPointer(Char_t)
         case kOffsetP + kShort:  ReadCBasicPointer(Short_t)
         case kOffsetP + kInt:    ReadCBasicPointer(Int_t)
         case kOffsetP + kLong:   ReadCBasicPointer(Long_t)
         case kOffsetP + kFloat:  ReadCBasicPointer(Float_t)
         case kOffsetP + kDouble: ReadCBasicPointer(Double_t)
         case kOffsetP + kUChar:  ReadCBasicPointer(UChar_t)
         case kOffsetP + kUShort: ReadCBasicPointer(UShort_t)
         case kOffsetP + kUInt:   ReadCBasicPointer(UInt_t)
         case kOffsetP + kULong:  ReadCBasicPointer(ULong_t)

         // Class *  Class derived from TObject and with comment field //->
         case kObjectp: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               TObject **obj = (TObject**)(pointer+offset);
               if (!(*obj)) {
                  TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                  *obj = (TObject*)el->GetClass()->New();
               }
               (*obj)->Streamer(b);
            }
            break;}

         // Class*   Class derived from TObject
         case kObjectP: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               TObject **obj = (TObject**)(pointer+offset);
               b >> *obj;
            }
            break;}

         // array counter [n]
         case kCounter: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               Int_t *x=(Int_t*)(pointer+offset);
               b >> *x;
               //Int_t *counter = (Int_t*)fMethod[i];
               //*counter = *x;
            }
            break;}

         // Class  derived from TObject
         case kObject:  {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               ((TObject*)(pointer+offset))->Streamer(b);
            }
            break;}

         // Special case for TString, TObject, TNamed
         case kTString: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               ((TString*)(pointer+offset))->Streamer(b);
            }
            break;}
         case kTObject: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               ((TObject*)(pointer+offset))->TObject::Streamer(b);
            }
            break;}
         case kTNamed:  {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k);
               ((TNamed*) (pointer+offset))->TNamed::Streamer(b);
            }
            break;}

         // Any Class not derived from TObject
         case kOffsetL + kObjectp:
         case kOffsetL + kObjectP:
         case kAny: {
                     TStreamerElement *element = (TStreamerElement*)fElem[i];
                     Streamer_t pstreamer = element->GetStreamer();
                     if (pstreamer == 0) {
                        if (gDebug > 0) printf("Warning, Streamer is null\n");
                        TClass *cle = element->GetClassPointer();
                        Int_t kk;
                        if (cle->InheritsFrom(TArray::Class())) {
                           //special case (frequent) with TArray classes
                           //The TArray Streamers not compatible with ReadBuffer
                           // (no byte count)
                           if (strchr(element->GetTypeName(),'*')) {
                              for (kk=0;kk<nc;kk++) {
                                 pointer = (char*)clones->UncheckedAt(kk);
                                 if (cle == TArrayI::Class()) {TArrayI **ar = (TArrayI**)(pointer+offset); b >> *ar;}
                                 if (cle == TArrayF::Class()) {TArrayF **ar = (TArrayF**)(pointer+offset); b >> *ar;}
                                 if (cle == TArrayC::Class()) {TArrayC **ar = (TArrayC**)(pointer+offset); b >> *ar;}
                                 if (cle == TArrayD::Class()) {TArrayD **ar = (TArrayD**)(pointer+offset); b >> *ar;}
                                 if (cle == TArrayS::Class()) {TArrayS **ar = (TArrayS**)(pointer+offset); b >> *ar;}
                                 if (cle == TArrayL::Class()) {TArrayL **ar = (TArrayL**)(pointer+offset); b >> *ar;}
                              }
                              break;
                           } else {
                              for (kk=0;kk<nc;kk++) {
                                 pointer = (char*)clones->UncheckedAt(kk);
                                 if (cle == TArrayI::Class()) {TArrayI *ar = (TArrayI*)(pointer+offset); ar->Streamer(b);}
                                 if (cle == TArrayF::Class()) {TArrayF *ar = (TArrayF*)(pointer+offset); ar->Streamer(b);}
                                 if (cle == TArrayC::Class()) {TArrayC *ar = (TArrayC*)(pointer+offset); ar->Streamer(b);}
                                 if (cle == TArrayD::Class()) {TArrayD *ar = (TArrayD*)(pointer+offset); ar->Streamer(b);}
                                 if (cle == TArrayS::Class()) {TArrayS *ar = (TArrayS*)(pointer+offset); ar->Streamer(b);}
                                 if (cle == TArrayL::Class()) {TArrayL *ar = (TArrayL*)(pointer+offset); ar->Streamer(b);}
                              }
                              break;
                           }
                        }
                        for (kk=0;kk<nc;kk++) {
                           pointer = (char*)clones->UncheckedAt(kk);
                           cle->ReadBuffer(b,pointer+offset);
                        }
                        break;
                     }
                     for (Int_t k=0;k<nc;k++) {
                        pointer = (char*)clones->UncheckedAt(k);
                        (*pstreamer)(b,pointer+offset,0);
                     }
                     break;
                   }

         // Base Class
         case kBase:    {
            TStreamerBase *element = (TStreamerBase*)fElem[i];
            TClass *clbase = element->GetClassPointer();
            Int_t clversion = element->GetBaseVersion();
            clbase->GetStreamerInfo(clversion)->ReadBufferClones(b,clones,nc,-1,0);
            //for (Int_t k=0;k<nc;k++) {
            //   pointer = (char*)clones->UncheckedAt(k)+baseOffset;
            //   element->ReadBuffer(b,pointer);
            //}
            break;}

         case kStreamer: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            printf("ERROR, Streamer is null\n");
                            element->ls();
                            break;
                         }
                         UInt_t start,count;
                         b.ReadVersion(&start,&count);
                         for (Int_t k=0;k<nc;k++) {
                            pointer = (char*)clones->UncheckedAt(k);
                            (*pstreamer)(b,pointer+offset,0);
                         }
                         b.CheckByteCount(start,count,IsA());
                         break;
                        }

         case kStreamLoop: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            printf("ERROR, Streamer is null\n");
                            element->ls();
                            break;
                         }
                         UInt_t start,count;
                         b.ReadVersion(&start,&count);
                         for (Int_t k=0;k<nc;k++) {
                            pointer = (char*)clones->UncheckedAt(k)+offset;
                            Int_t *counter = (Int_t*)(pointer+fMethod[i]);
                            (*pstreamer)(b,pointer+offset,*counter);
                         }
                         b.CheckByteCount(start,count,IsA());
                         break;
                        }


         // skip basic types
         case kSkip + kChar:    SkipCBasicType(Char_t)
         case kSkip + kShort:   SkipCBasicType(Short_t)
         case kSkip + kInt:     SkipCBasicType(Int_t)
         case kSkip + kLong:    SkipCBasicType(Long_t)
         case kSkip + kFloat:   SkipCBasicType(Float_t)
         case kSkip + kDouble:  SkipCBasicType(Double_t)
         case kSkip + kUChar:   SkipCBasicType(UChar_t)
         case kSkip + kUShort:  SkipCBasicType(UShort_t)
         case kSkip + kUInt:    SkipCBasicType(UInt_t)
         case kSkip + kULong:   SkipCBasicType(ULong_t)

         // skip array of basic types  array[8]
         case kSkipL + kChar:   SkipCBasicArray(Char_t)
         case kSkipL + kShort:  SkipCBasicArray(Short_t)
         case kSkipL + kInt:    SkipCBasicArray(Int_t)
         case kSkipL + kLong:   SkipCBasicArray(Long_t)
         case kSkipL + kFloat:  SkipCBasicArray(Float_t)
         case kSkipL + kDouble: SkipCBasicArray(Double_t)
         case kSkipL + kUChar:  SkipCBasicArray(UChar_t)
         case kSkipL + kUShort: SkipCBasicArray(UShort_t)
         case kSkipL + kUInt:   SkipCBasicArray(UInt_t)
         case kSkipL + kULong:  SkipCBasicArray(ULong_t)

         // skip pointer to an array of basic types  array[n]
         case kSkipP + kChar:   SkipCBasicPointer(Char_t)
         case kSkipP + kShort:  SkipCBasicPointer(Short_t)
         case kSkipP + kInt:    SkipCBasicPointer(Int_t)
         case kSkipP + kLong:   SkipCBasicPointer(Long_t)
         case kSkipP + kFloat:  SkipCBasicPointer(Float_t)
         case kSkipP + kDouble: SkipCBasicPointer(Double_t)
         case kSkipP + kUChar:  SkipCBasicPointer(UChar_t)
         case kSkipP + kUShort: SkipCBasicPointer(UShort_t)
         case kSkipP + kUInt:   SkipCBasicPointer(UInt_t)
         case kSkipP + kULong:  SkipCBasicPointer(ULong_t)

         // skip Class *  derived from TObject with comment field  //->
         case kSkip + kObjectp: {
            for (Int_t k=0;k<nc;k++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
            break;}

         // skip Class*   derived from TObject
         case kSkip + kObjectP: {
            for (Int_t k=0;k<nc;k++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
            break;}

         // skip array counter //[n]
         case kSkip + kCounter: {
            for (Int_t k=0;k<nc;k++) {
               //Int_t *counter = (Int_t*)fMethod[i];
               //b >> *counter;
               Int_t dummy; b >> dummy;
            }
            break;}

         // skip Class    derived from TObject
         case kSkip + kObject:  {
            for (Int_t k=0;k<nc;k++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
            break;}

         // skip Special case for TString, TObject, TNamed
         case kSkip + kTString: {
            TString s;
            for (Int_t k=0;k<nc;k++) {
               s.Streamer(b);
            }
            break;}
         case kSkip + kTObject: {
            TObject x;
            for (Int_t k=0;k<nc;k++) {
               x.Streamer(b);
            }
            break;}
         case kSkip + kTNamed:  {
            TNamed n;
            for (Int_t k=0;k<nc;k++) {
               n.Streamer(b);
            }
            break;}

         // skip Any Class not derived from TObject
         case kSkip + kAny:     {
            for (Int_t k=0;k<nc;k++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
            break;}

         // skip Base Class
         case kSkip + kBase:    {
            for (Int_t k=0;k<nc;k++) {
               b.ReadVersion(&start, &count);
               b.SetBufferOffset(start+count+sizeof(UInt_t));
            }
            break;}

         case kSkip + kStreamLoop:
         case kSkip + kStreamer: {
                         UInt_t start,count;
                         b.ReadVersion(&start,&count);
                         b.SetBufferOffset(start + count + sizeof(UInt_t));
                         break;
                        }
      }
   }
   return 0;
}

//______________________________________________________________________________
void TStreamerInfo::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerInfo.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TStreamerInfo::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TNamed::Streamer(R__b);
      R__b >> fCheckSum;
      R__b >> fClassVersion;
      R__b >> fElements;
      R__b.CheckByteCount(R__s, R__c, TStreamerInfo::IsA());
   } else {
      TStreamerInfo::Class()->WriteBuffer(R__b,this);
   }
}

//______________________________________________________________________________
void TStreamerInfo::TagFile()
{
   // Mark the classindex of teh current file as using this TStreamerInfo

   //TFile *file = gDirectory->GetFile();
   TFile *file = gFile;
   if (file) {
      TArrayC *cindex = file->GetClassIndex();
      Int_t nindex = cindex->GetSize();
      if (fNumber < 0 || fNumber >= nindex) {
         Error("TagFile","StreamerInfo: %s number: %d out of range[0,%d] in file: %s",
            GetName(),fNumber,nindex,file->GetName());
         return;
      }
      if (cindex->fArray[fNumber] == 0) {
         cindex->fArray[0]       = 1;
         cindex->fArray[fNumber] = 1;
      }
   }

}

//______________________________________________________________________________
Int_t TStreamerInfo::WriteBuffer(TBuffer &b, char *pointer, Int_t first)
{
//  The object at pointer is serialized to the buffer b

   //mark this class as being used in the current file
   TagFile();

//==========CPP macros
#define WriteBasicType(name) \
{ \
   name *x=(name*)(pointer+fOffset[i]); \
   b << *x; \
   break; \
}

#define WriteBasicArray(name) \
{ \
   name *x=(name*)(pointer+fOffset[i]); \
   b.WriteFastArray(x,fLength[i]); \
   break; \
}

#define WriteBasicPointer(name) \
{ \
   Int_t *l = (Int_t*)(pointer+fMethod[i]); \
   name **f = (name**)(pointer+fOffset[i]); \
   name *af = *f; \
   if (af)  b << Char_t(1); \
   else    {b << Char_t(0); break;}\
   b.WriteFastArray(af,*l); \
   break; \
}
//==========


   //loop on all active members
   Int_t last;
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;
   for (Int_t i=first;i<last;i++) {
      if (gDebug > 1) {
         TStreamerElement *element = (TStreamerElement*)fElem[i];
         printf("WriteBuffer, class:%s, name=%s, fType[%d]=%d, %s, bufpos=%d, pointer=%lx, offset=%d\n",fClass->GetName(),element->GetName(),i,fType[i],element->ClassName(),b.Length(),(Long_t)pointer,fOffset[i]);
      }
      switch (fType[i]) {
         // write basic types
         case kChar:              WriteBasicType(Char_t)
         case kShort:             WriteBasicType(Short_t)
         case kInt:               WriteBasicType(Int_t)
         case kLong:              WriteBasicType(Long_t)
         case kFloat:             WriteBasicType(Float_t)
         case kDouble:            WriteBasicType(Double_t)
         case kUChar:             WriteBasicType(UChar_t)
         case kUShort:            WriteBasicType(UShort_t)
         case kUInt:              WriteBasicType(UInt_t)
         case kULong:             WriteBasicType(ULong_t)

         // write array of basic types  array[8]
         case kOffsetL + kChar:   WriteBasicArray(Char_t)
         case kOffsetL + kShort:  WriteBasicArray(Short_t)
         case kOffsetL + kInt:    WriteBasicArray(Int_t)
         case kOffsetL + kLong:   WriteBasicArray(Long_t)
         case kOffsetL + kFloat:  WriteBasicArray(Float_t)
         case kOffsetL + kDouble: WriteBasicArray(Double_t)
         case kOffsetL + kUChar:  WriteBasicArray(UChar_t)
         case kOffsetL + kUShort: WriteBasicArray(UShort_t)
         case kOffsetL + kUInt:   WriteBasicArray(UInt_t)
         case kOffsetL + kULong:  WriteBasicArray(ULong_t)

         // write pointer to an array of basic types  array[n]
         case kOffsetP + kChar:   WriteBasicPointer(Char_t)
         case kOffsetP + kShort:  WriteBasicPointer(Short_t)
         case kOffsetP + kInt:    WriteBasicPointer(Int_t)
         case kOffsetP + kLong:   WriteBasicPointer(Long_t)
         case kOffsetP + kFloat:  WriteBasicPointer(Float_t)
         case kOffsetP + kDouble: WriteBasicPointer(Double_t)
         case kOffsetP + kUChar:  WriteBasicPointer(UChar_t)
         case kOffsetP + kUShort: WriteBasicPointer(UShort_t)
         case kOffsetP + kUInt:   WriteBasicPointer(UInt_t)
         case kOffsetP + kULong:  WriteBasicPointer(ULong_t)

         // Class *  Class derived from TObject and with comment field //->
         case kObjectp: { TObject **obj = (TObject**)(pointer+fOffset[i]);
                          if (!(*obj)) {
                             TStreamerObjectPointer *el = (TStreamerObjectPointer*)fElem[i];
                             if (gDebug) {
                                Error("WriteBuffer","-> specified but pointer is null");
                                el->ls();
                             }
                             *obj = (TObject*)el->GetClass()->New();
                             (*obj)->SetBit(kZombie);
                          }
                          (*obj)->Streamer(b);
                          break;
                        }

         // Class*   Class derived from TObject
         case kObjectP: { TObject **obj = (TObject**)(pointer+fOffset[i]);
                          //must write StreamerInfo if pointer is null
                          if (!(*obj)) {
                             TStreamerObjectPointer *element = (TStreamerObjectPointer*)fElem[i];
                             TClass *cl = element->GetClass();
                             cl->GetStreamerInfo()->ForceWriteInfo();
                          }
                          b << *obj;
                          break;
                        }

         // array counter [n]
         case kCounter: { Int_t *x=(Int_t*)(pointer+fOffset[i]);
                          b << x[0];
                          if ( i == last-1) return x[0]; // info used by TBranchElement::FillLeaves
                          //Int_t *counter = (Int_t*)fMethod[i];
                          //*counter = x[0];
                          break;
                        }

         // Class  derived from TObject
         case kObject:  { ((TObject*)(pointer+fOffset[i]))->Streamer(b); break;}

         // Special case for TString, TObject, TNamed
         case kTString: { ((TString*)(pointer+fOffset[i]))->Streamer(b); break;}
         case kTObject: { ((TObject*)(pointer+fOffset[i]))->TObject::Streamer(b); break;}
         case kTNamed:  { ((TNamed*) (pointer+fOffset[i]))->TNamed::Streamer(b); break;}

         // Any Class not derived from TObject
         case kOffsetL + kObjectp:
         case kOffsetL + kObjectP:
         case kAny:     {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            if (gDebug > 0) printf("WARNING, Streamer is null\n");
                            //Note that this does not work if the class has a custom Streamer
                            //with no bytecount
                            TClass *cle = element->GetClassPointer();
                            if (cle->InheritsFrom(TArray::Class())) {
                               //special case (frequent) with TArray classes
                               //The TArray Streamers not compatible with WriteBuffer
                               // (no byte count)
                               if (strchr(element->GetTypeName(),'*')) {
                                  if (cle == TArrayI::Class()) {TArrayI **ar = (TArrayI**)(pointer+fOffset[i]); b << *ar; break;}
                                  if (cle == TArrayF::Class()) {TArrayF **ar = (TArrayF**)(pointer+fOffset[i]); b << *ar; break;}
                                  if (cle == TArrayC::Class()) {TArrayC **ar = (TArrayC**)(pointer+fOffset[i]); b << *ar; break;}
                                  if (cle == TArrayD::Class()) {TArrayD **ar = (TArrayD**)(pointer+fOffset[i]); b << *ar; break;}
                                  if (cle == TArrayS::Class()) {TArrayS **ar = (TArrayS**)(pointer+fOffset[i]); b << *ar; break;}
                                  if (cle == TArrayL::Class()) {TArrayL **ar = (TArrayL**)(pointer+fOffset[i]); b << *ar; break;}
                               } else {
                                  if (cle == TArrayI::Class()) {TArrayI *ar = (TArrayI*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayF::Class()) {TArrayF *ar = (TArrayF*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayC::Class()) {TArrayC *ar = (TArrayC*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayD::Class()) {TArrayD *ar = (TArrayD*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayS::Class()) {TArrayS *ar = (TArrayS*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                                  if (cle == TArrayL::Class()) {TArrayL *ar = (TArrayL*)(pointer+fOffset[i]); ar->Streamer(b); break;}
                               }
                            }
                            cle->WriteBuffer(b,pointer+fOffset[i],"");
                            break;
                         }
                         (*pstreamer)(b,pointer+fOffset[i],0);
                         break;
                        }
         // Base Class
         case kBase:    { TStreamerBase *element = (TStreamerBase*)fElem[i];
                          element->WriteBuffer(b,pointer);
                          break;
                        }

         case kStreamer: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            printf("ERROR, Streamer is null\n");
                            element->ls();
                            break;
                         }
                         UInt_t pos = b.WriteVersion(IsA(),kTRUE);
                         (*pstreamer)(b,pointer+fOffset[i],0);
                         b.SetByteCount(pos,kTRUE);
                         break;
                        }

         case kStreamLoop: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            printf("ERROR, Streamer is null\n");
                            element->ls();
                            break;
                         }
                         Int_t *counter = (Int_t*)(pointer+fMethod[i]);
                         UInt_t pos = b.WriteVersion(IsA(),kTRUE);
                         (*pstreamer)(b,pointer+fOffset[i],*counter);
                         b.SetByteCount(pos,kTRUE);
                         break;
                        }
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TStreamerInfo::WriteBufferClones(TBuffer &b, TClonesArray *clones, Int_t nc, Int_t first, Int_t eoffset)
{
//  The TClonesArray clones is serialized to the buffer b

   //mark this class as being used in the current file
   TagFile();

   char *pointer;

//==========CPP macros
#define WriteCBasicType(name) \
{ \
   for (Int_t k=0;k<nc;k++) { \
      pointer = (char*)clones->UncheckedAt(k)+baseOffset; \
      name *x = (name*)(pointer+fOffset[i]); \
      b << *x; \
   } \
   break; \
}

#define WriteCBasicArray(name) \
{ \
   for (Int_t k=0;k<nc;k++) { \
      pointer = (char*)clones->UncheckedAt(k)+baseOffset; \
      name *x = (name*)(pointer+fOffset[i]); \
      b.WriteFastArray(x,fLength[i]); \
   } \
   break; \
}

#define WriteCBasicPointer(name) \
{ \
   for (Int_t k=0;k<nc;k++) { \
      pointer = (char*)clones->UncheckedAt(k)+baseOffset; \
      Int_t *l = (Int_t*)(pointer+fMethod[i]); \
      name **f = (name**)(pointer+fOffset[i]); \
      name *af = *f; \
      if (af)  b << Char_t(1); \
      else    {b << Char_t(0); continue;} \
      b.WriteFastArray(af,*l); \
   } \
   break; \
}
//==========

   //loop on all active members
   Int_t baseOffset = eoffset;
   Int_t last;
   if (first < 0) {first = 0; last = fNdata;}
   else            last = first+1;
   for (Int_t i=first;i<last;i++) {
      if (gDebug > 1) {
         TStreamerElement *element = (TStreamerElement*)fElem[i];
         printf("WriteBufferClones, class:%s, name=%s, fType[%d]=%d, %s, bufpos=%d, offset=%d\n",fClass->GetName(),element->GetName(),i,fType[i],element->ClassName(),b.Length(),fOffset[i]);
      }
      switch (fType[i]) {
         // write basic types
         case kChar:              WriteCBasicType(Char_t)
         case kShort:             WriteCBasicType(Short_t)
         case kInt:               WriteCBasicType(Int_t)
         case kLong:              WriteCBasicType(Long_t)
         case kFloat:             WriteCBasicType(Float_t)
         case kDouble:            WriteCBasicType(Double_t)
         case kUChar:             WriteCBasicType(UChar_t)
         case kUShort:            WriteCBasicType(UShort_t)
         case kUInt:              WriteCBasicType(UInt_t)
         case kULong:             WriteCBasicType(ULong_t)

         // write array of basic types  array[8]
         case kOffsetL + kChar:   WriteCBasicArray(Char_t)
         case kOffsetL + kShort:  WriteCBasicArray(Short_t)
         case kOffsetL + kInt:    WriteCBasicArray(Int_t)
         case kOffsetL + kLong:   WriteCBasicArray(Long_t)
         case kOffsetL + kFloat:  WriteCBasicArray(Float_t)
         case kOffsetL + kDouble: WriteCBasicArray(Double_t)
         case kOffsetL + kUChar:  WriteCBasicArray(UChar_t)
         case kOffsetL + kUShort: WriteCBasicArray(UShort_t)
         case kOffsetL + kUInt:   WriteCBasicArray(UInt_t)
         case kOffsetL + kULong:  WriteCBasicArray(ULong_t)

         // write pointer to an array of basic types  array[n]
         case kOffsetP + kChar:   WriteCBasicPointer(Char_t)
         case kOffsetP + kShort:  WriteCBasicPointer(Short_t)
         case kOffsetP + kInt:    WriteCBasicPointer(Int_t)
         case kOffsetP + kLong:   WriteCBasicPointer(Long_t)
         case kOffsetP + kFloat:  WriteCBasicPointer(Float_t)
         case kOffsetP + kDouble: WriteCBasicPointer(Double_t)
         case kOffsetP + kUChar:  WriteCBasicPointer(UChar_t)
         case kOffsetP + kUShort: WriteCBasicPointer(UShort_t)
         case kOffsetP + kUInt:   WriteCBasicPointer(UInt_t)
         case kOffsetP + kULong:  WriteCBasicPointer(ULong_t)

         // Class *  Class derived from TObject and with comment field //->
         case kObjectp: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               TObject **obj = (TObject**)(pointer+fOffset[i]);
               if (*obj) (*obj)->Streamer(b);
               else {
                  Error("WriteBufferCones","-> specified but pointer is null");
                  TStreamerElement *element = (TStreamerElement*)fElem[i];
                  element->ls();
               }
            }
            break;}

         // Class*   Class derived from TObject
         case kObjectP: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               TObject **obj = (TObject**)(pointer+fOffset[i]);
               b << *obj;
            }
            break;}

         // array counter [n]
         case kCounter: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               Int_t *x=(Int_t*)(pointer+fOffset[i]);
               b << x[0];
               //*counter = x[0];
            }
            break;}

         // Class  derived from TObject
         case kObject:  {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               ((TObject*)(pointer+fOffset[i]))->Streamer(b);
            }
            break;}

         // Special case for TString, TObject, TNamed
         case kTString: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               ((TString*)(pointer+fOffset[i]))->Streamer(b);
            }
            break;}
         case kTObject: {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               ((TObject*)(pointer+fOffset[i]))->TObject::Streamer(b);
            }
            break;}
         case kTNamed:  {
            for (Int_t k=0;k<nc;k++) {
               pointer = (char*)clones->UncheckedAt(k)+baseOffset;
               ((TNamed*) (pointer+fOffset[i]))->TNamed::Streamer(b);
            }
            break;}

         // Any Class not derived from TObject
         case kOffsetL + kObjectp:
         case kOffsetL + kObjectP:
         case kAny: {
                     TStreamerElement *element = (TStreamerElement*)fElem[i];
                     Streamer_t pstreamer = element->GetStreamer();
                     if (pstreamer == 0) {
                        printf("ERROR, Streamer is null\n");
                        element->ls();
                        break;
                     }
                     for (Int_t k=0;k<nc;k++) {
                        pointer = (char*)clones->UncheckedAt(k)+baseOffset;
                        (*pstreamer)(b,pointer+fOffset[i],0);
                     }
                     break;
                    }

         // Base Class
         case kBase: {
                       TStreamerBase *element = (TStreamerBase*)fElem[i];
                       //for (Int_t k=0;k<nc;k++) {
                       //   pointer = (char*)clones->UncheckedAt(k)+baseOffset;
                       //   element->WriteBuffer(b,pointer);
                       //}
                       TClass *clbase = element->GetClassPointer();
                       clbase->GetStreamerInfo()->WriteBufferClones(b,clones,nc,-1,0);
                       break;
                     }

         case kStreamer: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            printf("ERROR, Streamer is null\n");
                            element->ls();
                            break;
                         }
                         UInt_t pos = b.WriteVersion(IsA(),kTRUE);
                         for (Int_t k=0;k<nc;k++) {
                            pointer = (char*)clones->UncheckedAt(k)+baseOffset;
                            (*pstreamer)(b,pointer+fOffset[i],0);
                         }
                         b.SetByteCount(pos,kTRUE);
                         break;
                        }

         case kStreamLoop: {
                         TStreamerElement *element = (TStreamerElement*)fElem[i];
                         Streamer_t pstreamer = element->GetStreamer();
                         if (pstreamer == 0) {
                            printf("ERROR, Streamer is null\n");
                            element->ls();
                            break;
                         }
                         UInt_t pos = b.WriteVersion(IsA(),kTRUE);
                         for (Int_t k=0;k<nc;k++) {
                            pointer = (char*)clones->UncheckedAt(k)+baseOffset;
                            Int_t *counter = (Int_t*)(pointer+fMethod[i]);
                            (*pstreamer)(b,pointer+fOffset[i],*counter);
                         }
                         b.SetByteCount(pos,kTRUE);
                         break;
                        }
      }
   }
   return 0;
}
