// @(#)Root/io:$Id$
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
// A TStreamerInfo object describes a persistent version of a class.    //
// A ROOT file contains the list of TStreamerInfo objects for all the   //
// class versions written to this file.                                 //
// When reading a file, all the TStreamerInfo objects are read back in  //
// memory and registered to the TClass list of TStreamerInfo.           //
//                                                                      //
// One can see the list and contents of the TStreamerInfo on a file     //
// with, eg,                                                            //
//    TFile f("myfile.root");                                           //
//    f.ShowStreamerInfo();                                             //
//                                                                      //
// A TStreamerInfo is a list of TStreamerElement objects (one per data  //
// member or base class).                                               //
// When streaming an object, the system (TClass) loops on all the       //
// TStreamerElement objects and calls teh appropriate function for each //
// element type.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStreamerInfo.h"
#include "TFile.h"
#include "TROOT.h"
#include "TClonesArray.h"
#include "TStreamerElement.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TDataMember.h"
#include "TMethodCall.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TBaseClass.h"
#include "TBuffer.h"
#include "TArrayC.h"
#include "TArrayI.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TArrayS.h"
#include "TArrayL.h"
#include "TError.h"
#include "TRef.h"
#include "TProcessID.h"
#include "TSystem.h"

#include "TStreamer.h"
#include "TContainerConverters.h"
#include "TCollectionProxyFactory.h"
#include "TVirtualCollectionProxy.h"
#include "TInterpreter.h"

#include "TMemberInspector.h"

#include "TMakeProject.h"

#include "TSchemaRuleSet.h"
#include "TSchemaRule.h"

#include "TVirtualMutex.h"

#include "TStreamerInfoActions.h"

TStreamerElement *TStreamerInfo::fgElement = 0;
Int_t   TStreamerInfo::fgCount = 0;

const Int_t kRegrouped = TStreamerInfo::kOffsetL;

const Int_t kMaxLen = 1024;

ClassImp(TStreamerInfo)

static void R__TObjArray_InsertAt(TObjArray *arr, TObject *obj, Int_t at)
{
   // Slide by one.
   Int_t last = arr->GetLast();
   arr->AddAtAndExpand(arr->At(last),last+1);
   for(Int_t ind = last-1; ind >= at; --ind) {
      arr->AddAt( arr->At(ind), ind+1);
   };
   arr->AddAt( obj, at);
}

#if 0
static void R__TObjArray_InsertAfter(TObjArray *arr, TObject *newobj, TObject *oldobj)
{
   // Slide by one.
   Int_t last = arr->GetLast();
   Int_t at = 0;
   while (at<last && arr->At(at) != oldobj) {
      ++at;
   }
   if (at!=0) { 
      ++at; // we found the object, insert after it 
   }
   R__TObjArray_InsertAt(arr, newobj, at);
}
#endif

static void R__TObjArray_InsertBefore(TObjArray *arr, TObject *newobj, TObject *oldobj)
{
   // Slide by one.
   Int_t last = arr->GetLast();
   Int_t at = 0;
   while (at<last && arr->At(at) != oldobj) {
      ++at;
   }
   R__TObjArray_InsertAt(arr, newobj, at);
}

//______________________________________________________________________________
TStreamerInfo::TStreamerInfo()
{
   // Default ctor.

   fNumber   = fgCount;
   fClass    = 0;
   fElements = 0;
   fComp     = 0;
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
   fOnFileClassVersion = 0;
   fOldVersion = Class()->GetClassVersion();
   fNVirtualInfoLoc = 0;
   fVirtualInfoLoc = 0;
   fLiveCount = 0;
   
   fReadObjectWise = 0;
   fReadMemberWise = 0;
}

//______________________________________________________________________________
TStreamerInfo::TStreamerInfo(TClass *cl)
: TVirtualStreamerInfo(cl)
{
   // Create a TStreamerInfo object.

   fgCount++;
   fNumber   = fgCount;
   fClass    = cl;
   fElements = new TObjArray();
   fComp     = 0;
   fType     = 0;
   fNewType  = 0;
   fOffset   = 0;
   fLength   = 0;
   fElem     = 0;
   fMethod   = 0;
   fCheckSum = 0;
   fNdata    = 0;
   fSize     = 0;
   fClassVersion = fClass->GetClassVersion();
   fOnFileClassVersion = 0;
   fOldVersion = Class()->GetClassVersion();
   fNVirtualInfoLoc = 0;
   fVirtualInfoLoc = 0;
   fLiveCount = 0;
   
   fReadObjectWise = 0;
   fReadMemberWise = 0;
}

//______________________________________________________________________________
TStreamerInfo::~TStreamerInfo()
{
   // TStreamerInfo dtor.

   delete [] fType;    fType   =0;
   delete [] fNewType; fNewType=0;
   delete [] fOffset;  fOffset =0;
   delete [] fLength;  fLength =0;
   delete [] fElem;    fElem   =0;
   delete [] fMethod;  fMethod =0;
   delete [] fComp;    fComp   =0;
   delete [] fVirtualInfoLoc; fVirtualInfoLoc =0;

   delete fReadObjectWise;
   delete fReadMemberWise;

   if (!fElements) return;
   fElements->Delete();
   delete fElements; fElements=0;
}

//______________________________________________________________________________
void TStreamerInfo::Build()
{
   // Build the I/O data structure for the current class version.
   // A list of TStreamerElement derived classes is built by scanning
   // one by one the list of data members of the analyzed class.

   R__LOCKGUARD(gCINTMutex);

   // This is used to avoid unwanted recursive call to Build
   fIsBuilt = kTRUE;

   if (fClass->GetCollectionProxy()) {
      //FIXME: What about arrays of STL containers?
      TStreamerElement* element = new TStreamerSTL("This", "Used to call the proper TStreamerInfo case", 0, fClass->GetName(), fClass->GetName(), 0);
      fElements->Add(element);
      Compile();
      return;
   }

   TStreamerElement::Class()->IgnoreTObjectStreamer();

   fClass->BuildRealData();

   fCheckSum = fClass->GetCheckSum();

   Bool_t needAllocClass = kFALSE;
   Bool_t wasCompiled = fOffset != 0;
   const ROOT::TSchemaMatch* rules = 0;
   if (fClass->GetSchemaRules()) {
       rules = fClass->GetSchemaRules()->FindRules(fClass->GetName(), fClassVersion);
   }

   //
   // Iterate over base classes.
   //

   TBaseClass* base = 0;
   TIter nextb(fClass->GetListOfBases());
   while ((base = (TBaseClass*)nextb())) {
      TStreamerElement* element = 0;
      Int_t offset = base->GetDelta();
      if (offset == kMissing) {
         continue;
      }
      const char* bname  = base->GetName();
      const char* btitle = base->GetTitle();
      // this case appears with STL collections as base class.
      if (!strcmp(bname, "string")) {
         element = new TStreamerSTLstring(bname, btitle, offset, bname, kFALSE);
      } else if (base->IsSTLContainer()) {
         element = new TStreamerSTL(bname, btitle, offset, bname, 0, kFALSE);
      } else {
         element = new TStreamerBase(bname, btitle, offset);
         TClass* clm = element->GetClassPointer();
         if (!clm) {
            Error("Build", "%s, unknown type: %s %s\n", GetName(), bname, btitle);
            delete element;
            element = 0;
         } else {
            clm->GetStreamerInfo();
            if ((clm == TObject::Class()) && fClass->CanIgnoreTObjectStreamer()) {
               // -- An ignored TObject base class.
               // Note: The TClass kIgnoreTObjectStreamer == BIT(15), but
               // the TStreamerInfo kIgnoreTobjectStreamer == BIT(13) which
               // is confusing.
               SetBit(kIgnoreTObjectStreamer);
               // Flag the element to be ignored by setting its type to -1.
               // This flag will be used later by Compile() to prevent this
               // element from being inserted into the compiled info.
               element->SetType(-1);
            }
            if (!clm->IsLoaded()) {
               Warning("Build:", "%s: base class %s has no streamer or dictionary it will not be saved", GetName(), clm->GetName());
            }
         }
      }
      if (element) {
         fElements->Add(element);
      }
   } // end of base class loop

   //
   // Iterate over data members.
   //

   Int_t dsize;
   TDataMember* dm = 0;
   TIter nextd(fClass->GetListOfDataMembers());
   while ((dm = (TDataMember*) nextd())) {
      if (fClass->GetClassVersion() == 0) {
         continue;
      }
      if (!dm->IsPersistent()) {
         continue;
      }
      TMemberStreamer* streamer = 0;
      Int_t offset = GetDataMemberOffset(dm, streamer);
      if (offset == kMissing) {
         continue;
      }
      TStreamerElement* element = 0;
      dsize = 0;
      const char* dmName = dm->GetName();
      const char* dmTitle = dm->GetTitle();
      const char* dmType = dm->GetTypeName();
      const char* dmFull = dm->GetFullTypeName();
      Bool_t dmIsPtr = dm->IsaPointer();
      TDataMember* dmCounter = 0;
      if (dmIsPtr) {
         //
         // look for a pointer data member with a counter
         // in the comment string, like so:
         //
         //      int n;
         //      double* MyArray; //[n]
         //
         const char* lbracket = ::strchr(dmTitle, '[');
         const char* rbracket = ::strchr(dmTitle, ']');
         if (lbracket && rbracket) {
            const char* counterName = dm->GetArrayIndex();
            TRealData* rdCounter = (TRealData*) fClass->GetListOfRealData()->FindObject(counterName);
            if (!rdCounter || rdCounter->TestBit(TRealData::kTransient)) {
               Error("Build", "%s, discarding: %s %s, illegal %s\n", GetName(), dmFull, dmName, dmTitle);
               continue;
            }
            dmCounter = rdCounter->GetDataMember();
            TDataType* dtCounter = dmCounter->GetDataType();
            Bool_t isInteger = dtCounter && ((dtCounter->GetType() == 3) || (dtCounter->GetType() == 13));
            if (!dtCounter || !isInteger) {
               Error("Build", "%s, discarding: %s %s, illegal [%s] (must be Int_t)\n", GetName(), dmFull, dmName, counterName);
               continue;
            }
            TStreamerBasicType* bt = TStreamerInfo::GetElementCounter(counterName, dmCounter->GetClass());
            if (!bt) {
               if (dmCounter->GetClass()->Property() & kIsAbstract) {
                  continue;
               }
               Error("Build", "%s, discarding: %s %s, illegal [%s] must be placed before \n", GetName(), dmFull, dmName, counterName);
               continue;
            }
         }
      }
      TDataType* dt = dm->GetDataType();
      if (dt) {
         // found a basic type
         Int_t dtype = dt->GetType();
         dsize = dt->Size();
         if (!dmCounter && (strstr(dmFull, "char*") || strstr(dmFull, "Char_t*"))) {
            dtype = kCharStar;
            dsize = sizeof(char*);
         }
         if (dmIsPtr && (dtype != kCharStar)) {
            if (dmCounter) {
               // data member is pointer to an array of basic types
               element = new TStreamerBasicPointer(dmName, dmTitle, offset, dtype, dm->GetArrayIndex(), dmCounter->GetClass()->GetName(), dmCounter->GetClass()->GetClassVersion(), dmFull);
            } else {
               if ((fName == "TString") || (fName == "TClass")) {
                  continue;
               }
               Error("Build", "%s, discarding: %s %s, no [dimension]\n", GetName(), dmFull, dmName);
               continue;
            }
         } else {
            // data member is a basic type
            if ((fClass == TObject::Class()) && !strcmp(dmName, "fBits")) {
               //printf("found fBits, changing dtype from %d to 15\n", dtype);
               dtype = kBits;
            }
            element = new TStreamerBasicType(dmName, dmTitle, offset, dtype, dmFull);
         }
      } else {
         // try STL container or string
         static const char* full_string_name = "basic_string<char,char_traits<char>,allocator<char> >";
         if (!strcmp(dmType, "string") || !strcmp(dmType, full_string_name)) {
            element = new TStreamerSTLstring(dmName, dmTitle, offset, dmFull, dmIsPtr);
         } else if (dm->IsSTLContainer()) {
            element = new TStreamerSTL(dmName, dmTitle, offset, dmFull, dm->GetTrueTypeName(), dmIsPtr);
         } else {
            TClass* clm = TClass::GetClass(dmType);
            if (!clm) {
               Error("Build", "%s, unknown type: %s %s\n", GetName(), dmFull, dmName);
               continue;
            }
            if (dmIsPtr) {
               // a pointer to a class
               if (dmCounter) {
                  element = new TStreamerLoop(dmName, dmTitle, offset, dm->GetArrayIndex(), dmCounter->GetClass()->GetName(), dmCounter->GetClass()->GetClassVersion(), dmFull);
               } else {
                  if (clm->InheritsFrom(TObject::Class())) {
                     element = new TStreamerObjectPointer(dmName, dmTitle, offset, dmFull);
                  } else {
                     element = new TStreamerObjectAnyPointer(dmName, dmTitle, offset, dmFull);
                     if (!streamer && !clm->GetStreamer() && !clm->IsLoaded()) {
                        Error("Build:", "%s: %s has no streamer or dictionary, data member %s will not be saved", GetName(), dmFull, dmName);
                     }
                  }
               }
            } else if (clm->InheritsFrom(TObject::Class())) {
               element = new TStreamerObject(dmName, dmTitle, offset, dmFull);
            } else if ((clm == TString::Class()) && !dmIsPtr) {
               element = new TStreamerString(dmName, dmTitle, offset);
            } else {
               element = new TStreamerObjectAny(dmName, dmTitle, offset, dmFull);
               if (!streamer && !clm->GetStreamer() && !clm->IsLoaded()) {
                  Warning("Build:", "%s: %s has no streamer or dictionary, data member \"%s\" will not be saved", GetName(), dmFull, dmName);
               }
            }
         }
      }
      if (!element) {
         // If we didn't make an element, there is nothing to do.
         continue;
      }
      Int_t ndim = dm->GetArrayDim();
      if (!dsize) {
         dsize = dm->GetUnitSize();
      }
      for (Int_t i = 0; i < ndim; ++i) {
         element->SetMaxIndex(i, dm->GetMaxIndex(i));
      }
      element->SetArrayDim(ndim);
      Int_t narr = element->GetArrayLength();
      if (!narr) {
         narr = 1;
      }
      element->SetSize(dsize*narr);
      element->SetStreamer(streamer);
      if (!streamer) {
         Int_t k = element->GetType();
         if (k == kStreamer) {
            //if ((k == kSTL) || (k == kSTL + kOffsetL) || (k == kStreamer) || (k == kStreamLoop))
            element->SetType(-1);
         }
      }

      if ( !wasCompiled && (rules && rules->HasRuleWithSource( element->GetName(), kTRUE )) ) {
         needAllocClass = kTRUE;

         // If this is optimized to re-use TStreamerElement(s) in case of variable renaming,
         // then we must revisit the code in TBranchElement::InitInfo that recalculate the
         // fID (i.e. the index of the TStreamerElement to be used for streaming).

         TStreamerElement *cached = element;
         // Now that we are caching the unconverted element, we do not assign it to the real type even if we could have!
         if (element->GetNewType()>0 /* intentionally not including base class for now */ 
             && rules && !rules->HasRuleWithTarget( element->GetName(), kTRUE ) ) 
         {
            TStreamerElement *copy = (TStreamerElement*)element->Clone();
            fElements->Add(copy);
            copy->SetBit(TStreamerElement::kRepeat);
            cached = copy;

            // Warning("BuildOld","%s::%s is not set from the version %d of %s (You must add a rule for it)\n",GetName(), element->GetName(), GetClassVersion(), GetName() );
         }
         cached->SetBit(TStreamerElement::kCache);
         cached->SetNewType( cached->GetType() );
      }

      fElements->Add(element);
   } // end of member loop

   // Now add artificial TStreamerElement (i.e. rules that creates new members or set transient members).
   InsertArtificialElements(rules);

   if (needAllocClass) {
      TStreamerInfo *infoalloc  = (TStreamerInfo *)Clone(TString::Format("%s@@%d",GetName(),GetClassVersion()));
      infoalloc->BuildCheck();
      infoalloc->BuildOld();
      TClass *allocClass = infoalloc->GetClass();

      {
         TIter next(fElements);
         TStreamerElement* element;
         while ((element = (TStreamerElement*) next())) {
            if (element->TestBit(TStreamerElement::kRepeat) && element->IsaPointer()) {
               TStreamerElement *other = (TStreamerElement*) infoalloc->GetElements()->FindObject(element->GetName());
               if (other) {
                  other->SetBit(TStreamerElement::kDoNotDelete);
               }
            }
         }
         infoalloc->GetElements()->Compress();
      }
      {
         TIter next(fElements);
         TStreamerElement* element;
         while ((element = (TStreamerElement*) next())) {
            if (element->TestBit(TStreamerElement::kCache)) {
               element->SetOffset(infoalloc->GetOffset(element->GetName()));            
            }
         }
      }

      TStreamerElement *el = new TStreamerArtificial("@@alloc","", 0, TStreamerInfo::kCacheNew, allocClass->GetName());
      R__TObjArray_InsertAt( fElements, el, 0 );

      el = new TStreamerArtificial("@@dealloc","", 0, TStreamerInfo::kCacheDelete, allocClass->GetName());
      fElements->Add( el );
   }

   //
   // Make a more compact version.
   //
   Compile();
}

//______________________________________________________________________________
void TStreamerInfo::BuildCheck()
{
   // Check if built and consistent with the class dictionary.
   // This method is called by TFile::ReadStreamerInfo.

   R__LOCKGUARD(gCINTMutex);

   TObjArray* array = 0;
   fClass = TClass::GetClass(GetName());
   if (!fClass) {
      fClass = new TClass(GetName(), fClassVersion, 0, 0, -1, -1);
      fClass->SetBit(TClass::kIsEmulation);
      array = fClass->GetStreamerInfos();
   } else {
      if (TClassEdit::IsSTLCont(fClass->GetName())) {
         SetBit(kCanDelete);
         return;
      }
      array = fClass->GetStreamerInfos();
      TStreamerInfo* info = 0;

      if (fClass->TestBit(TClass::kIsEmulation) && array->GetEntries()==0) {
         // We have an emulated class that has no TStreamerInfo, this 
         // means it was created to insert a (default) rule.  Consequently
         // the error message about the missing dictionary was not printed.
         // For consistency, let's print it now!
         
         ::Warning("TClass::TClass", "no dictionary for class %s is available", GetName());
      }

      // If the user has not specified a class version (this _used to_
      // always be the case when the class is Foreign) or if the user
      // has specified a version to be explicitly 1. [We can not
      // distringuish the two cases using the information in the "on
      // file" StreamerInfo.]

      Bool_t searchOnChecksum = kFALSE;
      if (fClass->IsLoaded() && fClass->GetClassVersion() >= 2) {
         // We know for sure that the user specified the version.

         if (fOnFileClassVersion >= 2) {
            // The class version was specified when the object was
            // written

            searchOnChecksum = kFALSE;

         } else {
            // The class version was not specified when the object was
            // written OR it was specified to be 1.

            searchOnChecksum = kTRUE;            
         }
      } else if (fClass->IsLoaded() && !fClass->IsForeign()) {
         // We are in the case where the class has a Streamer function.
         // and fClass->GetClassVersion is 1, we still assume that the
         // Class Version is specified (to be one).

         searchOnChecksum = kFALSE;

      } else if (fClass->IsLoaded() /* implied: && fClass->IsForeign() */ ) {
         // We are in the case of a Foreign class with no specified
         // class version.

         searchOnChecksum = kTRUE;

      }
      else {
         // We are in the case of an 'emulated' class.

         if (fOnFileClassVersion >= 2) {
            // The class version was specified when the object was
            // written

            searchOnChecksum = kFALSE;

         } else {
            // The class version was not specified when the object was
            // written OR it was specified to be 1.

            searchOnChecksum = kTRUE;

            TStreamerInfo* v1 = (TStreamerInfo*) array->At(1);
            if (v1) {
               if (fCheckSum != v1->GetCheckSum()) {
                  fClassVersion = array->GetLast() + 1;
               }
            }
         }
      }

      if (!searchOnChecksum) {
         if (fClassVersion < array->GetEntriesFast()) {
            info = (TStreamerInfo*) array->At(fClassVersion);
         }
      } else {
         Int_t ninfos = array->GetEntriesFast() - 1;
         for (Int_t i = -1; i < ninfos; ++i) {
            info = (TStreamerInfo*) array->UncheckedAt(i);
            if (!info) {
               continue;
            }
            if (fCheckSum == info->GetCheckSum() && (info->GetOnFileClassVersion()==1 || info->GetOnFileClassVersion()==0)) {
               // We must match on the same checksum, an existing TStreamerInfo
               // for one of the 'unversioned' class layout (i.e. version was 1).
               fClassVersion = i;
               break;
            }
            info = 0;
         }
         if (info==0) {
            // Find an empty slot.
            ninfos = array->GetEntriesFast() - 1;
            Int_t slot = 1; // Start of Class version 1.
            while ((slot < ninfos) && (array->UncheckedAt(slot) != 0)) {
               ++slot;
            }
            fClassVersion = slot;
         }
      }

      // NOTE: Should we check if the already existing info is the same as
      // the current one? Yes
      // In case a class (eg Event.h) has a TClonesArray of Tracks, it could be
      // that the old info does not have the class name (Track) in the data
      // member title. Set old title to new title
      if (info) {
         // We found an existing TStreamerInfo for our ClassVersion
         Bool_t match = kTRUE;
         Bool_t done = kFALSE;
         Bool_t oldIsNonVersioned = kFALSE;
         if (fClassVersion!=0 && !fClass->TestBit(TClass::kWarned) && (fClassVersion == info->GetClassVersion()) && (fCheckSum != info->GetCheckSum())) {
            // The TStreamerInfo's checksum is different from the checksum for the compile class.

            match = kFALSE;
            oldIsNonVersioned = info->fOnFileClassVersion==1 && info->fClassVersion != 1;

            if (fClass->IsLoaded() && (fClassVersion == fClass->GetClassVersion()) && fClass->GetListOfDataMembers() && (fClass->GetClassInfo())) {
               // In the case where the read-in TStreamerInfo does not
               // match in the 'current' in memory TStreamerInfo for
               // a non foreign class (we can not get here if this is
               // a foreign class so we do not need to test it),
               // we need to add this one more test since the CINT behaviour
               // with enums changed over time, so verify the checksum ignoring
               // members of type enum. We also used to not count the //[xyz] comment
               // in the checksum, so test for that too.
               if (  (fCheckSum == fClass->GetCheckSum() || fCheckSum == fClass->GetCheckSum(1) || fCheckSum == fClass->GetCheckSum(2))
                     &&(info->GetCheckSum() == fClass->GetCheckSum() || info->GetCheckSum() == fClass->GetCheckSum(1) || info->GetCheckSum() == fClass->GetCheckSum(2))
                     )
                  {
                     match = kTRUE;
                  }
               if (fOldVersion <= 2) {
                  // Names of STL base classes was modified in vers==3. Allocators removed
                  // (We could be more specific (see test for the same case below)
                  match = kTRUE;
               }
               if (!match && CompareContent(0,info,kFALSE,kFALSE)) {
                  match = kTRUE;
               }
            } else {
               // The on-file TStreamerInfo's checksum differs from the checksum of a TStreamerInfo on another file.

               match = kFALSE;
               oldIsNonVersioned = info->fOnFileClassVersion==1 && info->fClassVersion != 1;

               // In the case where the read-in TStreamerInfo does not
               // match in the 'current' in memory TStreamerInfo for
               // a non foreign class (we can not get here if this is
               // a foreign class so we do not need to test it),
               // we need to add this one more test since the CINT behaviour
               // with enums changed over time, so verify the checksum ignoring
               // members of type enum. We also used to not count the //[xyz] comment
               // in the checksum, so test for that too.
               if (fCheckSum == info->GetCheckSum(0) || fCheckSum == info->GetCheckSum(1) || fCheckSum == info->GetCheckSum(2)
                   || GetCheckSum(0) == info->GetCheckSum() || GetCheckSum(1) == info->GetCheckSum() || GetCheckSum(2) == info->GetCheckSum() 
                   || GetCheckSum(0) == info->GetCheckSum(0))
                  {
                     match = kTRUE;
                  }
               if (fOldVersion <= 2) {
                  // Names of STL base classes was modified in vers==3. Allocators removed
                  // (We could be more specific (see test for the same case below)
                  match = kTRUE;
               }
               if (!match && CompareContent(0,info,kFALSE,kFALSE)) {
                  match = kTRUE;
               }
            }
         }
         if (info->IsBuilt()) {
            SetBit(kCanDelete);
            fNumber = info->GetNumber();
            Int_t nel = fElements->GetEntriesFast();
            TObjArray* elems = info->GetElements();
            TStreamerElement* e1 = 0;
            TStreamerElement* e2 = 0;
            for (Int_t i = 0; i < nel; ++i) {
               e1 = (TStreamerElement*) fElements->UncheckedAt(i);
               e2 = (TStreamerElement*) elems->At(i);
               if (!e1 || !e2) {
                  continue;
               }
               if (strlen(e1->GetTitle()) != strlen(e2->GetTitle())) {
                  e2->SetTitle(e1->GetTitle());
               }
            }
            
            if (TestBit(kCannotOptimize)) {
               info->SetBit(TVirtualStreamerInfo::kCannotOptimize);
               if (info->IsOptimized()) 
               {
                  // Optimizing does not work with splitting.
                  info->Compile();
               }
            } 
            done = kTRUE;
         } else {
            array->RemoveAt(fClassVersion);
            delete info;
            info = 0;
         }
         TString origin;
         if (!match && !fClass->TestBit(TClass::kWarned)) {
            if (oldIsNonVersioned) {
               if (gDirectory && gDirectory->GetFile()) {
                  Warning("BuildCheck", "\n\
   The class %s transitioned from not having a specified class version\n\
   to having a specified class version (the current class version is %d).\n\
   However too many different non-versioned layouts of the class have been\n\
   loaded so far.  This prevent the proper reading of objects written with\n\
   the class layout version %d, in particular from the file:\n\
   %s.\n\
   To work around this issue, load fewer 'old' files in the same ROOT session.",
                          GetName(),fClass->GetClassVersion(),fClassVersion,gDirectory->GetFile()->GetName());
               } else {
                  Warning("BuildCheck", "\n\
   The class %s transitioned from not having a specified class version\n\
   to having a specified class version (the current class version is %d).\n\
   However too many different non-versioned layouts of the class have been\n\
   loaded so far.  This prevent the proper reading of objects written with\n\
   the class layout version %d.\n\
   To work around this issue, load fewer 'old' files in the same ROOT session.",
                          GetName(),fClass->GetClassVersion(),fClassVersion);
               }
            } else {
               if (gDirectory && gDirectory->GetFile()) {
                  if (done) {
                     Warning("BuildCheck", "\n\
   The StreamerInfo for version %d of class %s read from the file %s\n\
   has a different checksum than the previously loaded StreamerInfo.\n\
   Reading objects of type %s from the file %s \n\
   (and potentially other files) might not work correctly.\n\
   Most likely the version number of the class was not properly\n\
   updated [See ClassDef(%s,%d)].", 
                             fClassVersion, GetName(), gDirectory->GetFile()->GetName(), GetName(), gDirectory->GetFile()->GetName(), GetName(), fClassVersion);
                  } else {
                     Warning("BuildCheck", "\n\
   The StreamerInfo from %s does not match existing one (%s:%d)\n\
   The existing one has not been used yet and will be discarded.\n\
   Reading the file %s will work properly, however writing object of\n\
   type %s will not work properly.  Most likely the version number\n\
   of the class was not properly updated [See ClassDef(%s,%d)].", 
                             gDirectory->GetFile()->GetName(), GetName(), fClassVersion,gDirectory->GetFile()->GetName(),GetName(), GetName(), fClassVersion);
                  }
               } else {
                  if (done) {
                     Warning("BuildCheck", "\n\
   The StreamerInfo for version %d of class %s\n\
   has a different checksum than the previously loaded StreamerInfo.\n\
   Reading objects of type %s\n\
   (and potentially other files) might not work correctly.\n\
   Most likely the version number of the class was not properly\n\
   updated [See ClassDef(%s,%d)].", 
                             fClassVersion, GetName(), GetName(), GetName(), fClassVersion);
                  } else {
                     Warning("BuildCheck", "\n\
   The StreamerInfo from %s does not match existing one (%s:%d)\n\
   The existing one has not been used yet and will be discarded.\n\
   Reading should work properly, however writing object of\n\
   type %s will not work properly.  Most likely the version number\n\
   of the class was not properly updated [See ClassDef(%s,%d)].", 
                             gDirectory->GetFile()->GetName(), GetName(), fClassVersion, GetName(), GetName(), fClassVersion);
                  }
               }
            }
            CompareContent(0,info,kTRUE,kTRUE);
            fClass->SetBit(TClass::kWarned);
         }
         if (done) {
            return;
         }
      }
      // The slot was free, however it might still be reserved for the current 
      // loaded version of the class
      if (fClass->IsLoaded() 
          && fClass->GetListOfDataMembers() 
          && (fClassVersion != 0) // We don't care about transient classes
          && (fClassVersion == fClass->GetClassVersion()) 
          && (fCheckSum != fClass->GetCheckSum()) 
          && (fClass->GetClassInfo())) {

         // If the old TStreamerInfo matches the in-memory one when we either
         //   - ignore the members of type enum
         // or
         //   - ignore the comments annotation (//[xyz])
         // we can accept the old TStreamerInfo.

         if (fCheckSum != fClass->GetCheckSum(1) && fCheckSum != fClass->GetCheckSum(2)) {

            Bool_t warn = !fClass->TestBit(TClass::kWarned);
            if (warn) {
               warn = !CompareContent(fClass,0,kFALSE,kFALSE);
            }
            if (warn && (fOldVersion <= 2)) {
               // Names of STL base classes was modified in vers==3. Allocators removed
               //
               TIter nextBC(fClass->GetListOfBases());
               TBaseClass* bc = 0;
               while ((bc = (TBaseClass*) nextBC())) {
                  if (TClassEdit::IsSTLCont(bc->GetName())) {
                     warn = kFALSE;
                  }
               }
            }
            if (warn) {
               if (gDirectory && gDirectory->GetFile()) {
                  Warning("BuildCheck", "\n\
   The StreamerInfo of class %s read from file %s\n\
   has the same version (=%d) as the active class but a different checksum.\n\
   You should update the version to ClassDef(%s,%d).\n\
   Do not try to write objects with the current class definition,\n\
   the files will not be readable.\n", GetName(), gDirectory->GetFile()->GetName(), fClassVersion, GetName(), fClassVersion + 1);
               } else {
                  Warning("BuildCheck", "\n\
   The StreamerInfo of class %s \n\
   has the same version (=%d) as the active class but a different checksum.\n\
   You should update the version to ClassDef(%s,%d).\n\
   Do not try to write objects with the current class definition,\n\
   the files will not be readable.\n", GetName(), fClassVersion, GetName(), fClassVersion + 1);
               }
               CompareContent(fClass,0,kTRUE,kTRUE);
               fClass->SetBit(TClass::kWarned);
            }
         } else {
            if (fClass->IsForeign()) {
               R__ASSERT(0);
            }
         }
      }
      if (!fClass->IsLoaded() &&  this->fOnFileClassVersion>1)
      {
         ROOT::ResetClassVersion(fClass,(const char*)-1, this->fClassVersion);
      }
   }
   // FIXME: This code can never execute because Build() calls
   // TStreamerElement::Class()->IgnoreTObjectStreamer()
   // so our bits are never saved to the file.
   if (TestBit(kIgnoreTObjectStreamer)) {
      fClass->IgnoreTObjectStreamer();
   }
   if ((fClassVersion < -1) || (fClassVersion > 65000)) {
      printf("ERROR reading TStreamerInfo: %s fClassVersion=%d\n", GetName(), fClassVersion);
      SetBit(kCanDelete);
      fNumber = -1;
      return;
   }

   array->AddAtAndExpand(this, fClassVersion);
   ++fgCount;
   fNumber = fgCount;

   // Since we just read this streamerInfo from file, it has already been built.
   fIsBuilt = kTRUE;

   //add to the global list of StreamerInfo
   TObjArray* infos = (TObjArray*) gROOT->GetListOfStreamerInfo();
   infos->AddAtAndExpand(this, fNumber);
}

//______________________________________________________________________________
void TStreamerInfo::BuildEmulated(TFile *file)
{
   // Create an Emulation TStreamerInfo object.

   R__LOCKGUARD(gCINTMutex);

   TString duName;
   R__ASSERT(file);
   Int_t fv = file->GetVersion()%100000;
   R__ASSERT(fv < 30000);
   fClassVersion = -1;
   fCheckSum = 2001;
   TObjArray *elements = GetElements();
   if (!elements) return;
   Int_t ndata = elements->GetEntries();
   if (ndata == 0) return;
   TStreamerElement *element;
   Int_t i;
   for (i=0;i < ndata;i++) {
      element = (TStreamerElement*)elements->UncheckedAt(i);
      if (!element) break;
      int ty = element->GetType();
      if (ty < kChar || ty >kULong+kOffsetL)    continue;
      if (ty == kLong)                         element->SetType(kInt);
      if (ty == kULong)                         element->SetType(kUInt);
      if (ty == kLong + kOffsetL)                element->SetType(kInt + kOffsetL);
      if (ty == kULong + kOffsetL)                element->SetType(kUInt + kOffsetL);
      if (ty <= kULong)                         continue;
      duName = element->GetName();
      duName.Append("QWERTY");
      TStreamerBasicType *bt = new TStreamerBasicType(duName, "", 0, kInt,"Int_t");
      {for (int j=ndata-1;j>=i;j--) {elements->AddAtAndExpand(elements->At(j),j+1);}}
      elements->AddAt(bt,i);
      ndata++;
      i++;
   }
   BuildOld();
}

//______________________________________________________________________________
Bool_t TStreamerInfo::BuildFor( const TClass *in_memory_cl )
{
   //---------------------------------------------------------------------------
   // Check if we can build this for foreign class - do we have some rules
   // to do that
   //---------------------------------------------------------------------------
   R__LOCKGUARD(gCINTMutex);

   if( !in_memory_cl || !in_memory_cl->GetSchemaRules() )
      return kFALSE;

   const TObjArray* rules;

   rules = in_memory_cl->GetSchemaRules()->FindRules( GetName(), fOnFileClassVersion, fCheckSum );

   if( !rules && !TClassEdit::IsSTLCont( in_memory_cl->GetName() ) ) {
      Warning( "BuildFor", "The build of %s streamer info for %s has been requested, but no matching conversion rules were specified", GetName(), in_memory_cl->GetName() );
      return kFALSE;
   }

   fClass = const_cast<TClass*>(in_memory_cl);

   return kTRUE;
}

//______________________________________________________________________________
// Helper function for BuildOld
namespace {
   Bool_t ClassWasMovedToNamespace(TClass *oldClass, TClass *newClass)
   {
      // Returns true if oldClass is the same as newClass but newClass is in a
      // namespace (and oldClass was not in a namespace).

      if (oldClass == 0 || newClass == 0) return kFALSE;

      UInt_t newlen = strlen(newClass->GetName());
      UInt_t oldlen = strlen(oldClass->GetName());

      const char *oldname = oldClass->GetName();
      for (UInt_t i = oldlen, done = false, nest = 0; (i>0) && !done ; --i) {
         switch (oldClass->GetName()[i-1]) {
            case '>' : ++nest; break;
            case '<' : --nest; break;
            case ':' : if (nest == 0) oldname= &(oldClass->GetName()[i]); done = kTRUE; break;
         }
      }
      oldlen = strlen(oldname);
      if (!(strlen(newClass->GetName()) > strlen(oldClass->GetName()))) {
         return kFALSE;
      }

      const char* newEnd = & (newClass->GetName()[newlen-oldlen]);

      if (0 != strcmp(newEnd, oldname)) {
         return kFALSE;
      }

      Int_t oldv = oldClass->GetStreamerInfo()->GetClassVersion();

      if (newClass->GetStreamerInfos() && oldv < newClass->GetStreamerInfos()->GetSize() && newClass->GetStreamerInfos()->At(oldv) && strcmp(newClass->GetStreamerInfos()->At(oldv)->GetName(), oldClass->GetName()) != 0) {
         // The new class has already a TStreamerInfo for the the same version as
         // the old class and this was not the result of an import.  So we do not
         // have a match
         return kFALSE;
      }
      return kTRUE;
   }

   Int_t ImportStreamerInfo(TClass *oldClass, TClass *newClass) {
      // Import the streamerInfo from oldClass to newClass
      // In case of conflict, returns the version number of the StreamerInfo
      // with the conflict.
      // Return 0 in case of success

      TIter next(oldClass->GetStreamerInfos());
      TStreamerInfo *info;
      while ((info = (TStreamerInfo*)next())) {
         info = (TStreamerInfo*)info->Clone();
         info->SetClass(newClass);
         Int_t oldv = info->GetClassVersion();
         if (oldv > newClass->GetStreamerInfos()->GetSize() || newClass->GetStreamerInfos()->At(oldv) == 0) {
            // All is good.
            newClass->GetStreamerInfos()->AddAtAndExpand(info,oldv);
         } else {
            // We verify that we are consistent and that
            //   newcl->GetStreamerInfos()->UncheckedAt(info->GetClassVersion)
            // is already the same as info.
            if (strcmp(newClass->GetStreamerInfos()->At(oldv)->GetName(),
                         oldClass->GetName()) != 0) {
               // The existing StreamerInfo does not already come from OldClass.
               // This is a real problem!
               return oldv;
            }
         }
      }
      return 0;
   }

   Bool_t ContainerMatchTClonesArray(TClass *newClass)
   {
      // Return true if newClass is a likely valid conversion from
      // a TClonesArray

      return newClass->GetCollectionProxy()
             && newClass->GetCollectionProxy()->GetValueClass()
             && !newClass->GetCollectionProxy()->HasPointers();
   }

   Bool_t CollectionMatch(const TClass *oldClass, const TClass* newClass)
   {
      // Return true if oldClass and newClass points to 2 compatible collection.
      // i.e. they contains the exact same type.

      TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
      TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();
      
      TClass *oldContent = oldProxy->GetValueClass();
      TClass *newContent = newProxy->GetValueClass();

      Bool_t contentMatch = kFALSE;
      if (oldContent) {
         if (oldContent == newContent) {
            contentMatch = kTRUE;
         } else if (newContent) {
            TString oldFlatContent( TMakeProject::UpdateAssociativeToVector(oldContent->GetName()) );
            TString newFlatContent( TMakeProject::UpdateAssociativeToVector(newContent->GetName()) );
            contentMatch = kTRUE;
         } else {
            contentMatch = kFALSE;
         }
      } else {
         contentMatch = (newContent==0);
      }

      if (contentMatch) {
         if ((oldContent==0 && oldProxy->GetType() == newProxy->GetType())
             ||(oldContent && oldProxy->HasPointers() == newProxy->HasPointers())) {
            // We have compatibles collections (they have the same content)!
            return kTRUE;
         }
      }
      return kFALSE;
   }

   Bool_t CollectionMatchFloat16(const TClass *oldClass, const TClass* newClass)
   {
      // Return true if oldClass and newClass points to 2 compatible collection.
      // i.e. they contains the exact same type.

      TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
      TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();

      if (oldProxy->GetValueClass() == 0 && newProxy->GetValueClass() == 0
          && (oldProxy->GetType() == kFloat_t || oldProxy->GetType() == kFloat16_t)
          && (newProxy->GetType() == kFloat_t || newProxy->GetType() == kFloat16_t )) {
            // We have compatibles collections (they have the same content)!
         return (TClassEdit::IsSTLCont(oldClass->GetName()) == TClassEdit::IsSTLCont(newClass->GetName()));
      }
      return kFALSE;
   }

   Bool_t CollectionMatchDouble32(const TClass *oldClass, const TClass* newClass)
   {
      // Return true if oldClass and newClass points to 2 compatible collection.
      // i.e. they contains the exact same type.

      TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
      TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();

      if (oldProxy->GetValueClass() == 0 && newProxy->GetValueClass() == 0
          && (oldProxy->GetType() == kDouble_t || oldProxy->GetType() == kDouble32_t)
          && (newProxy->GetType() == kDouble_t || newProxy->GetType() == kDouble32_t )) {
            // We have compatibles collections (they have the same content)!
         return (TClassEdit::IsSTLCont(oldClass->GetName()) == TClassEdit::IsSTLCont(newClass->GetName()));
      }
      return kFALSE;
   }

   Bool_t CollectionMatchLong64(const TClass *oldClass, const TClass* newClass)
   {
      // Return true if oldClass and newClass points to 2 compatible collection.
      // i.e. they contains the exact same type.

      TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
      TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();

      if (oldProxy->GetValueClass() == 0 && newProxy->GetValueClass() == 0
          && (oldProxy->GetType() == kLong_t || oldProxy->GetType() == kLong64_t)
          && (newProxy->GetType() == kLong_t || newProxy->GetType() == kLong64_t )) {
         // We have compatibles collections (they have the same content)!
         return (TClassEdit::IsSTLCont(oldClass->GetName()) == TClassEdit::IsSTLCont(newClass->GetName()));
      }
      return kFALSE;
   }

   Bool_t CollectionMatchULong64(const TClass *oldClass, const TClass* newClass)
   {
      // Return true if oldClass and newClass points to 2 compatible collection.
      // i.e. they contains the exact same type.

      TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
      TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();

      if (oldProxy->GetValueClass() == 0 && newProxy->GetValueClass() == 0
          && (oldProxy->GetType() == kULong_t || oldProxy->GetType() == kULong64_t)
          && (newProxy->GetType() == kULong_t || newProxy->GetType() == kULong64_t )) {
         // We have compatibles collections (they have the same content)!
         return (TClassEdit::IsSTLCont(oldClass->GetName()) == TClassEdit::IsSTLCont(newClass->GetName()));
      }
      return kFALSE;
   }
}

//______________________________________________________________________________
void TStreamerInfo::BuildOld()
{
   // rebuild the TStreamerInfo structure

   R__LOCKGUARD(gCINTMutex);

   if (gDebug > 0) {
      printf("\n====>Rebuilding TStreamerInfo for class: %s, version: %d\n", GetName(), fClassVersion);
   }

   Bool_t wasCompiled = IsCompiled();

   // This is used to avoid unwanted recursive call to Build
   fIsBuilt = kTRUE;

   if (fClass->GetClassVersion() == fClassVersion) {
      fClass->BuildRealData();
   }
   else {
      // This is to support the following case
      //  Shared library: Event v2
      //  calling cl->GetStreamerInfo(1)->BuildOld();  (or equivalent)
      //  which calls cl->BuildReadData()
      //  which set fRealData to some value
      //  then call Event()
      //  which call cl->GetStreamerInfo()
      //  which call cl->BuildRealData();
      //  which returns immediately (upon seeing fRealData!=0)
      //  then the main StreamerInfo build using the partial content of fRealData
      //  then BuildRealData returns
      //  then GetStreamerInfo() returns
      //  then Event() returns
      //  then fRealData is finished being populated
      //  then this function continue,
      //  then it uses the main streamerInfo
      //  .... which is incomplete.
      //
      //  Instead we force the creation of the main streamerInfo object
      //  before the creation of fRealData.
      fClass->GetStreamerInfo();
   }

   TIter next(fElements);
   TStreamerElement* element;
   Int_t offset = 0;
   TMemberStreamer* streamer = 0;

   Int_t sp = sizeof(void*);
#if defined(R__SGI64)
   sp = 8;
#endif

   int nBaze = 0;

   if (fClass->GetCollectionProxy() && (fElements->GetEntries() == 1) && !strcmp(fElements->At(0)->GetName(), "This")) {
      element = (TStreamerElement*)next();
      element->SetNewType( element->GetType() );
      element->SetNewClass( fClass );
   }

   TClass *allocClass = 0;
   TStreamerInfo *infoalloc = 0;

   //---------------------------------------------------------------------------
   // Get schema rules for this class
   //---------------------------------------------------------------------------
   const ROOT::TSchemaMatch*   rules   = 0;
   const ROOT::TSchemaRuleSet* ruleSet = fClass->GetSchemaRules();

   rules = (ruleSet ? ruleSet->FindRules( GetName(), fOnFileClassVersion, fCheckSum ) : 0);

   Bool_t shouldHaveInfoLoc = fClass->TestBit(TClass::kIsEmulation) && !TClassEdit::IsStdClass(fClass->GetName());
   Int_t virtualInfoLocAlloc = 0;
   fNVirtualInfoLoc = 0;
   delete [] fVirtualInfoLoc;
   fVirtualInfoLoc = 0;

   while ((element = (TStreamerElement*) next())) {
      if (element->IsA()==TStreamerArtificial::Class() 
          || element->TestBit(TStreamerElement::kCache) ) 
      {
         // Prevent BuildOld from modifying existing ArtificialElement (We need to review when and why BuildOld 
         // needs to be re-run; it might be needed if the 'current' class change (for example from being an onfile 
         // version to being a version loaded from a shared library) and we thus may have to remove the artifical 
         // element at the beginning of BuildOld)

         continue;
      };

      element->SetNewType(element->GetType());
      element->Init();
      if (element->IsBase()) {
         //---------------------------------------------------------------------
         // Dealing with nonSTL bases
         //---------------------------------------------------------------------
         if (element->IsA() == TStreamerBase::Class()) {
            TStreamerBase* base = (TStreamerBase*) element;
#if defined(PROPER_IMPLEMEMANTION_OF_BASE_CLASS_RENAMING)
            TClass* baseclass =  fClass->GetBaseClass( base->GetName() );
#else
            // Currently the base class renaming does not work, so we use the old
            // version of the code which essentially disable the next if(!baseclass ..
            // statement.
            TClass* baseclass =  base->GetClassPointer();
#endif

            //------------------------------------------------------------------
            // We do not have this base class - check if we're renaming
            //------------------------------------------------------------------
            if( !baseclass && !fClass->TestBit( TClass::kIsEmulation ) ) {
               const ROOT::TSchemaRule* rule = (rules ? rules->GetRuleWithSource( base->GetName() ) : 0);

               //---------------------------------------------------------------
               // No renaming, sorry
               //---------------------------------------------------------------
               if( !rule ) {
                  Error("BuildOld", "Could not find base class: %s for %s and could not find any matching rename rule\n", base->GetName(), GetName());
                  continue;
               }

               //----------------------------------------------------------------
               // Find a new target class
               //----------------------------------------------------------------
               const TObjArray* targets = rule->GetTarget();
               if( !targets ) {
                  Error("BuildOld", "Could not find base class: %s for %s, renaming rule was found but is malformed\n", base->GetName(), GetName());
               }
               TString newBaseClass = ((TObjString*)targets->At(0))->GetString();
               baseclass = TClass::GetClass( newBaseClass );
               base->SetNewBaseClass( baseclass );
            }
            //-------------------------------------------------------------------
            // No base class in emulated mode
            //-------------------------------------------------------------------
            else if( !baseclass ) {
               baseclass = base->GetClassPointer();
               if (!baseclass) {
                  Warning("BuildOld", "Missing base class: %s skipped", base->GetName());
                  // FIXME: Why is the version number 1 here? Answer: because we don't know any better at this point
                  baseclass = new TClass(element->GetName(), 1, 0, 0, -1, -1);
                  element->Update(0, baseclass);
               }
            }
            baseclass->BuildRealData();

            // Force the StreamerInfo "Compilation" of the base classes first. This is necessary in 
            // case the base class contains a member used as an array dimension in the derived classes.
            Int_t version = base->GetBaseVersion();
            TStreamerInfo* infobase = (TStreamerInfo*)baseclass->GetStreamerInfo(version);
            if (infobase->GetTypes() == 0) {
               infobase->BuildOld();
            }
            Int_t baseOffset = fClass->GetBaseClassOffset(baseclass);

            if (shouldHaveInfoLoc && baseclass->TestBit(TClass::kIsEmulation) ) {
               if ( (fNVirtualInfoLoc + infobase->fNVirtualInfoLoc) > virtualInfoLocAlloc ) {
                  ULong_t *store = fVirtualInfoLoc;
                  virtualInfoLocAlloc = 16 * ( (fNVirtualInfoLoc + infobase->fNVirtualInfoLoc) / 16 + 1);
                  fVirtualInfoLoc = new ULong_t[virtualInfoLocAlloc];
                  if (store) {
                     memcpy(fVirtualInfoLoc, store, sizeof(ULong_t)*fNVirtualInfoLoc);
                     delete [] store;
                  }
               }
               for (int nloc = 0; nloc < infobase->fNVirtualInfoLoc; ++nloc) {
                  fVirtualInfoLoc[ fNVirtualInfoLoc + nloc ] = baseOffset + infobase->fVirtualInfoLoc[nloc];
               }
               fNVirtualInfoLoc += infobase->fNVirtualInfoLoc;
            }
            // FIXME: Presumably we're in emulated mode, but is still does not make any sense
            // shouldn't it be element->SetNewType(-1) ?
            if (baseOffset < 0) {
               baseOffset = 0;
            }
            element->SetOffset(baseOffset);
            offset += baseclass->Size();

            continue;
         } else {
            // Not a base elem but still base, string or STL as a base
            nBaze++;
            TList* listOfBases = fClass->GetListOfBases();
            Int_t baseOffset = -1;
            Int_t asize = 0;
            if (listOfBases) {
               // Do a search for the classname and some of its alternatives spelling.

               TBaseClass* bc = 0;
               TIter nextBC(fClass->GetListOfBases());
               while ((bc = (TBaseClass*) nextBC())) {
                  if (strchr(bc->GetName(), '<') || !strcmp(bc->GetName(),"string")) {
                     TString bcName(TClassEdit::ShortType(bc->GetName(), TClassEdit::kDropStlDefault).c_str());
                     TString elName(TClassEdit::ShortType(element->GetTypeName(), TClassEdit::kDropStlDefault).c_str());
                     if (bcName == elName) {
                        break;
                     }
                  }
               }

               if (!bc) {
                  Error("BuildOld", "Could not find STL base class: %s for %s\n", element->GetName(), GetName());
                  continue;
               }
               baseOffset = bc->GetDelta();
               asize = bc->GetClassPointer()->Size();

            } else if (fClass->TestBit( TClass::kIsEmulation )) {
               // Do a search for the classname and some of its alternatives spelling.

               TStreamerInfo* newInfo = (TStreamerInfo*) fClass->GetStreamerInfos()->At(fClass->GetClassVersion());
               if (newInfo == this) {
                  baseOffset = offset;
                  asize = element->GetSize();
               } else if (newInfo) {
                  TIter newElems( newInfo->GetElements() );
                  TStreamerElement *newElement;
                  while( (newElement = (TStreamerElement*)newElems()) ) {
                     const char *newElName = newElement->GetName();
                     if (newElement->IsBase() && (strchr(newElName,'<') || !strcmp(newElName,"string")) ) {
                        TString bcName(TClassEdit::ShortType(newElName, TClassEdit::kDropStlDefault).c_str());
                        TString elName(TClassEdit::ShortType(element->GetTypeName(), TClassEdit::kDropStlDefault).c_str());
                        if (bcName == elName) {
                           break;
                        }
                     }
                  }
                  if (!newElement) {
                     Error("BuildOld", "Could not find STL base class: %s for %s\n", element->GetName(), GetName());
                     continue;
                  }
                  baseOffset = newElement->GetOffset();
                  asize = newElement->GetSize();
               }
            }
            if (baseOffset == -1) {
               TClass* cb = element->GetClassPointer();
               if (!cb) {
                  element->SetNewType(-1);
                  continue;
               }
               asize = cb->Size();
               baseOffset = fClass->GetBaseClassOffset(cb);
            }

            //  we know how to read but do we know where to read?
            if (baseOffset < 0) {
               element->SetNewType(-1);
               continue;
            }
            element->SetOffset(baseOffset);
            offset += asize;
            continue;
         }
      }

      // If we get here, this means that we looked at all the base classes.
      if (shouldHaveInfoLoc && fNVirtualInfoLoc==0) {
         fNVirtualInfoLoc = 1;
         fVirtualInfoLoc = new ULong_t[1]; // To allow for a single delete statement.
         fVirtualInfoLoc[0] = offset;
         offset += sizeof(TStreamerInfo*);
      }

      TDataMember* dm = 0;

      // First set the offset and sizes.
      if (fClass->GetDeclFileLine() < 0) {
         // Note the initilization in this case are
         // delayed until __after__ the schema evolution
         // section, just in case the info has changed.

         // We are in the emulated case
         streamer = 0;
         element->Init(this);
      } else {
         // The class is loaded.

         // First look for the data member in the current class
         dm = (TDataMember*) fClass->GetListOfDataMembers()->FindObject(element->GetName());
         if (dm && dm->IsPersistent()) {
            fClass->BuildRealData();
            streamer = 0;
            offset = GetDataMemberOffset(dm, streamer);
            element->SetOffset(offset);
            element->Init(this);
            element->SetStreamer(streamer);
            int narr = element->GetArrayLength();
            if (!narr) {
               narr = 1;
            }
            int dsize = dm->GetUnitSize();
            element->SetSize(dsize*narr);
         } else {
            // We did not find it, let's look for it in the base classes via TRealData
            TRealData* rd = fClass->GetRealData(element->GetName());
            if (rd && rd->GetDataMember()) {
               element->SetOffset(rd->GetThisOffset());
               element->Init(this);
               dm = rd->GetDataMember();
               int narr = element->GetArrayLength();
               if (!narr) {
                  narr = 1;
               }
               int dsize = dm->GetUnitSize();
               element->SetSize(dsize*narr);
            }
         }
      }

      // Now let's deal with Schema evolution
      Int_t newType = -1;
      TClassRef newClass;

      if (dm && dm->IsPersistent()) {
         if (dm->GetDataType()) {
            Bool_t isPointer = dm->IsaPointer();
            Bool_t isArray = element->GetArrayLength() >= 1;
            Bool_t hasCount = element->HasCounter();
            // data member is a basic type
            if ((fClass == TObject::Class()) && !strcmp(dm->GetName(), "fBits")) {
               //printf("found fBits, changing dtype from %d to 15\n", dtype);
               newType = kBits;
            } else {            
               // All the values of EDataType have the same semantic in EReadWrite 
               newType = (EReadWrite)dm->GetDataType()->GetType();
            }
            if ((newType == ::kChar_t) && isPointer && !isArray && !hasCount) {
               newType = ::kCharStar;
            } else if (isPointer) {
               newType += kOffsetP;
            } else if (isArray) {
               newType += kOffsetL;
            }
         }
         if (newType == -1) {
            newClass = TClass::GetClass(dm->GetTypeName());
         }
      } else {
         // Either the class is not loaded or the data member is gone
         if (!fClass->IsLoaded()) {
            TStreamerInfo* newInfo = (TStreamerInfo*) fClass->GetStreamerInfos()->At(fClass->GetClassVersion());
            if (newInfo && (newInfo != this)) {
               TStreamerElement* newElems = (TStreamerElement*) newInfo->GetElements()->FindObject(element->GetName());
               newClass = newElems ?  newElems->GetClassPointer() : 0;
               if (newClass == 0) {
                  newType = newElems ? newElems->GetType() : -1;
                  if (!(newType < kObject)) {
                     // sanity check.
                     newType = -1;
                  }
               }
            } else {
               newClass = element->GetClassPointer();
               if (newClass.GetClass() == 0) {
                  newType = element->GetType();
                  if (!(newType < kObject)) {
                     // sanity check.
                     newType = -1;
                  }
               }
            }
         }
      }

      if (newType > 0) {
         // Case of a numerical type
         if (element->GetType() != newType) {
            element->SetNewType(newType);
            if (gDebug > 0) {
               // coverity[mixed_enums] - All the values of EDataType have the same semantic in EReadWrite 
               Info("BuildOld", "element: %s %s::%s has new type: %s/%d", element->GetTypeName(), GetName(), element->GetName(), dm ? dm->GetFullTypeName() : TDataType::GetTypeName((EDataType)newType), newType);
            }
         }
      } else if (newClass.GetClass()) {
         // Sometime BuildOld is called again.
         // In that case we migth already have fix up the streamer element.
         // So we need to go back to the original information!
         newClass.Reset();
         TClass* oldClass = TClass::GetClass(TClassEdit::ShortType(element->GetTypeName(), TClassEdit::kDropTrailStar).c_str());
         if (oldClass == newClass.GetClass()) {
            // Nothing to do :)
         } else if (ClassWasMovedToNamespace(oldClass, newClass.GetClass())) {
            Int_t oldv;
            if (0 != (oldv = ImportStreamerInfo(oldClass, newClass.GetClass()))) {
                Warning("BuildOld", "Can not properly load the TStreamerInfo from %s into %s due to a conflict for the class version %d", oldClass->GetName(), newClass->GetName(), oldv);
            } else {
               element->SetTypeName(newClass->GetName());
               if (gDebug > 0) {
                  Warning("BuildOld", "element: %s::%s %s has new type %s", GetName(), element->GetTypeName(), element->GetName(), newClass->GetName());
               }
            }
         } else if (oldClass == TClonesArray::Class()) {
            if (ContainerMatchTClonesArray(newClass.GetClass())) {
               Int_t elemType = element->GetType();
               Bool_t isPrealloc = (elemType == kObjectp) || (elemType == kAnyp) || (elemType == (kObjectp + kOffsetL)) || (elemType == (kAnyp + kOffsetL));
               element->Update(oldClass, newClass.GetClass());
               TVirtualCollectionProxy *cp = newClass->GetCollectionProxy();
               TConvertClonesArrayToProxy *ms = new TConvertClonesArrayToProxy(cp, element->IsaPointer(), isPrealloc);
               element->SetStreamer(ms);

               // When the type is kObject, the TObject::Streamer is used instead
               // of the TStreamerElement's streamer.  So let force the usage
               // of our streamer
               if (element->GetType() == kObject) {
                  element->SetNewType(kAny);
                  element->SetType(kAny);
               }
               if (gDebug > 0) {
                  Warning("BuildOld","element: %s::%s %s has new type %s", GetName(), element->GetTypeName(), element->GetName(), newClass->GetName());
               }
            } else {
               element->SetNewType(-2);
            }
         } else if (oldClass  && oldClass->GetCollectionProxy() && newClass->GetCollectionProxy()) {
            if (CollectionMatch(oldClass, newClass)) {
               Int_t oldkind = TMath::Abs(TClassEdit::IsSTLCont( oldClass->GetName() ));
               Int_t newkind = TMath::Abs(TClassEdit::IsSTLCont( newClass->GetName() ));

               if ( (oldkind==TClassEdit::kMap || oldkind==TClassEdit::kMultiMap) &&
                    (newkind!=TClassEdit::kMap && newkind!=TClassEdit::kMultiMap) ) {

                  Int_t elemType = element->GetType();
                  Bool_t isPrealloc = (elemType == kObjectp) || (elemType == kAnyp) || (elemType == (kObjectp + kOffsetL)) || (elemType == (kAnyp + kOffsetL));

                  TClassStreamer *streamer2 = newClass->GetStreamer();
                  if (streamer2) {
                     TConvertMapToProxy *ms = new TConvertMapToProxy(streamer2, element->IsaPointer(), isPrealloc);
                     if (ms && ms->IsValid()) {
                        element->SetStreamer(ms);
                        switch( element->GetType() ) {
                           //case TStreamerInfo::kSTLvarp:           // Variable size array of STL containers.
                        case TStreamerInfo::kSTLp:                // Pointer to container with no virtual table (stl) and no comment
                        case TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL:     // array of pointers to container with no virtual table (stl) and no comment
                           element->SetNewType(-2);                       
                           break;
                        case TStreamerInfo::kSTL:             // container with no virtual table (stl) and no comment
                        case TStreamerInfo::kSTL + TStreamerInfo::kOffsetL:  // array of containers with no virtual table (stl) and no comment
                           break;
                        }
                     } else {
                        delete ms;
                     }
                  }
                  element->Update(oldClass, newClass.GetClass());

               } else if ( (newkind==TClassEdit::kMap || newkind==TClassEdit::kMultiMap) &&
                           (oldkind!=TClassEdit::kMap && oldkind!=TClassEdit::kMultiMap) ) {
                  element->SetNewType(-2); 
               } else {                                    
                  element->Update(oldClass, newClass.GetClass());
               }
               // Is this needed ? : element->SetSTLtype(newelement->GetSTLtype());
               if (gDebug > 0) {
                  Warning("BuildOld","element: %s::%s %s has new type %s", GetName(), element->GetTypeName(), element->GetName(), newClass->GetName());
               }
            } else if (CollectionMatchFloat16(oldClass,newClass)) {
               // Actually nothing to do, since both are the same collection of double in memory.
            } else if (CollectionMatchDouble32(oldClass,newClass)) {
               // Actually nothing to do, since both are the same collection of double in memory.              
            } else if (CollectionMatchLong64(oldClass,newClass)) {
               // Not much to do since both are the same collection of 8 bits entities on file.
               element->Update(oldClass, newClass.GetClass());
            } else if (CollectionMatchULong64(oldClass,newClass)) {
               // Not much to do since both are the same collection of 8 bits unsigned entities on file              
               element->Update(oldClass, newClass.GetClass());
            } else {
               element->SetNewType(-2);
            }

         } else if(oldClass && 
                   newClass.GetClass() && 
                   newClass->GetSchemaRules() && 
                   newClass->GetSchemaRules()->HasRuleWithSourceClass( oldClass->GetName() ) ) {
            //-----------------------------------------------------------------------
            // We can convert one type to another (at least for some of the versions.
            //-----------------------------------------------------------------------
            element->SetNewClass( newClass );               
         } else {
            element->SetNewType(-2);
         }
         // Humm we still need to make sure we have the same 'type' (pointer, embedded object, array, etc..)
         Bool_t cannotConvert = kFALSE;
         if (element->GetNewType() != -2) {
            if (dm) {
               if (dm->IsaPointer()) {
                  if (strncmp(dm->GetTitle(),"->",2)==0) {
                     // We are fine, nothing to do.
                     if (newClass->InheritsFrom(TObject::Class())) {
                        newType = kObjectp;
                     } else if (newClass->GetCollectionProxy()) {
                        newType = kSTLp;
                     } else {
                        newType = kAnyp;
                     }
                  } else {
                     if (TClass::GetClass(dm->GetTypeName())->InheritsFrom(TObject::Class())) {
                        newType = kObjectP;
                     } else if (newClass->GetCollectionProxy()) {
                        newType = kSTLp;
                     } else {
                        newType = kAnyP;
                     }
                  }
               } else {
                  if (newClass->GetCollectionProxy()) {
                     newType = kSTL;
                  } else if (newClass == TString::Class()) {
                     newType = kTString;
                  } else if (newClass == TObject::Class()) {
                     newType = kTObject;
                  } else if (newClass == TNamed::Class()) {
                     newType = kTNamed;
                  } else if (newClass->InheritsFrom(TObject::Class())) {
                     newType = kObject;
                  } else {
                     newType = kAny;                           
                  }
               }
               if ((!dm->IsaPointer() || newType==kSTLp) && dm->GetArrayDim() > 0) {
                  newType += kOffsetL;
               }               
            } else if (!fClass->IsLoaded()) {
               TStreamerInfo* newInfo = (TStreamerInfo*) fClass->GetStreamerInfos()->At(fClass->GetClassVersion());
               if (newInfo && (newInfo != this)) {
                  TStreamerElement* newElems = (TStreamerElement*) newInfo->GetElements()->FindObject(element->GetName());
                  if (newElems) {
                     newType = newElems->GetType();
                  }
               } else {
                  newType = element->GetType();
               }
            }
            if (element->GetType() == kSTL 
                || ((element->GetType() == kObject || element->GetType() == kAny || element->GetType() == kObjectp || element->GetType() == kAnyp) 
                    && oldClass == TClonesArray::Class())) 
            {
               cannotConvert = (newType != kSTL && newType != kObject && newType != kAny && newType != kSTLp && newType != kObjectp && newType != kAnyp);
               
            } else if (element->GetType() == kSTLp  || ((element->GetType() == kObjectP || element->GetType() == kAnyP) && oldClass == TClonesArray::Class()) ) 
            {
               cannotConvert = (newType != kSTL && newType != kObject && newType != kAny && newType != kSTLp && newType != kObjectP && newType != kAnyP);               

            } else if (element->GetType() == kSTL + kOffsetL
                || ((element->GetType() == kObject + kOffsetL|| element->GetType() == kAny + kOffsetL|| element->GetType() == kObjectp+ kOffsetL || element->GetType() == kAnyp+ kOffsetL) 
                    && oldClass == TClonesArray::Class())) 
            {
               cannotConvert = (newType != kSTL + kOffsetL && newType != kObject+ kOffsetL && newType != kAny+ kOffsetL && newType != kSTLp+ kOffsetL && newType != kObjectp+ kOffsetL && newType != kAnyp+ kOffsetL);
               
            } else if (element->GetType() == kSTLp + kOffsetL || ((element->GetType() == kObjectP+ kOffsetL || element->GetType() == kAnyP+ kOffsetL) && oldClass == TClonesArray::Class()) ) 
            {
               cannotConvert = (newType != kSTL+ kOffsetL && newType != kObject+ kOffsetL && newType != kAny+ kOffsetL && newType != kSTLp + kOffsetL&& newType != kObjectP+ kOffsetL && newType != kAnyP+ kOffsetL);               

            } else if ((element->GetType() == kObjectp || element->GetType() == kAnyp 
                 || element->GetType() == kObject || element->GetType() == kAny 
                 || element->GetType() == kTObject || element->GetType() == kTNamed || element->GetType() == kTString )) {
               // We had Type* ... ; //-> or Type ...;
               // this is completely compatible with the same and with a embedded object.
               if (newType != -1) {
                  if (newType == kObjectp || newType == kAnyp
                      || newType == kObject || newType == kAny
                      || newType == kTObject || newType == kTNamed || newType == kTString) {
                     // We are fine, no transformation to make
                     element->SetNewType(newType);
                  } else {
                     // We do not support this yet.
                     cannotConvert = kTRUE;
                  }
               } else {
                  // We have no clue 
                  cannotConvert = kTRUE;
               }
            } else if (element->GetType() == kObjectP || element->GetType() == kAnyP) {
               if (newType != -1) {
                  if (newType == kObjectP || newType == kAnyP ) {
                     // nothing to do}
                  } else {
                     cannotConvert = kTRUE;
                  }
               } else {
                  // We have no clue 
                  cannotConvert = kTRUE;
               }
            }
         }
         if (cannotConvert) {
            element->SetNewType(-2);
            if (gDebug > 0) {
               // coverity[mixed_enums] - All the values of EDataType have the same semantic in EReadWrite 
               Info("BuildOld", "element: %s %s::%s has new type: %s/%d", element->GetTypeName(), GetName(), element->GetName(), dm ? dm->GetFullTypeName() : TDataType::GetTypeName((EDataType)newType), newType);
            }
         }
      } else {
         element->SetNewType(-1);
         offset = kMissing;
         element->SetOffset(kMissing);
      }

      if (offset != kMissing && fClass->GetDeclFileLine() < 0) {
         // Note the initilization in this case are
         // delayed until __after__ the schema evolution
         // section, just in case the info has changed.
         Int_t asize = element->GetSize();
         // align the non-basic data types (required on alpha and IRIX!!)
         if ((offset % sp) != 0) {
            offset = offset - (offset % sp) + sp;
         }
         element->SetOffset(offset);
         offset += asize;
      }

      if ( !wasCompiled && rules && rules->HasRuleWithSource( element->GetName(), kTRUE ) ) {
         
         if (allocClass == 0) {
            infoalloc  = (TStreamerInfo *)Clone(TString::Format("%s@@%d",GetName(),GetOnFileClassVersion()));
            infoalloc->BuildCheck();
            infoalloc->BuildOld();
            allocClass = infoalloc->GetClass();
         }

         // Now that we are caching the unconverted element, we do not assign it to the real type even if we could have!
         if (element->GetNewType()>0 /* intentionally not including base class for now */ 
             && !rules->HasRuleWithTarget( element->GetName(), kTRUE ) ) 
         {
            TStreamerElement *copy = (TStreamerElement*)element->Clone();
            R__TObjArray_InsertBefore( fElements, copy, element );
            next(); // move the cursor passed the insert object.
            copy->SetBit(TStreamerElement::kRepeat);
            element = copy;

            // Warning("BuildOld","%s::%s is not set from the version %d of %s (You must add a rule for it)\n",GetName(), element->GetName(), GetClassVersion(), GetName() );
         }
         element->SetBit(TStreamerElement::kCache);
         element->SetNewType( element->GetType() );
         element->SetOffset(infoalloc->GetOffset(element->GetName()));
      }

      if (element->GetNewType() == -2) {
         Warning("BuildOld", "Cannot convert %s::%s from type:%s to type:%s, skip element", GetName(), element->GetName(), element->GetTypeName(), newClass->GetName());
      }
   }

   // If we get here, this means that there no data member after the last base class
   // (or no base class at all).
   if (shouldHaveInfoLoc && fNVirtualInfoLoc==0) {
      fNVirtualInfoLoc = 1;
      fVirtualInfoLoc = new ULong_t[1]; // To allow for a single delete statement.
      fVirtualInfoLoc[0] = offset;
      offset += sizeof(TStreamerInfo*);
   }

   // change order , move "bazes" to the end. Workaround old bug
   if ((fOldVersion <= 2) && nBaze) {
      SetBit(kRecovered);
      TObjArray& arr = *fElements;
      TObjArray tai(nBaze);
      int narr = arr.GetLast() + 1;
      int iel;
      int jel = 0;
      int kel = 0;
      for (iel = 0; iel < narr; ++iel) {
         element = (TStreamerElement*) arr[iel];
         if (element->IsBase() && (element->IsA() != TStreamerBase::Class())) {
            tai[kel++] = element;
         } else {
            arr[jel++] = element;
         }
      }
      for (kel = 0; jel < narr;) {
         arr[jel++] = tai[kel++];
      }
   }

   // Now add artificial TStreamerElement (i.e. rules that creates new members or set transient members).
   if (!wasCompiled) InsertArtificialElements(rules);

   if (!wasCompiled && allocClass) {

      TStreamerElement *el = new TStreamerArtificial("@@alloc","", 0, TStreamerInfo::kCacheNew, allocClass->GetName());
      R__TObjArray_InsertAt( fElements, el, 0 );

      el = new TStreamerArtificial("@@dealloc","", 0, TStreamerInfo::kCacheDelete, allocClass->GetName());
      fElements->Add( el );
   }

   Compile();
   delete rules;
}

//______________________________________________________________________________
void TStreamerInfo::Clear(Option_t *option)
{
   // If opt cointains 'built', reset this StreamerInfo as if Build or BuildOld
   // was never called on it (usefull to force their re-running).

   TString opt = option;
   opt.ToLower();

   if (opt.Contains("build")) {
      delete [] fType;     fType    = 0;
      delete [] fNewType;  fNewType = 0;
      delete [] fOffset;   fOffset  = 0;
      delete [] fLength;   fLength  = 0;
      delete [] fElem;     fElem    = 0;
      delete [] fMethod;   fMethod  = 0;
      delete [] fComp;     fComp    = 0;
      fNdata = 0;
      fSize = 0;
      ResetBit(kIsCompiled);
      
      if (fReadObjectWise) fReadObjectWise->fActions.clear();
      if (fReadMemberWise) fReadMemberWise->fActions.clear();
   }
}

namespace {
   // TMemberInfo
   // Local helper class to be able to compare data member represened by
   // 2 distinct TStreamerInfos
   class TMemberInfo {
   public:
      TString fName;
      TString fClassName;
      TString fComment;
      void SetName(const char *name) {
         fName = name;
      }
      void SetClassName(const char *name) {
         fClassName = TClassEdit::ShortType( name, TClassEdit::kDropStlDefault | TClassEdit::kDropStd );
      }
      void SetComment(const char *title) {
         const char *left = strstr(title,"[");
         if (left) {
            const char *right = strstr(left,"]");
            if (right) {
               ++left;
               fComment.Append(left,right-left);
            }
         }
      }
      void Clear() {
         fName.Clear();
         fClassName.Clear();
         fComment.Clear();
      }
      /* Hide this not yet used implementation to suppress warnings message
       from icc 11 
      Bool_t operator==(const TMemberInfo &other) {
         return fName==other.fName
            && fClassName == other.fClassName
            && fComment == other.fComment;
      }
       */
      Bool_t operator!=(const TMemberInfo &other) {
         if (fName != other.fName) return kTRUE;
         if (fClassName != other.fClassName) {
            if ( (fClassName == "long" && (other.fClassName == "long long" || other.fClassName == "Long64_t"))
                  || ( (fClassName == "long long" || fClassName == "Long64_t") && other.fClassName == "long") ) {
               // This is okay both have the same on file format.
            } else if ( (fClassName == "unsigned long" && (other.fClassName == "unsigned long long" || other.fClassName == "ULong64_t"))
                       || ( (fClassName == "unsigned long long" || fClassName == "ULong64_t") && other.fClassName == "unsigned long") ) {
               // This is okay both have the same on file format.
            } else {
               return kTRUE;
            }
         }
         return fComment != other.fComment;
      }
   };
}

//______________________________________________________________________________
void TStreamerInfo::CallShowMembers(void* obj, TMemberInspector &insp) const
{
   // Emulated a call ShowMembers() on the obj of this class type, passing insp and parent.

   TIter next(fElements);
   TStreamerElement* element = (TStreamerElement*) next();

   TString elementName;

   for (; element; element = (TStreamerElement*) next()) {

      // Skip elements which have not been allocated memory.
      if (element->GetOffset() == kMissing) {
         continue;
      }

      char* eaddr = ((char*)obj) + element->GetOffset();

      if (element->IsBase()) {
         // Nothing to do this round.
      } else if (element->IsaPointer()) {
         elementName.Form("*%s",element->GetFullName());
         insp.Inspect(fClass, insp.GetParent(), elementName.Data(), eaddr);
      } else {
         insp.Inspect(fClass, insp.GetParent(), element->GetFullName(), eaddr);         
         Int_t etype = element->GetType();
         switch(etype) {
            case kObject:
            case kAny:
            case kTObject:
            case kTString:
            case kTNamed:
            case kSTL:
            {
               TClass *ecl = element->GetClassPointer();
               if (ecl && (fClass!=ecl /* This happens 'artificially for stl container see the use of "This" */)) { 
                  insp.InspectMember(ecl, eaddr, TString(element->GetName()) + ".");
               }
               break;
            }
         } // switch(etype)
      } // if IsaPointer()
   } // Loop over elements

   // And now do the base classes
   next.Reset();
   element = (TStreamerElement*) next();
   for (; element; element = (TStreamerElement*) next()) {
      if (element->IsBase()) {
         // Skip elements which have not been allocated memory.
         if (element->GetOffset() == kMissing) {
            continue;
         }

         char* eaddr = ((char*)obj) + element->GetOffset();

         TClass *ecl = element->GetClassPointer();
         if (ecl) {
            ecl->CallShowMembers(eaddr, insp);
         }
      } // If is a abse
   } // Loop over elements
}

//______________________________________________________________________________
TObject *TStreamerInfo::Clone(const char *newname) const
{
   // Make a clone of an object using the Streamer facility.
   // If newname is specified, this will be the name of the new object.

   TStreamerInfo *newinfo = (TStreamerInfo*)TNamed::Clone(newname);
   if (newname && newname[0] && fName != newname) {
      TObjArray *newelems = newinfo->GetElements(); 
      Int_t ndata = newelems->GetEntries();
      for(Int_t i = 0; i < ndata; ++i) {
         TObject *element = newelems->UncheckedAt(i);
         if (element->IsA() == TStreamerLoop::Class()) {
            TStreamerLoop *eloop = (TStreamerLoop*)element;
            if (fName == eloop->GetCountClass()) {
               eloop->SetCountClass(newname);
               eloop->Init();
            }
         } else if (element->IsA() == TStreamerBasicPointer::Class()) {
            TStreamerBasicPointer *eptr = (TStreamerBasicPointer*)element;
            if (fName == eptr->GetCountClass()) {
               eptr->SetCountClass(newname);
               eptr->Init();
            }
         }
      }
   }
   return newinfo;
}

//______________________________________________________________________________
Bool_t TStreamerInfo::CompareContent(TClass *cl, TVirtualStreamerInfo *info, Bool_t warn, Bool_t complete)
{
   // Return True if the current StreamerInfo in cl or info is equivalent to this TStreamerInfo.
   // 'Equivalent' means the same number of persistent data member which the same actual C++ type and
   // the same name.
   // if 'warn' is true, Warning message are printed to explicit the differences.
   // if 'complete' is false, stop at the first error, otherwise continue until all members have been checked.

   Bool_t result = kTRUE;
   R__ASSERT( (cl==0 || info==0) && (cl!=0 || info!=0) /* must compare to only one thhing! */);

   TString name;
   TString type;
   TStreamerElement *el;
   TStreamerElement *infoel;

   TIter next(GetElements());
   TIter infonext((TList*)0);
   TIter basenext((TList*)0);
   TIter membernext((TList*)0);
   if (info) {
      infonext = info->GetElements();
   }
   if (cl) {
      TList *tlb = cl->GetListOfBases();
      if (tlb) {   // Loop over bases
         basenext = tlb;
      }
      tlb = cl->GetListOfDataMembers();
      if (tlb) {
         membernext = tlb;
      }
   }

   // First let's compare base classes
   Bool_t done = kFALSE;
   TString localClass;
   TString otherClass;
   while(!done) {
      localClass.Clear();
      otherClass.Clear();
      el = (TStreamerElement*)next();
      if (el && el->IsBase()) {
         localClass = el->GetName();
      } else {
         el = 0;
      }
      if (cl) {
         TBaseClass *tbc = (TBaseClass*)basenext();
         if (tbc) {
            otherClass = tbc->GetName();
         } else if (el==0) {
            done = kTRUE;
            break;
         }
      } else {
         infoel = (TStreamerElement*)infonext();
         if (infoel && infoel->IsBase()) {
            otherClass = infoel->GetName();
         } else if (el==0) {
            done = kTRUE;
            break;
         }
      }
      // Need to normalized the name
      if (localClass != otherClass) {
         if (warn) {
            if (el==0) {
               Warning("CompareContent",
                       "The in-memory layout version %d for class '%s' has a base class (%s) that the on-file layout version %d does not have.",
                       GetClassVersion(), GetName(), otherClass.Data(), GetClassVersion());
            } else if (otherClass.Length()==0) {
               Warning("CompareContent",
                       "The on-file layout version %d for class '%s'  has a base class (%s) that the in-memory layout version %d does not have",
                       GetClassVersion(), GetName(), localClass.Data(), GetClassVersion());
            } else {
               Warning("CompareContent",
                       "One base class of the on-file layout version %d and of the in memory layout version %d for '%s' is different: '%s' vs '%s'",
                       GetClassVersion(), GetClassVersion(), GetName(), localClass.Data(), otherClass.Data());
            }
         }
         if (!complete) return kFALSE;
         result = result && kFALSE;
      }
   }
   if (!result && !complete) {
      return result;
   }
   // Next the datamembers
   done = kFALSE;
   next.Reset();
   infonext.Reset();

   TMemberInfo local;
   TMemberInfo other;
   UInt_t idx = 0;
   while(!done) {
      local.Clear();
      other.Clear();
      el = (TStreamerElement*)next();
      while (el && (el->IsBase() || el->IsA() == TStreamerArtificial::Class())) {
         el = (TStreamerElement*)next();
         ++idx;
      }
      if (el) {
         local.SetName( el->GetName() );
         local.SetClassName( el->GetTypeName() );
         local.SetComment( el->GetTitle() );
      }
      if (cl) {
         TDataMember *tdm = (TDataMember*)membernext();
         while(tdm && ( (!tdm->IsPersistent()) || (tdm->Property()&kIsStatic) || (el && local.fName != tdm->GetName()) )) {
            tdm = (TDataMember*)membernext();
         }
         if (tdm) {
            other.SetName( tdm->GetName() );
            other.SetClassName( tdm->GetFullTypeName() );
            other.SetComment( tdm->GetTitle() );
         } else if (el==0) {
            done = kTRUE;
            break;
         }
      } else {
         infoel = (TStreamerElement*)infonext();
         while (infoel && (infoel->IsBase() || infoel->IsA() == TStreamerArtificial::Class())) {
            infoel = (TStreamerElement*)infonext();
         }
         if (infoel) {
            other.SetName( infoel->GetName() );
            other.SetClassName( infoel->GetTypeName() );
            other.SetComment( infoel->GetTitle() );
         } else if (el==0) {
            done = kTRUE;
            break;
         }
      }
      if (local!=other) {
         if (warn) {
            if (!el) {
               Warning("CompareContent","The following data member of\nthe on-file layout version %d of class '%s' is missing from \nthe in-memory layout version %d:\n"
                       "   %s %s; //%s"
                       ,GetClassVersion(), GetName(), GetClassVersion()
                       ,other.fClassName.Data(),other.fName.Data(),other.fComment.Data());

            } else if (other.fName.Length()==0) {
               Warning("CompareContent","The following data member of\nthe in-memory layout version %d of class '%s' is missing from \nthe on-file layout version %d:\n"
                       "   %s %s; //%s"
                       ,GetClassVersion(), GetName(), GetClassVersion()
                       ,local.fClassName.Data(),local.fName.Data(),local.fComment.Data());
            } else {
               Warning("CompareContent","The following data member of\nthe on-file layout version %d of class '%s' differs from \nthe in-memory layout version %d:\n"
                       "   %s %s; //%s\n"
                       "vs\n"
                       "   %s %s; //%s"
                       ,GetClassVersion(), GetName(), GetClassVersion()
                       ,local.fClassName.Data(),local.fName.Data(),local.fComment.Data()
                       ,other.fClassName.Data(),other.fName.Data(),other.fComment.Data());
            }
         }
         result = result && kFALSE;
         if (!complete) return result;
      }
      ++idx;
   }
   return result;
}


//______________________________________________________________________________
void TStreamerInfo::ComputeSize()
{
   // Compute total size of all persistent elements of the class

   TStreamerElement *element = (TStreamerElement*)fElements->Last();
   //faster and more precise to use last element offset +size
   //on 64 bit machines, offset may be forced to be a multiple of 8 bytes
   fSize = element->GetOffset() + element->GetSize();
   if (fNVirtualInfoLoc > 0 && (fVirtualInfoLoc[0]+sizeof(TStreamerInfo*)) >= (ULong_t)fSize) {
      fSize = fVirtualInfoLoc[0] + sizeof(TStreamerInfo*);
   }
}

//______________________________________________________________________________
void TStreamerInfo::ForceWriteInfo(TFile* file, Bool_t force)
{
   // -- Recursively mark streamer infos for writing to a file.
   //
   // Will force this TStreamerInfo to the file and also
   // all the dependencies.
   //
   // If argument force > 0 the loop on class dependencies is forced.
   //
   // This function is called when streaming a class that contains
   // a null pointer. In this case, the TStreamerInfo for the class
   // with the null pointer must be written to the file and also all
   // the TStreamerInfo of all the classes referenced by the class.
   //
   //--
   // We must be given a file to write to.
   if (!file) {
      return;
   }
   // Get the given file's list of streamer infos marked for writing.
   TArrayC* cindex = file->GetClassIndex();
   //the test below testing fArray[fNumber]>1 is to avoid a recursivity
   //problem in some cases like:
   //        class aProblemChild: public TNamed {
   //        aProblemChild *canBeNull;
   //        };
   if ( // -- Done if already marked, and we are not forcing, or forcing is blocked.
      (cindex->fArray[fNumber] && !force) || // Streamer info is already marked, and not forcing, or
      (cindex->fArray[fNumber] > 1) // == 2 means ignore forcing to prevent infinite recursion.
   ) {
      return;
   }
   // We do not want to write streamer info to the file
   // for std::string.
   static TClassRef string_classref("string");
   if (fClass == string_classref) { // We are std::string.
      return;
   }
   // We do not want to write streamer info to the file
   // for STL containers.
   if (fClass==0) {
      // Build or BuildCheck has not been called yet.
      // Let's use another means of checking.
      if (fElements && fElements->GetEntries()==1 && strcmp("This",fElements->UncheckedAt(0)->GetName())==0) {
         // We are an STL collection.
         return;
      }
   } else if (fClass->GetCollectionProxy()) { // We are an STL collection.
      return;
   }
   // Mark ourselves for output, and block
   // forcing to prevent infinite recursion.
   cindex->fArray[fNumber] = 2;
   // Signal the file that the marked streamer info list has changed.
   cindex->fArray[0] = 1;
   // Recursively mark the streamer infos for
   // all of our elements.
   TIter next(fElements);
   TStreamerElement* element = (TStreamerElement*) next();
   for (; element; element = (TStreamerElement*) next()) {
      TClass* cl = element->GetClassPointer();
      if (cl) {
         TVirtualStreamerInfo* si = 0;
         if (cl->Property() & kIsAbstract) {
            // If the class of the element is abstract, register the
            // TStreamerInfo only if it has already been built.
            // Otherwise call cl->GetStreamerInfo() would generate an
            // incorrect StreamerInfo.
            si = cl->GetCurrentStreamerInfo();
         } else {
            si = cl->GetStreamerInfo();
         }
         if (si) {
            si->ForceWriteInfo(file, force);
         }
      }
   }
}

//______________________________________________________________________________
TClass *TStreamerInfo::GetActualClass(const void *obj) const 
{
   // Assuming that obj points to (the part of) an object that is of the
   // type described by this streamerInfo, return the actual type of the
   // object (i.e. the type described by this streamerInfo is a base class
   // of the actual type of the object.
   // This routine should only be called if the class decribed by this
   // StreamerInfo is 'emulated'.
   
   R__ASSERT(!fClass->IsLoaded());
   
   if (fNVirtualInfoLoc != 0) {
      TStreamerInfo *allocator = *(TStreamerInfo**)( (const char*)obj + fVirtualInfoLoc[0] );
      if (allocator) return allocator->GetClass();
   }
   return (TClass*)fClass;
}
   
//______________________________________________________________________________
UInt_t TStreamerInfo::GetCheckSum(UInt_t code) const 
{
   // Recalculate the checksum of this TStreamerInfo based on its code.
   // 
   // The class ckecksum is used by the automatic schema evolution algorithm
   // to uniquely identify a class version.
   // The check sum is built from the names/types of base classes and
   // data members.
   // Algorithm from Victor Perevovchikov (perev@bnl.gov).
   //
   // if code==1 data members of type enum are not counted in the checksum
   // if code==2 return the checksum of data members and base classes, not including the ranges and array size found in comments.  
   //            This is needed for backward compatibility.
   //
   // WARNING: this function must be kept in sync with TClass::GetCheckSum.
   // They are both used to handle backward compatibility and should both return the same values.
   // TStreamerInfo uses the information in TStreamerElement while TClass uses the information
   // from TClass::GetListOfBases and TClass::GetListOfDataMembers.

   UInt_t id = 0;

   int il;
   TString name = GetName();
   TString type;
   il = name.Length();
   for (int i=0; i<il; i++) id = id*3+name[i];

   TIter next(GetElements());
   TStreamerElement *el;
   while ( (el=(TStreamerElement*)next()) ) {
      if (el->IsBase()) {
         name = el->GetName();
         il = name.Length();
         for (int i=0; i<il; i++) id = id*3+name[i];
      }
   } /* End of Base Loop */

   next.Reset();
   while ( (el=(TStreamerElement*)next()) ) {
      if (el->IsBase()) continue;

      // humm can we tell if a TStreamerElement is an enum?
      // Maybe something like:
      Bool_t isenum = kFALSE;
      if ( el->GetType()==3 && gROOT->GetType(el->GetTypeName())==0) {
         // If the type is not an enum but a typedef to int then 
         // el->GetTypeName() should be return 'int'
         isenum = kTRUE;
      }
      if ( (code != 1) && isenum) id = id*3 + 1;

      name = el->GetName();  il = name.Length();

      int i;
      for (i=0; i<il; i++) id = id*3+name[i];

      type = el->GetTypeName();
      if (TClassEdit::IsSTLCont(type)) {
         type = TClassEdit::ShortType( type, TClassEdit::kDropStlDefault | TClassEdit::kLong64 );
      }

      il = type.Length();
      for (i=0; i<il; i++) id = id*3+type[i];

      int dim = el->GetArrayDim();
      if (dim) {
         for (i=0;i<dim;i++) id = id*3+el->GetMaxIndex(i);
      }


      if (code != 2) {
         const char *left = strstr(el->GetTitle(),"[");
         if (left) {
            const char *right = strstr(left,"]");
            if (right) {
               ++left;
               while (left != right) {
                  id = id*3 + *left;
                  ++left;
               }
            }
         }
      }
   }
   return id;
}

//______________________________________________________________________________
static void R__WriteConstructorBody(FILE *file, TIter &next)
{
   TStreamerElement *element = 0;
   next.Reset();
   while ((element = (TStreamerElement*)next())) {
      if (element->GetType() == TVirtualStreamerInfo::kObjectp || element->GetType() == TVirtualStreamerInfo::kObjectP ||
          element->GetType() == TVirtualStreamerInfo::kAnyp || element->GetType() == TVirtualStreamerInfo::kAnyP || 
          element->GetType() == TVirtualStreamerInfo::kCharStar || element->GetType() == TVirtualStreamerInfo::kSTLp || 
          element->GetType() == TVirtualStreamerInfo::kStreamLoop) {
         if(element->GetArrayLength() <= 1) {
            fprintf(file,"   %s = 0;\n",element->GetName());
         } else {
            fprintf(file,"   memset(%s,0,%d);\n",element->GetName(),element->GetSize());
         }
      }
      if (TVirtualStreamerInfo::kOffsetP <= element->GetType() && element->GetType() < TVirtualStreamerInfo::kObject ) {
         fprintf(file,"   %s = 0;\n",element->GetName());
      }
   }
}

//______________________________________________________________________________
static void R__WriteMoveConstructorBody(FILE *file, const TString &protoname, TIter &next)
{
   // Write down the body of the 'move' constructor.

   TStreamerElement *element = 0;
   next.Reset();
   Bool_t atstart = kTRUE;
   while ((element = (TStreamerElement*)next())) {
      if (element->IsBase()) {
         if (atstart) { fprintf(file,"   : "); atstart = kFALSE; }
         else fprintf(file,"   , ");
         fprintf(file, "%s(const_cast<%s &>( rhs ))\n", element->GetName(),protoname.Data());
      } else {
         if (element->GetArrayLength() <= 1) {
            if (atstart) { fprintf(file,"   : "); atstart = kFALSE; }
            else fprintf(file,"   , ");
            fprintf(file, "%s(const_cast<%s &>( rhs ).%s)\n",element->GetName(),protoname.Data(),element->GetName());
         }
      }
   }
   fprintf(file,"{\n");
   fprintf(file,"   // This is NOT a copy constructor. This is actually a move constructor (for stl container's sake).\n");
   fprintf(file,"   // Use at your own risk!\n");
   fprintf(file,"   if (&rhs) {} // avoid warning about unused parameter\n");
   next.Reset();
   Bool_t defMod = kFALSE;
   while ((element = (TStreamerElement*)next())) {
      if (element->GetType() == TVirtualStreamerInfo::kObjectp || element->GetType() == TVirtualStreamerInfo::kObjectP||
          element->GetType() == TVirtualStreamerInfo::kAnyp || element->GetType() == TVirtualStreamerInfo::kAnyP
          || element->GetType() == TVirtualStreamerInfo::kAnyPnoVT) 
      {
         if (!defMod) { fprintf(file,"   %s &modrhs = const_cast<%s &>( rhs );\n",protoname.Data(),protoname.Data()); defMod = kTRUE; };
         const char *ename = element->GetName();
         const char *colon2 = strstr(ename,"::");
         if (colon2) ename = colon2+2;
         if(element->GetArrayLength() <= 1) {
            fprintf(file,"   modrhs.%s = 0;\n",ename);
         } else {
            fprintf(file,"   memset(modrhs.%s,0,%d);\n",ename,element->GetSize());
         }
      } else {
         const char *ename = element->GetName();
         if (element->GetType() == kCharStar) {
            if (!defMod) { 
               fprintf(file,"   %s &modrhs = const_cast<%s &>( rhs );\n",protoname.Data(),protoname.Data()); defMod = kTRUE; 
            };
            fprintf(file,"   modrhs.%s = 0;\n",ename);
         } else if (TVirtualStreamerInfo::kOffsetP <= element->GetType() && element->GetType() < TVirtualStreamerInfo::kObject ) { 
            if (!defMod) { 
               fprintf(file,"   %s &modrhs = const_cast<%s &>( rhs );\n",protoname.Data(),protoname.Data()); defMod = kTRUE; 
            };
            fprintf(file,"   modrhs.%s = 0;\n",ename);
         } else if (element->GetArrayLength() > 1) {
            // FIXME: Need to add support for variable length array.
            if (element->GetArrayDim() == 1) {
               fprintf(file,"   for (Int_t i=0;i<%d;i++) %s[i] = rhs.%s[i];\n",element->GetArrayLength(),ename,ename);            
            } else if (element->GetArrayDim() >= 2) {
               fprintf(file,"   for (Int_t i=0;i<%d;i++) (&(%s",element->GetArrayLength(),ename);
               for (Int_t d = 0; d < element->GetArrayDim(); ++d) {
                  fprintf(file,"[0]");
               }
               fprintf(file,"))[i] = (&(rhs.%s",ename);
               for (Int_t d = 0; d < element->GetArrayDim(); ++d) {
                  fprintf(file,"[0]");
               }
               fprintf(file,"))[i];\n");
            }
         } else if (element->GetType() == TVirtualStreamerInfo::kSTLp) {
            if (!defMod) { fprintf(file,"   %s &modrhs = const_cast<%s &>( rhs );\n",protoname.Data(),protoname.Data()); defMod = kTRUE; };
            fprintf(file,"   modrhs.%s = 0;\n",ename);
         } else if (element->GetType() == TVirtualStreamerInfo::kSTL) {
            if (!defMod) { 
               fprintf(file,"   %s &modrhs = const_cast<%s &>( rhs );\n",protoname.Data(),protoname.Data()); defMod = kTRUE; 
            }
            if (element->IsBase()) {
               fprintf(file,"   modrhs.clear();\n");
            } else {
               fprintf(file,"   modrhs.%s.clear();\n",ename);
            }
         }
      }
   }
}

//______________________________________________________________________________
static void R__WriteDestructorBody(FILE *file, TIter &next)
{
   TStreamerElement *element = 0;
   next.Reset();
   while ((element = (TStreamerElement*)next())) {
      if (element->GetType() == TVirtualStreamerInfo::kObjectp || element->GetType() == TVirtualStreamerInfo::kObjectP||
          element->GetType() == TVirtualStreamerInfo::kAnyp || element->GetType() == TVirtualStreamerInfo::kAnyP
          || element->GetType() == TVirtualStreamerInfo::kAnyPnoVT) 
      {
         const char *ename = element->GetName();
         const char *colon2 = strstr(ename,"::");
         if (colon2) ename = colon2+2;
         if (element->TestBit(TStreamerElement::kDoNotDelete)) {
            if(element->GetArrayLength() <= 1) {
               fprintf(file,"   %s = 0;\n",ename);
            } else {
               fprintf(file,"   memset(%s,0,%d);\n",ename,element->GetSize());
            }
         } else {
            if(element->GetArrayLength() <= 1) {
               fprintf(file,"   delete %s;   %s = 0;\n",ename,ename);
            } else {
               fprintf(file,"   for (Int_t i=0;i<%d;i++) delete %s[i];   memset(%s,0,%d);\n",element->GetArrayLength(),ename,ename,element->GetSize());
            }
         }
      }
      if (element->GetType() == TVirtualStreamerInfo::kCharStar) {
         const char *ename = element->GetName();
         if (element->TestBit(TStreamerElement::kDoNotDelete)) {
            fprintf(file,"   %s = 0;\n",ename);
         } else {
            fprintf(file,"   delete [] %s;   %s = 0;\n",ename,ename);
         }
      }
      if (TVirtualStreamerInfo::kOffsetP <= element->GetType() && element->GetType() < TVirtualStreamerInfo::kObject ) { 
         const char *ename = element->GetName();
         if (element->TestBit(TStreamerElement::kDoNotDelete)) {
            fprintf(file,"   %s = 0;\n",ename);
         } else if (element->HasCounter()) {
            fprintf(file,"   delete %s;   %s = 0;\n",ename,ename);
         } else {
            fprintf(file,"   delete [] %s;   %s = 0;\n",ename,ename);
         }
      }
      if (element->GetType() == TVirtualStreamerInfo::kSTL || element->GetType() == TVirtualStreamerInfo::kSTLp) {
         const char *ename = element->GetName();
         const char *prefix = "";
         if ( element->GetType() == TVirtualStreamerInfo::kSTLp ) {
            prefix = "*";
         } else if ( element->IsBase() ) {
            ename = "this";
         }
         TVirtualCollectionProxy *proxy = element->GetClassPointer()->GetCollectionProxy();
         if (!element->TestBit(TStreamerElement::kDoNotDelete) && element->GetClassPointer() && proxy) {
            Int_t stltype = ((TStreamerSTL*)element)->GetSTLtype();
            
            if (proxy->HasPointers()) {
               fprintf(file,"   std::for_each( (%s %s).rbegin(), (%s %s).rend(), DeleteObjectFunctor() );\n",prefix,ename,prefix,ename);
               //fprintf(file,"      %s::iterator iter;\n");
               //fprintf(file,"      %s::iterator begin = (%s %s).begin();\n");
               //fprintf(file,"      %s::iterator end (%s %s).end();\n");
               //fprintf(file,"      for( iter = begin; iter != end; ++iter) { delete *iter; }\n");
            } else {
               if (stltype == TStreamerElement::kSTLmap || stltype == TStreamerElement::kSTLmultimap) {
                  TString enamebasic = TMakeProject::UpdateAssociativeToVector(element->GetTypeNameBasic());
                  std::vector<std::string> inside;
                  int nestedLoc;
                  TClassEdit::GetSplit(enamebasic, inside, nestedLoc, TClassEdit::kLong64);
                  if (inside[1][inside[1].size()-1]=='*' || inside[2][inside[2].size()-1]=='*') {
                     fprintf(file,"   std::for_each( (%s %s).rbegin(), (%s %s).rend(), DeleteObjectFunctor() );\n",prefix,ename,prefix,ename);
                  }
               }
               }
         }
         if ( prefix[0] ) {
            fprintf(file,"   delete %s;   %s = 0;\n",ename,ename);
         }
      }
   }
}

//______________________________________________________________________________
void TStreamerInfo::GenerateDeclaration(FILE *fp, FILE *sfp, const TList *subClasses, Bool_t top)
{
   // Write the Declaration of class.

   if (fClassVersion == -3) {
      return;
   }

   bool needGenericTemplate = fElements==0 || fElements->GetEntries() == 0;
   Bool_t isTemplate = kFALSE;
   const char *clname = GetName();
   TString template_protoname;
   if (strchr(clname, ':')) {
      // We might have a namespace in front of the classname.
      Int_t len = strlen(clname);
      const char *name = clname;
      UInt_t nest = 0;
      UInt_t pr_pos = 0;
      for (Int_t cur = 0; cur < len; ++cur) {
         switch (clname[cur]) {
            case '<':
               ++nest;
               pr_pos = cur;
               isTemplate = kTRUE;
               break;
            case '>':
               --nest;
               break;
            case ':': {
               if (nest == 0 && clname[cur+1] == ':') {
                  // We have a scope
                  isTemplate = kFALSE;
                  name = clname + cur + 2;
               }
               break;
            }
         }
      }
      if (isTemplate) {
         template_protoname.Append(clname,pr_pos);
      }
      clname = name;
   } else {
      const char *where = strstr(clname, "<");
      isTemplate = where != 0;
      if (isTemplate) {
         template_protoname.Append(clname,where-clname);
      }
   }

   if (needGenericTemplate && isTemplate) {
      TString templateName(TMakeProject::GetHeaderName("template "+template_protoname,0));
      fprintf(fp, "#ifndef %s_h\n", templateName.Data());
      fprintf(fp, "#define %s_h\n", templateName.Data());
   }

   TString protoname;
   UInt_t numberOfNamespaces = TMakeProject::GenerateClassPrefix(fp, GetName(), top, protoname, 0, kFALSE, needGenericTemplate);

   // Generate class statement with base classes.
   TStreamerElement *element;
   TIter next(fElements);
   Int_t nbase = 0;
   while ((element = (TStreamerElement*)next())) {
      if (!element->IsBase()) continue;
      nbase++;
      const char *ename = element->GetName();
      if (nbase == 1) fprintf(fp," : public %s",ename);
      else            fprintf(fp," , public %s",ename);
   }
   fprintf(fp," {\n");

   // Generate forward declaration nested classes.
   if (subClasses && subClasses->GetEntries()) {
      bool needheader = true;

      TIter subnext(subClasses);
      TStreamerInfo *subinfo;
      Int_t len = strlen(GetName());
      while ((subinfo = (TStreamerInfo*)subnext())) {
         if (strncmp(GetName(),subinfo->GetName(),len)==0 && (subinfo->GetName()[len]==':') ) {
            if (subinfo->GetName()[len+1]==':' && strstr(subinfo->GetName()+len+2,":")==0) {
               if (needheader) {
                  fprintf(fp,"\npublic:\n");
                  fprintf(fp,"// Nested classes forward declaration.\n");
                  needheader = false;
               }
               TString sub_protoname;
               UInt_t sub_numberOfClasses = 0;
               UInt_t sub_numberOfNamespaces;
               if (subinfo->GetClassVersion() == -3) {
                  sub_numberOfNamespaces = TMakeProject::GenerateClassPrefix(fp, subinfo->GetName() + len+2, kFALSE, sub_protoname, &sub_numberOfClasses, 3);                  
               } else {
                  sub_numberOfNamespaces = TMakeProject::GenerateClassPrefix(fp, subinfo->GetName() + len+2, kFALSE, sub_protoname, &sub_numberOfClasses, kFALSE);
                  fprintf(fp, ";\n");
               }

               for (UInt_t i = 0;i < sub_numberOfClasses;++i) {
                  fprintf(fp, "}; // end of class.\n");
               }
               if (sub_numberOfNamespaces > 0) {
                  Error("GenerateDeclaration","Nested classes %s thought to be inside a namespace inside the class %s",subinfo->GetName(),GetName());
               }
            }
         }
      }
   }

   fprintf(fp,"\npublic:\n");
   fprintf(fp,"// Nested classes declaration.\n");

   // Generate nested classes.
   if (subClasses && subClasses->GetEntries()) {
      TIter subnext(subClasses,kIterBackward);
      TStreamerInfo *subinfo;
      Int_t len = strlen(GetName());
      while ((subinfo = (TStreamerInfo*)subnext())) {
         if (strncmp(GetName(),subinfo->GetName(),len)==0 && (subinfo->GetName()[len]==':')) {
            if (subinfo->GetName()[len+1]==':' && strstr(subinfo->GetName()+len+2,":")==0) {
               subinfo->GenerateDeclaration(fp, sfp, subClasses, kFALSE);
            }
         }
      }
   }

   fprintf(fp,"\npublic:\n");
   fprintf(fp,"// Data Members.\n");

   {
      // Generate data members.
      TString name(128);
      Int_t ltype = 12;
      Int_t ldata = 10;
      Int_t lt,ld,is;
      TString line;
      line.Resize(kMaxLen);      
      next.Reset();
      while ((element = (TStreamerElement*)next())) {

         if (element->IsBase()) continue;
         const char *ename = element->GetName();
         
         name = ename;
         for (Int_t i=0;i < element->GetArrayDim();i++) {
            name += TString::Format("[%d]",element->GetMaxIndex(i));
         }
         name += ";";
         ld = name.Length();
         
         TString enamebasic = element->GetTypeNameBasic();
         if (element->IsA() == TStreamerSTL::Class()) {
            // If we have a map, multimap, set or multiset,
            // and the key is a class, we need to replace the
            // container by a vector since we don't have the
            // comparator function.
            Int_t stltype = ((TStreamerSTL*)element)->GetSTLtype();
            switch (stltype) {
               case TStreamerElement::kSTLmap: 
               case TStreamerElement::kSTLmultimap:
               case TStreamerElement::kSTLset:
               case TStreamerElement::kSTLmultiset:
               {
                  enamebasic = TMakeProject::UpdateAssociativeToVector(enamebasic);
               }
               default:
                  // nothing to do.
                  break;
            }
         } 
         
         lt = enamebasic.Length();
         
         line = "   ";
         line += enamebasic;
         if (lt>=ltype) ltype = lt+1;
         
         for (is = 3+lt; is < (3+ltype); ++is) line += ' ';

         line += name;
         if (element->IsaPointer() && !strchr(line,'*')) line[2+ltype] = '*';
         
         if (ld>=ldata) ldata = ld+1;
         for (is = 3+ltype+ld; is < (3+ltype+ldata); ++is) line += ' ';

         line += "   //";
         line += element->GetTitle();
         fprintf(fp,"%s\n",line.Data());
      }
   }
   if (needGenericTemplate && isTemplate) {
      // Generate default functions, ClassDef and trailer.
      fprintf(fp,"\n   %s() {\n",protoname.Data());
      R__WriteConstructorBody(fp,next);      
      fprintf(fp,"   }\n");
      fprintf(fp,"   %s(const %s & rhs )\n",protoname.Data(),protoname.Data());
      R__WriteMoveConstructorBody(fp,protoname,next);
      fprintf(fp,"   }\n");
      fprintf(fp,"   virtual ~%s() {\n",protoname.Data());
      R__WriteDestructorBody(fp,next);
      fprintf(fp,"   }\n\n");

   } else {
      // Generate default functions, ClassDef and trailer.
      fprintf(fp,"\n   %s();\n",protoname.Data());
      fprintf(fp,"   %s(const %s & );\n",protoname.Data(),protoname.Data());
      fprintf(fp,"   virtual ~%s();\n\n",protoname.Data());

      // Add the implementations to the source.cxx file.
      TString guard( TMakeProject::GetHeaderName( GetName(), 0, kTRUE ) );
      fprintf(sfp,"#ifndef %s_cxx\n",guard.Data());
      fprintf(sfp,"#define %s_cxx\n",guard.Data());
      fprintf(sfp,"%s::%s() {\n",GetName(),protoname.Data());
      R__WriteConstructorBody(sfp,next);
      fprintf(sfp,"}\n");

      fprintf(sfp,"%s::%s(const %s & rhs)\n",GetName(),protoname.Data(),protoname.Data());
      R__WriteMoveConstructorBody(sfp,protoname,next);
      fprintf(sfp,"}\n");

      fprintf(sfp,"%s::~%s() {\n",GetName(),protoname.Data());
      R__WriteDestructorBody(sfp,next);
      fprintf(sfp,"}\n");
      fprintf(sfp,"#endif // %s_cxx\n\n",guard.Data());
   }

   TClass *cl = gROOT->GetClass(GetName());
   if (fClassVersion > 1 || (cl && cl->InheritsFrom(TObject::Class())) ) {
      // add 1 to class version in case we didn't manage reproduce the class layout to 100%.
      if (fClassVersion == 0) {
         // If the class was declared 'transient', keep it that way.
         fprintf(fp,"   ClassDef(%s,%d); // Generated by MakeProject.\n",protoname.Data(),0);         
      } else {
         fprintf(fp,"   ClassDef(%s,%d); // Generated by MakeProject.\n",protoname.Data(),fClassVersion + 1);
      }
   }
   fprintf(fp,"};\n");

   for(UInt_t i=0;i<numberOfNamespaces;++i) {
      fprintf(fp,"} // namespace\n");
   }

   if (needGenericTemplate && isTemplate) {
      fprintf(fp,"#endif // generic template declaration\n");
   }
}

//______________________________________________________________________________
UInt_t TStreamerInfo::GenerateIncludes(FILE *fp, char *inclist, const TList *extrainfos)
{
   // Add to the header file, the #include need for this class

   UInt_t ninc = 0;

   const char *clname = GetName();
   if (strchr(clname,'<')) {
      // This is a template, we need to check the template parameter.
      ninc += TMakeProject::GenerateIncludeForTemplate(fp, clname, inclist, kFALSE, extrainfos);
   }

   TString name(1024);
   Int_t ltype = 10;
   Int_t ldata = 10;
   Int_t lt;
   Int_t ld;
   TIter next(fElements);
   TStreamerElement *element;
   Bool_t incRiostream = kFALSE;
   while ((element = (TStreamerElement*)next())) {
      //if (element->IsA() == TStreamerBase::Class()) continue;
      const char *ename = element->GetName();
      const char *colon2 = strstr(ename,"::");
      if (colon2) ename = colon2+2;
      name = ename;
      for (Int_t i=0;i < element->GetArrayDim();i++) {
         name += TString::Format("[%d]",element->GetMaxIndex(i));
      }
      ld = name.Length();
      lt = strlen(element->GetTypeName());
      if (ltype < lt) ltype = lt;
      if (ldata < ld) ldata = ld;

      //must include Riostream.h in case of an STL container
      if (!incRiostream && element->InheritsFrom(TStreamerSTL::Class())) {
         incRiostream = kTRUE;
         TMakeProject::AddInclude( fp, "Riostream.h", kFALSE, inclist);
      }

      //get include file name if any
      const char *include = element->GetInclude();
      if (strlen(include) == 0) continue;

      Bool_t greater = (include[0]=='<');
      include++;

      if (strncmp(include,"include/",8)==0) {
         include += 8;
      } 
      if (strncmp(include,"include\\",9)==0) {
         include += 9;
      }
      if (strncmp(element->GetTypeName(),"pair<",strlen("pair<"))==0) {
         TMakeProject::AddInclude( fp, "utility", kTRUE, inclist);
      } else if (strncmp(element->GetTypeName(),"auto_ptr<",strlen("auto_ptr<"))==0) {
         TMakeProject::AddInclude( fp, "memory", kTRUE, inclist);
      } else {
         TString incName( include, strlen(include)-1 );
         incName = TMakeProject::GetHeaderName(incName,extrainfos);
         TMakeProject::AddInclude( fp, incName.Data(), greater, inclist);
      }

      if (strchr(element->GetTypeName(),'<')) {
         // This is a template, we need to check the template parameter.
         ninc += TMakeProject::GenerateIncludeForTemplate(fp, element->GetTypeName(), inclist, kFALSE, extrainfos);
      }
   }
   if (inclist[0]==0) {
      TMakeProject::AddInclude( fp, "TNamed.h", kFALSE, inclist);
   }
   return ninc;
}

//______________________________________________________________________________
Int_t TStreamerInfo::GenerateHeaderFile(const char *dirname, const TList *subClasses, const TList *extrainfos)
{
   // Generate header file for the class described by this TStreamerInfo
   // the function is called by TFile::MakeProject for each class in the file

   // if (fClassVersion == -4) return 0;
   if (TClassEdit::IsSTLCont(GetName())) return 0;
   if (strncmp(GetName(),"pair<",strlen("pair<"))==0) return 0;
   if (strncmp(GetName(),"auto_ptr<",strlen("auto_ptr<"))==0) return 0;

   TClass *cl = TClass::GetClass(GetName());
   if (cl) {
      if (cl->GetClassInfo()) return 0; // skip known classes
   }
   Bool_t isTemplate = kFALSE;
   if (strchr(GetName(),':')) {
      UInt_t len = strlen(GetName());
      UInt_t nest = 0;
      UInt_t scope = 0;
      for(UInt_t i=len; i>0; --i) {
         switch(GetName()[i]) {
            case '>': ++nest; if (scope==0) { isTemplate = kTRUE; } break;
            case '<': --nest; break;
            case ':': 
               if (nest==0 && GetName()[i-1]==':') {
                  // We have a scope
                  TString nsname(GetName(), i-1);
                  cl = gROOT->GetClass(nsname);
                  if (cl && (cl->Size()!=0 || (cl->Size()==0 && cl->GetClassInfo()==0 /*empty 'base' class on file*/))) {
                     // This class is actually nested.
                     return 0;
                  } else if (cl == 0 && extrainfos != 0) {
                     TStreamerInfo *clinfo = (TStreamerInfo*)extrainfos->FindObject(nsname);
                     if (clinfo && clinfo->GetClassVersion() == -5) {
                        // This class is actually nested.
                        return 0;
                     }
                  }
                  ++scope;
               }
               break;
         }
      }
   }
   Bool_t needGenericTemplate = isTemplate && (fElements==0 || fElements->GetEntries()==0); 

   if (gDebug) printf("generating code for class %s\n",GetName());

   // Open the file

   TString headername( TMakeProject::GetHeaderName( GetName(), extrainfos ) );
   TString filename;
   filename.Form("%s/%s.h",dirname,headername.Data());

   FILE *fp = fopen(filename.Data(),"w");
   if (!fp) {
      Error("MakeProject","Cannot open output file:%s\n",filename.Data());
      return 0;
   }

   filename.Form("%s/%sProjectHeaders.h",dirname,gSystem->BaseName(dirname));
   FILE *allfp = fopen(filename.Data(),"a");
   if (!allfp) {
      Error("MakeProject","Cannot open output file:%s\n",filename.Data());
      fclose(fp);
      return 0;
   }
   fprintf(allfp,"#include \"%s.h\"\n", headername.Data());
   fclose(allfp);

   char *inclist = new char[50000];
   inclist[0] = 0;

   // Generate class header.
   TDatime td;
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"//   This class has been generated by TFile::MakeProject\n");
   fprintf(fp,"//     (%s by ROOT version %s)\n",td.AsString(),gROOT->GetVersion());
   fprintf(fp,"//      from the StreamerInfo in file %s\n",gDirectory->GetFile()->GetName());
   fprintf(fp,"//////////////////////////////////////////////////////////\n");
   fprintf(fp,"\n");
   fprintf(fp,"\n");
   fprintf(fp,"#ifndef %s_h\n",headername.Data());
   fprintf(fp,"#define %s_h\n",headername.Data());
   TMakeProject::GenerateForwardDeclaration(fp, GetName(), inclist, kFALSE, needGenericTemplate, extrainfos);
   fprintf(fp,"\n");

   UInt_t ninc = 0;
   ninc += GenerateIncludes(fp, inclist, extrainfos);
   if (subClasses) {
      TIter subnext(subClasses);
      TStreamerInfo *subinfo;
      while ((subinfo = (TStreamerInfo*)subnext())) {
         ninc = subinfo->GenerateIncludes(fp, inclist, extrainfos);
      }
   }   
   fprintf(fp,"\n");

   TString sourcename; sourcename.Form( "%s/%sProjectSource.cxx", dirname, gSystem->BaseName(dirname) );
   FILE *sfp = fopen( sourcename.Data(), "a" );
   GenerateDeclaration(fp, sfp, subClasses);
   
   TMakeProject::GeneratePostDeclaration(fp, this, inclist);

   fprintf(fp,"#endif\n");

   delete [] inclist;
   fclose(fp);
   fclose(sfp);
   return 1;
}

//______________________________________________________________________________
TStreamerElement *TStreamerInfo::GetCurrentElement()
{
   //static function returning a pointer to the current TStreamerElement
   //fgElement points to the current TStreamerElement being read in ReadBuffer
   return fgElement;
}

//______________________________________________________________________________
Int_t TStreamerInfo::GetDataMemberOffset(TDataMember *dm, TMemberStreamer *&streamer) const
{
   // Compute data member offset
   // return pointer to the Streamer function if one exists

   TIter nextr(fClass->GetListOfRealData());
   char dmbracket[256];
   snprintf(dmbracket,255,"%s[",dm->GetName());
   Int_t offset = kMissing;
   if (fClass->GetDeclFileLine() < 0) offset = dm->GetOffset();
   TRealData *rdm;
   while ((rdm = (TRealData*)nextr())) {
      char *rdmc = (char*)rdm->GetName();
      //next statement required in case a class and one of its parent class
      //have data members with the same name
      if (dm->IsaPointer() && rdmc[0] == '*') rdmc++;

      if (rdm->GetDataMember() != dm) continue;
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
Int_t TStreamerInfo::GetOffset(const char *elementName) const
{
   // return the offset of the data member as indicated by this StreamerInfo

   if (elementName == 0) return 0;

   Int_t offset = 0;
   TStreamerElement *elem = (TStreamerElement*)fElements->FindObject(elementName);
   if (elem) offset = elem->GetOffset();

   return offset;
}

//______________________________________________________________________________
Int_t TStreamerInfo::GetSize() const
{
   //  return total size of all persistent elements of the class (with offsets)

   return fSize;
}

//______________________________________________________________________________
Int_t TStreamerInfo::GetSizeElements() const
{
   //  return total size of all persistent elements of the class
   //  use GetSize if you want to get the real size in memory

   TIter next(fElements);
   TStreamerElement *element;
   Int_t asize = 0;
   while ((element = (TStreamerElement*)next())) {
      asize += element->GetSize();
   }
   return asize;
}

//______________________________________________________________________________
TStreamerElement* TStreamerInfo::GetStreamerElement(const char* datamember, Int_t& offset) const
{
   // Return the StreamerElement of "datamember" inside our
   // class or any of its base classes.  The offset information
   // contained in the StreamerElement is related to its immediately
   // containing class, so we return in 'offset' the offset inside
   // our class.

   if (!fElements) {
      return 0;
   }

   // Look first at the data members and base classes
   // of our class.
   TStreamerElement* element = (TStreamerElement*) fElements->FindObject(datamember);
   if (element) {
      offset = element->GetOffset();
      return element;
   }

   // Not found, so now try the data members and base classes
   // of the base classes of our class.
   if (fClass->GetClassInfo()) {
      // Our class has a dictionary loaded, use it to search the base classes.
      TStreamerElement* base_element = 0;
      TBaseClass* base = 0;
      TClass* base_cl = 0;
      Int_t base_offset = 0;
      Int_t local_offset = 0;
      TIter nextb(fClass->GetListOfBases());
      // Iterate on list of base classes.
      while ((base = (TBaseClass*) nextb())) {
         base_cl = TClass::GetClass(base->GetName());
         base_element = (TStreamerElement*) fElements->FindObject(base->GetName());
         if (!base_cl || !base_element) {
            continue;
         }
         base_offset = base_element->GetOffset();
         element = ((TStreamerInfo*)base_cl->GetStreamerInfo())->GetStreamerElement(datamember, local_offset);
         if (element) {
            offset = base_offset + local_offset;
            return element;
         }
      }
   } else {
      // Our class's dictionary is not loaded. Search through the base class streamer elements.
      TIter next(fElements);
      TStreamerElement* curelem = 0;
      while ((curelem = (TStreamerElement*) next())) {
         if (curelem->InheritsFrom(TStreamerBase::Class())) {
            TClass* baseClass = curelem->GetClassPointer();
            if (!baseClass) {
               continue;
            }
            Int_t base_offset = curelem->GetOffset();
            Int_t local_offset = 0;
            element = ((TStreamerInfo*)baseClass->GetStreamerInfo())->GetStreamerElement(datamember, local_offset);
            if (element) {
               offset = base_offset + local_offset;
               return element;
            }
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
TStreamerElement* TStreamerInfo::GetStreamerElementReal(Int_t i, Int_t j) const
{
   //  TStreamerInfo  holds two types of data structures
   //    -TObjArray* fElements; containing the list of all TStreamerElement
   //       objects for this class version.
   //    -ULong_t*  fElem;  containing the preprocessed information
   //       by TStreamerInfo::Compile In case consecutive data members
   //       are of the same type, the Compile function declares the consecutive
   //       elements as one single element in fElems.
   //
   //  example with the class TAttLine
   //   TClass::GetClass("TAttLine")->GetStreamerInfo()->ls(); produces;
   //      StreamerInfo for class: TAttLine, version=1
   //       short        fLineColor      offset=  4 type= 2 line color
   //       short        fLineStyle      offset=  6 type= 2 line style
   //       short        fLineWidth      offset=  8 type= 2 line width
   //        i= 0, fLineColor      type= 22, offset=  4, len=3, method=0
   //  For I/O implementations (eg. XML) , one has to know the original name
   //  of the data member. This function can be used to return a pointer
   //  to the original TStreamerElement object corresponding to the j-th
   //  element of a compressed array in fElems.
   //
   //  parameters description:
   //    - i: the serial number in array fElem
   //    - j: the element number in the array of consecutive types
   //  In the above example the class TAttLine has 3 consecutive data members
   //  of the same type "short". Compile makes one single array of 3 elements.
   //  To access the TStreamerElement for the second element
   //  of this array, one can call:
   //     TStreamerElement *el = GetStreamerElementReal(0,1);
   //     const char* membername = el->GetName();
   //  This function is typically called from Tbuffer, TXmlBuffer

   if (i < 0 || i >= fNdata) return 0;
   if (j < 0) return 0;
   if (!fElements) return 0;
   TStreamerElement *se = (TStreamerElement*)fElem[i];
   if (!se) return 0;
   Int_t nelems = fElements->GetEntriesFast();
   for (Int_t ise=0;ise < nelems;ise++) {
      if (se != (TStreamerElement*)fElements->UncheckedAt(ise)) continue;
      if (ise+j >= nelems) return 0;
      return (TStreamerElement*)fElements->UncheckedAt(ise+j);
   }
   return 0;
}

//______________________________________________________________________________
Double_t  TStreamerInfo::GetValueAux(Int_t type, void *ladd, Int_t k, Int_t len)
{
   // Get the value from inside a collection.

   if (type>=kConv && type<kSTL) {
      type -= kConv;
   }
   switch (type) {
      // basic types
      case kBool:              {Bool_t *val   = (Bool_t*)ladd;   return Double_t(*val);}
      case kChar:              {Char_t *val   = (Char_t*)ladd;   return Double_t(*val);}
      case kShort:             {Short_t *val  = (Short_t*)ladd;  return Double_t(*val);}
      case kInt:               {Int_t *val    = (Int_t*)ladd;    return Double_t(*val);}
      case kLong:              {Long_t *val   = (Long_t*)ladd;   return Double_t(*val);}
      case kLong64:            {Long64_t *val = (Long64_t*)ladd; return Double_t(*val);}
      case kFloat:             {Float_t *val  = (Float_t*)ladd;  return Double_t(*val);}
      case kFloat16:           {Float_t *val  = (Float_t*)ladd;  return Double_t(*val);}
      case kDouble:            {Double_t *val = (Double_t*)ladd; return Double_t(*val);}
      case kDouble32:          {Double_t *val = (Double_t*)ladd; return Double_t(*val);}
      case kUChar:             {UChar_t *val  = (UChar_t*)ladd;  return Double_t(*val);}
      case kUShort:            {UShort_t *val = (UShort_t*)ladd; return Double_t(*val);}
      case kUInt:              {UInt_t *val   = (UInt_t*)ladd;   return Double_t(*val);}
      case kULong:             {ULong_t *val  = (ULong_t*)ladd;  return Double_t(*val);}
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case kULong64:           {Long64_t *val = (Long64_t*)ladd;  return Double_t(*val);}
#else
      case kULong64:           {ULong64_t *val= (ULong64_t*)ladd; return Double_t(*val);}
#endif
      case kBits:              {UInt_t *val   = (UInt_t*)ladd;   return Double_t(*val);}

         // array of basic types  array[8]
      case kOffsetL + kBool:    {Bool_t *val   = (Bool_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kChar:    {Char_t *val   = (Char_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kShort:   {Short_t *val  = (Short_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kInt:     {Int_t *val    = (Int_t*)ladd;    return Double_t(val[k]);}
      case kOffsetL + kLong:    {Long_t *val   = (Long_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kLong64:  {Long64_t *val = (Long64_t*)ladd; return Double_t(val[k]);}
      case kOffsetL + kFloat:   {Float_t *val  = (Float_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kFloat16: {Float_t *val  = (Float_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kDouble:  {Double_t *val = (Double_t*)ladd; return Double_t(val[k]);}
      case kOffsetL + kDouble32:{Double_t *val = (Double_t*)ladd; return Double_t(val[k]);}
      case kOffsetL + kUChar:   {UChar_t *val  = (UChar_t*)ladd;  return Double_t(val[k]);}
      case kOffsetL + kUShort:  {UShort_t *val = (UShort_t*)ladd; return Double_t(val[k]);}
      case kOffsetL + kUInt:    {UInt_t *val   = (UInt_t*)ladd;   return Double_t(val[k]);}
      case kOffsetL + kULong:   {ULong_t *val  = (ULong_t*)ladd;  return Double_t(val[k]);}
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case kOffsetL + kULong64: {Long64_t *val = (Long64_t*)ladd;  return Double_t(val[k]);}
#else
      case kOffsetL + kULong64:{ULong64_t *val= (ULong64_t*)ladd; return Double_t(val[k]);}
#endif

#define READ_ARRAY(TYPE_t)                               \
         {                                               \
            Int_t sub_instance, index;                   \
            Int_t instance = k;                          \
            if (len) {                                   \
               index = instance / len;                   \
               sub_instance = instance % len;            \
            } else {                                     \
               index = instance;                         \
               sub_instance = 0;                         \
            }                                            \
            TYPE_t **val =(TYPE_t**)(ladd);              \
            return Double_t((val[sub_instance])[index]); \
         }

         // pointer to an array of basic types  array[n]
      case kOffsetP + kBool_t:    READ_ARRAY(Bool_t)
      case kOffsetP + kChar_t:    READ_ARRAY(Char_t)
      case kOffsetP + kShort_t:   READ_ARRAY(Short_t)
      case kOffsetP + kInt_t:     READ_ARRAY(Int_t)
      case kOffsetP + kLong_t:    READ_ARRAY(Long_t)
      case kOffsetP + kLong64_t:  READ_ARRAY(Long64_t)
      case kOffsetP + kFloat16_t:
      case kOffsetP + kFloat_t:   READ_ARRAY(Float_t)
      case kOffsetP + kDouble32_t:
      case kOffsetP + kDouble_t:  READ_ARRAY(Double_t)
      case kOffsetP + kUChar_t:   READ_ARRAY(UChar_t)
      case kOffsetP + kUShort_t:  READ_ARRAY(UShort_t)
      case kOffsetP + kUInt_t:    READ_ARRAY(UInt_t)
      case kOffsetP + kULong_t:   READ_ARRAY(ULong_t)
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
      case kOffsetP + kULong64_t: READ_ARRAY(Long64_t)
#else
      case kOffsetP + kULong64_t: READ_ARRAY(ULong64_t)
#endif

          // array counter //[n]
      case kCounter:           {Int_t *val    = (Int_t*)ladd;    return Double_t(*val);}
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
      if (i < 0) return 0;
      ladd  = pointer + fOffset[i];
      atype = fNewType[i];
      len = ((TStreamerElement*)fElem[i])->GetArrayLength();
      if (atype == kSTL) {
         TClass *newClass = ((TStreamerElement*)fElem[i])->GetNewClass();
         if (newClass == 0) {
            newClass = ((TStreamerElement*)fElem[i])->GetClassPointer();
         }
         TClass *innerClass = newClass->GetCollectionProxy()->GetValueClass();
         if (innerClass) {
            return 0; // We don't know which member of the class we would want.
         } else {
            TVirtualCollectionProxy *proxy = newClass->GetCollectionProxy();
            // EDataType is a subset of TStreamerInfo::EReadWrite
            atype = (TStreamerInfo::EReadWrite)proxy->GetType();
            TVirtualCollectionProxy::TPushPop pop(proxy,ladd);
            Int_t nc = proxy->Size();
            if (j >= nc) return 0;
            char *element_ptr = (char*)proxy->At(j);
            return GetValueAux(atype,element_ptr,0,1);
         }
      }
   }
   return GetValueAux(atype,ladd,j,len);
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
   return GetValueAux(fType[i],ladd,k,((TStreamerElement*)fElem[i])->GetArrayLength());
}

//______________________________________________________________________________
Double_t TStreamerInfo::GetValueSTL(TVirtualCollectionProxy *cont, Int_t i, Int_t j, int k, Int_t eoffset) const
{
   //  return value of element i in object number j in a TClonesArray and eventually
   // element k in a sub-array.

   Int_t nc = cont->Size();
   if (j >= nc) return 0;

   char *pointer = (char*)cont->At(j);
   char *ladd    = pointer + eoffset + fOffset[i];
   return GetValueAux(fType[i],ladd,k,((TStreamerElement*)fElem[i])->GetArrayLength());
}

//______________________________________________________________________________
Double_t TStreamerInfo::GetValueSTLP(TVirtualCollectionProxy *cont, Int_t i, Int_t j, int k, Int_t eoffset) const
{
   //  return value of element i in object number j in a TClonesArray and eventually
   // element k in a sub-array.
   Int_t nc = cont->Size();

   if (j >= nc) return 0;

   char **ptr = (char**)cont->At(j);
   char *pointer = *ptr;

   char *ladd    = pointer + eoffset + fOffset[i];
   return GetValueAux(fType[i],ladd,k,((TStreamerElement*)fElem[i])->GetArrayLength());
}

//______________________________________________________________________________
void TStreamerInfo::InsertArtificialElements(const TObjArray *rules) 
{
   // Insert new members as expressed in the array of TSchemaRule(s).

   if (!rules) return;

   TIter next(fElements);
   UInt_t count = 0;

   for(Int_t art = 0; art < rules->GetEntries(); ++art) {
      ROOT::TSchemaRule *rule = (ROOT::TSchemaRule*)rules->At(art);
      if( rule->IsRenameRule() || rule->IsAliasRule() )
         continue;
      next.Reset();
      Bool_t match = kFALSE;
      TStreamerElement *element;
      while ((element = (TStreamerElement*) next())) {
         if ( rule->HasTarget( element->GetName() ) ) {            
            // If the rule targets an existing member but it is also a source,
            // we still need to insert the rule.
            match = ! ((ROOT::TSchemaMatch*)rules)->HasRuleWithSource( element->GetName(), kTRUE );
            
            // Check whether this is an 'attribute' rule.
            if ( rule->GetAttributes()[0] != 0 ) {
               TString attr( rule->GetAttributes() );
               attr.ToLower();
               if (attr.Contains("owner")) {
                  if (attr.Contains("notowner")) {
                     element->SetBit(TStreamerElement::kDoNotDelete);
                  } else {
                     element->ResetBit(TStreamerElement::kDoNotDelete);
                  }
               }
            
            }
            break;
         }
      }
      if (!match) {
         TStreamerArtificial *newel;
         if (rule->GetTarget()==0) {
            TString newName;
            newName.Form("%s_rule%d",fClass->GetName(),count);
            newel = new TStreamerArtificial(newName,"", 
                                            fClass->GetDataMemberOffset(newName), 
                                            TStreamerInfo::kArtificial, 
                                            "void");
            newel->SetReadFunc( rule->GetReadFunctionPointer() );
            newel->SetReadRawFunc( rule->GetReadRawFunctionPointer() );
            fElements->Add(newel);
         } else {
            TObjString * objstr = (TObjString*)(rule->GetTarget()->At(0));
            if (objstr) {
               TString newName = objstr->String();
               if ( fClass->GetDataMember( newName ) ) {
                  newel = new TStreamerArtificial(newName,"", 
                                                  fClass->GetDataMemberOffset(newName),
                                                  TStreamerInfo::kArtificial, 
                                                  fClass->GetDataMember( newName )->GetTypeName());
                  newel->SetReadFunc( rule->GetReadFunctionPointer() );
                  newel->SetReadRawFunc( rule->GetReadRawFunctionPointer() );
                  fElements->Add(newel);
               } else {
                  // This would be a completely new member (so it would need to be cached)
                  // TOBEDONE
               }
               for(Int_t other = 1; other < rule->GetTarget()->GetEntries(); ++other) {
                  objstr = (TObjString*)(rule->GetTarget()->At(other));
                  if (objstr) {
                     newName = objstr->String();
                     if ( fClass->GetDataMember( newName ) ) {
                        newel = new TStreamerArtificial(newName,"", 
                                                        fClass->GetDataMemberOffset(newName),
                                                        TStreamerInfo::kArtificial, 
                                                        fClass->GetDataMember( newName )->GetTypeName());
                        fElements->Add(newel);
                     }
                  }
               }
            } // For each target of the rule
         }
      } // None of the target of the rule are on file.
   }
}

//______________________________________________________________________________
void TStreamerInfo::ls(Option_t *option) const
{
   //  List the TStreamerElement list and also the precomputed tables
   if (fClass && fClass->IsForeign() && fClass->GetClassVersion()<2) {
      Printf("\nStreamerInfo for class: %s, checksum=0x%x",GetName(),GetCheckSum());
   } else {
      Printf("\nStreamerInfo for class: %s, version=%d, checksum=0x%x",GetName(),fClassVersion,GetCheckSum());
   }

   if (fElements) {
      TIter    next(fElements);
      TObject *obj;
      while ((obj = next()))
         obj->ls(option);
   }
   for (Int_t i=0;i < fNdata;i++) {
      TStreamerElement *element = (TStreamerElement*)fElem[i];
      TString sequenceType = " [";
      Bool_t first = kTRUE;
      if (element->TestBit(TStreamerElement::kCache)) {
         first = kFALSE;
         sequenceType += "cached";
      }
      if (element->TestBit(TStreamerElement::kRepeat)) {
         if (!first) sequenceType += ",";
         first = kFALSE;
         sequenceType += "repeat";
      }
      if (element->TestBit(TStreamerElement::kDoNotDelete)) {
         if (!first) sequenceType += ",";
         first = kFALSE;
         sequenceType += "nodelete";
      }
      if (first) sequenceType.Clear();
      else sequenceType += "]";

      Printf("   i=%2d, %-15s type=%3d, offset=%3d, len=%d, method=%ld%s",i,element->GetName(),fType[i],fOffset[i],fLength[i],fMethod[i],sequenceType.Data());
   }
}

//______________________________________________________________________________
void* TStreamerInfo::New(void *obj)
{
   // An emulated object is created at address obj, if obj is null we
   // allocate memory for the object.

   //???FIX ME: What about varying length array elements?

   char* p = (char*) obj;

   TIter next(fElements);

   if (!p) {
      // Allocate and initialize the memory block.
      p = new char[fSize];
      memset(p, 0, fSize);
   }

   next.Reset();
   TStreamerElement* element = (TStreamerElement*) next();

   for (; element; element = (TStreamerElement*) next()) {

      // Skip elements which have not been allocated memory.
      if (element->GetOffset() == kMissing) {
         continue;
      }

      // Skip elements for which we do not have any class
      // information.  FIXME: Document how this could happen.
      TClass* cle = element->GetClassPointer();
      if (!cle) {
         continue;
      }

      char* eaddr = p + element->GetOffset();
      Int_t etype = element->GetType();

      //cle->GetStreamerInfo(); //necessary in case "->" is not specified

      switch (etype) {

         case kAnyP:
         case kObjectP:
         case kSTLp:
         {
            // Initialize array of pointers with null pointers.
            char** r = (char**) eaddr;
            Int_t len = element->GetArrayLength();
            for (Int_t i = 0; i < len; ++i) {
               r[i] = 0;
            }
         }
         break;

         case kObjectp:
         case kAnyp:
         {
            // If the option "->" is given in the data member comment field
            // it is assumed that the object exists before reading data in,
            // so we create an object.
            if (cle != TClonesArray::Class()) {
               void** r = (void**) eaddr;
               *r = cle->New();
            } else {
               // In the case of a TClonesArray, the class name of
               // the contained objects must be specified in the
               // data member comment in this format:
               //    TClonesArray* myVar; //->(className)
               const char* title = element->GetTitle();
               const char* bracket1 = strrchr(title, '(');
               const char* bracket2 = strrchr(title, ')');
               if (bracket1 && bracket2 && (bracket2 != (bracket1 + 1))) {
                  Int_t len = bracket2 - (bracket1 + 1);
                  char* clonesClass = new char[len+1];
                  clonesClass[0] = '\0';
                  strncat(clonesClass, bracket1 + 1, len);
                  void** r = (void**) eaddr;
                  *r = (void*) new TClonesArray(clonesClass);
                  delete[] clonesClass;
               } else {
                  //Warning("New", "No class name found for TClonesArray initializer in data member comment (expected \"//->(className)\"");
                  void** r = (void**) eaddr;
                  *r = (void*) new TClonesArray();
               }
            }
         }
         break;

         case kBase:
         case kObject:
         case kAny:
         case kTObject:
         case kTString:
         case kTNamed:
         case kSTL:
         {
            cle->New(eaddr);
         }
         break;

         case kObject + kOffsetL:
         case kAny + kOffsetL:
         case kTObject + kOffsetL:
         case kTString + kOffsetL:
         case kTNamed + kOffsetL:
         case kSTL + kOffsetL:
         {
            Int_t size = cle->Size();
            char* r = eaddr;
            Int_t len = element->GetArrayLength();
            for (Int_t i = 0; i < len; ++i, r += size) {
               cle->New(r);
            }
         }
         break;

      } // switch etype
   } // for TIter next(fElements)

   for(int nbase = 0; nbase < fNVirtualInfoLoc; ++nbase) {
      *(TStreamerInfo**)(p + fVirtualInfoLoc[nbase]) = this;
   }
   ++fLiveCount;
   return p;
}

//______________________________________________________________________________
void* TStreamerInfo::NewArray(Long_t nElements, void *ary)
{
   // An array of emulated objects is created at address ary, if ary is null,
   // we allocate memory for the array.

   if (fClass == 0) {
      Error("NewArray", "TClass pointer is null!");
      return 0;
   }

   Int_t size = fClass->Size();

   char* p = (char*) ary;

   if (!p) {
      Long_t len = nElements * size + sizeof(Long_t)*2;
      p = new char[len];
      memset(p, 0, len);
   }

   // Store the array cookie
   Long_t* r = (Long_t*) p;
   r[0] = size;
   r[1] = nElements;
   char* dataBegin = (char*) &r[2];

   // Do a placement new for each element.
   p = dataBegin;
   for (Long_t cnt = 0; cnt < nElements; ++cnt) {
      New(p);
      p += size;
   } // for nElements

   return dataBegin;
}


#define DeleteBasicPointer(addr,element,name)                           \
   {                                                                    \
      name **f = (name**)(addr);                                        \
      int n = element->GetArrayLength() ? element->GetArrayLength() : 1;\
      for(int j=0;j<n;j++) {                                            \
         delete [] f[j];                                                \
         f[j] = 0;                                                      \
      }                                                                 \
   }

//______________________________________________________________________________
void TStreamerInfo::DestructorImpl(void* obj, Bool_t dtorOnly)
{
   // Internal part of the destructor.
   // Destruct each of the datamembers in the same order
   // as the implicit destructor would.

   R__ASSERT(obj != 0);

   char *p = (char*)obj;

   Int_t nelements = fElements->GetEntriesFast();
   //for (; ele; ele = (TStreamerElement*) next())
   for (Int_t elenum = nelements - 1; elenum >= 0; --elenum) {
      TStreamerElement* ele = (TStreamerElement*) fElements->UncheckedAt(elenum);
      if (ele->GetOffset() == kMissing) continue;
      char* eaddr = p + ele->GetOffset();


      Int_t etype = ele->GetType();

      switch(etype) {
         case TStreamerInfo::kOffsetP + TStreamerInfo::kBool:   DeleteBasicPointer(eaddr,ele,Bool_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:   DeleteBasicPointer(eaddr,ele,Char_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:  DeleteBasicPointer(eaddr,ele,Short_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:    DeleteBasicPointer(eaddr,ele,Int_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:   DeleteBasicPointer(eaddr,ele,Long_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64: DeleteBasicPointer(eaddr,ele,Long64_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat16:
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:  DeleteBasicPointer(eaddr,ele,Float_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble: DeleteBasicPointer(eaddr,ele,Double_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:  DeleteBasicPointer(eaddr,ele,UChar_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort: DeleteBasicPointer(eaddr,ele,UShort_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:   DeleteBasicPointer(eaddr,ele,UInt_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:  DeleteBasicPointer(eaddr,ele,ULong_t);  continue;
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64:DeleteBasicPointer(eaddr,ele,ULong64_t);  continue;
      }



      TClass* cle = ele->GetClassPointer();
      if (!cle) continue;


      if (etype == kObjectp || etype == kAnyp) {
         // Destroy an array of pre-allocated objects.
         Int_t len = ele->GetArrayLength();
         if (!len) {
            len = 1;
         }
         void** r = (void**) eaddr;
         for (Int_t j = len - 1; j >= 0; --j) {
            if (r[j]) {
               cle->Destructor(r[j]);
               r[j] = 0;
            }
         }
      }

      if ((etype == kObjectP || etype == kAnyP || etype == kSTLp) && !ele->TestBit(TStreamerElement::kDoNotDelete)) {
         // Destroy an array of pointers to not-pre-allocated objects.
         Int_t len = ele->GetArrayLength();
         if (!len) {
            len = 1;
         }
         void** r = (void**) eaddr;
         for (Int_t j = len - 1; j >= 0; --j) {
            if (r[j]) {
               cle->Destructor(r[j]);
               r[j] = 0;
            }
         }
      }

      if (etype == kObject || etype == kAny || etype == kBase ||
          etype == kTObject || etype == kTString || etype == kTNamed) {
         // A data member is destroyed, but not deleted.
         cle->Destructor(eaddr, kTRUE);
      }

      if (etype == kSTL) {
         // A data member is destroyed, but not deleted.
         TVirtualCollectionProxy *pr = cle->GetCollectionProxy();
         if (!pr) {
            cle->Destructor(eaddr, kTRUE);
         } else {
            if (ele->TestBit(TStreamerElement::kDoNotDelete)) {
               TVirtualCollectionProxy::TPushPop env(cle->GetCollectionProxy(), eaddr); // used for both this 'clear' and the 'clear' inside destructor.
               cle->GetCollectionProxy()->Clear(); // empty the collection without deleting the pointer
               pr->Destructor(eaddr, kTRUE);
            } else {
               pr->Destructor(eaddr, kTRUE);
            }
         }
      }

      if (etype == kObject  + kOffsetL || etype == kAny     + kOffsetL ||
          etype == kTObject + kOffsetL || etype == kTString + kOffsetL ||
          etype == kTNamed  + kOffsetL || etype == kSTL     + kOffsetL) {
         // For a data member which is an array of objects, we
         // destroy the objects, but do not delete them.
         Int_t len = ele->GetArrayLength();
         Int_t size = cle->Size();
         char* r = eaddr + (size * (len - 1));
         for (Int_t j = len - 1; j >= 0; --j, r -= size) {
            cle->Destructor(r, kTRUE);
         }
      }
   } // iter over elements

   if (!dtorOnly) {
      delete[] p;
   }
   --fLiveCount;
}   

//______________________________________________________________________________
void TStreamerInfo::Destructor(void* obj, Bool_t dtorOnly)
{
   // Emulated destructor for this class.
   // An emulated object is destroyed at address p.
   // Destruct each of the datamembers in the same order
   // as the implicit destructor would.

   // Do nothing if passed a null pointer.
   if (obj == 0) return;

   char* p = (char*) obj;

   if (!dtorOnly && fNVirtualInfoLoc) {
      // !dtorOnly is used to filter out the case where this is called for
      // a base class or embeded object of the outer most class.
      TStreamerInfo *allocator = *(TStreamerInfo**)(p + fVirtualInfoLoc[0]);
      if (allocator != this) {

         Int_t baseoffset = allocator->GetClass()->GetBaseClassOffset(GetClass());

         p -= baseoffset;
         allocator->DestructorImpl(p, kFALSE);
         return;
      }
   }
   DestructorImpl(p, dtorOnly);
}

//______________________________________________________________________________
void TStreamerInfo::DeleteArray(void* ary, Bool_t dtorOnly)
{
   // Destroy an array of emulated objects, with optional delete.

   // Do nothing if passed a null pointer.
   if (ary == 0) return;

   //???FIX ME: What about varying length arrays?

   Long_t* r = (Long_t*) ary;
   Long_t arrayLen = r[-1];
   Long_t size = r[-2];
   char* memBegin = (char*) &r[-2];

   char* p = ((char*) ary) + ((arrayLen - 1) * size);
   for (Long_t cnt = 0; cnt < arrayLen; ++cnt, p -= size) {
      // Destroy each element, but do not delete it.
      Destructor(p, kTRUE);
   } // for arrayItemSize

   if (!dtorOnly) {
      delete[] memBegin;
   }
}

//______________________________________________________________________________
void TStreamerInfo::PrintValue(const char *name, char *pointer, Int_t i, Int_t len, Int_t lenmax) const
{
   //  print value of element i in object at pointer
   //  The function may be called in two ways:
   //    -method1  len < 0
   //           i is assumed to be the TStreamerElement number i in StreamerInfo
   //    -method2  len >= 0
   //           i is the type
   //           address of variable is directly pointer.
   //           len is the number of elements to be printed starting at pointer.

   char *ladd;
   Int_t atype,aleng;
   printf(" %-15s = ",name);

   TStreamerElement * aElement  = 0;
   Int_t *count  = 0;
   if (len >= 0) {
      ladd  = pointer;
      atype = i;
      aleng = len;
   } else        {
      if (i < 0) {
         if (pointer==0) {
            printf("NULL\n");
         } else {
            static TClassRef stringClass("string");
            if (fClass == stringClass) {
               std::string *st = (std::string*)(pointer);
               printf("%s\n",st->c_str());               
            } else if (fClass == TString::Class()) {
               TString *st = (TString*)(pointer);
               printf("%s\n",st->Data());               
            } else {
               printf("(%s*)0x%lx\n",GetName(),(ULong_t)pointer);
            }
         }
         return;
      }
      ladd  = pointer + fOffset[i];
      atype = fNewType[i];
      aleng = fLength[i];
      aElement  = (TStreamerElement*)fElem[i];
      count = (Int_t*)(pointer+fMethod[i]);
   }
   if (aleng > lenmax) aleng = lenmax;

   PrintValueAux(ladd,atype,aElement,aleng,count);
   printf("\n");
}

//______________________________________________________________________________
void TStreamerInfo::PrintValueClones(const char *name, TClonesArray *clones, Int_t i, Int_t eoffset, Int_t lenmax) const
{
   //  print value of element i in a TClonesArray

   if (!clones) {printf(" %-15s = \n",name); return;}
   printf(" %-15s = ",name);
   Int_t nc = clones->GetEntriesFast();
   if (nc > lenmax) nc = lenmax;

   Int_t offset = eoffset + fOffset[i];
   TStreamerElement *aElement  = (TStreamerElement*)fElem[i];
   int aleng = fLength[i];
   if (aleng > lenmax) aleng = lenmax;

   for (Int_t k=0;k < nc;k++) {
      char *pointer = (char*)clones->UncheckedAt(k);
      char *ladd = pointer+offset;
      Int_t *count = (Int_t*)(pointer+fMethod[i]);
      PrintValueAux(ladd,fNewType[i],aElement, aleng, count);
      if (k < nc-1) printf(", ");
   }
   printf("\n");
}

//______________________________________________________________________________
void TStreamerInfo::PrintValueSTL(const char *name, TVirtualCollectionProxy *cont, Int_t i, Int_t eoffset, Int_t lenmax) const
{
   //  print value of element i in a TClonesArray

   if (!cont) {printf(" %-15s = \n",name); return;}
   printf(" %-15s = ",name);
   Int_t nc = cont->Size();
   if (nc > lenmax) nc = lenmax;

   Int_t offset = eoffset + fOffset[i];
   TStreamerElement *aElement  = (TStreamerElement*)fElem[i];
   int aleng = fLength[i];
   if (aleng > lenmax) aleng = lenmax;

   for (Int_t k=0;k < nc;k++) {
      char *pointer = (char*)cont->At(k);
      char *ladd = pointer+offset;
      Int_t *count = (Int_t*)(pointer+fMethod[i]);
      PrintValueAux(ladd,fNewType[i],aElement, aleng, count);
      if (k < nc-1) printf(", ");
   }
   printf("\n");
}

//______________________________________________________________________________
void TStreamerInfo::Streamer(TBuffer &R__b)
{
   // Stream an object of class TStreamerInfo.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      fOldVersion = R__v;
      if (R__v > 1) {
         //R__b.ReadClassBuffer(TStreamerInfo::Class(), this, R__v, R__s, R__c);
         R__b.ClassBegin(TStreamerInfo::Class(), R__v);
         R__b.ClassMember("TNamed");
         TNamed::Streamer(R__b);
         fName = TClassEdit::GetLong64_Name( fName.Data() ).c_str();
         R__b.ClassMember("fCheckSum","UInt_t");
         R__b >> fCheckSum;
         R__b.ClassMember("fClassVersion","Int_t");
         R__b >> fClassVersion;
         fOnFileClassVersion = fClassVersion;
         R__b.ClassMember("fElements","TObjArray*"); 	 
         R__b >> fElements;
         R__b.ClassEnd(TStreamerInfo::Class());
         R__b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
         ResetBit(kIsCompiled);
         return;
      }
      //====process old versions before automatic schema evolution
      TNamed::Streamer(R__b);
      fName = TClassEdit::GetLong64_Name( fName.Data() ).c_str();
      R__b >> fCheckSum;
      R__b >> fClassVersion;
      fOnFileClassVersion = fClassVersion;
      R__b >> fElements;
      R__b.CheckByteCount(R__s, R__c, TStreamerInfo::IsA());
   } else {
      R__c = R__b.WriteVersion(TStreamerInfo::IsA(), kTRUE);
      R__b.ClassBegin(TStreamerInfo::Class());
      R__b.ClassMember("TNamed");
      TNamed::Streamer(R__b);
      R__b.ClassMember("fCheckSum","UInt_t");
      R__b << fCheckSum;
      R__b.ClassMember("fClassVersion","Int_t");
      R__b << ((fClassVersion > 0) ? fClassVersion : -fClassVersion);

      //------------------------------------------------------------------------
      // Stream only non-artificial streamer elements
      //------------------------------------------------------------------------
      R__b.ClassMember("fElements","TObjArray*");
#if NOTYET
      if (has_no_artificial_member) {
         R__b << fElements;
      } else 
#endif
      { 
         R__LOCKGUARD(gCINTMutex);
         Int_t nobjects = fElements->GetEntriesFast();
         TObjArray store( *fElements );
         TStreamerElement *el;
         for (Int_t i = 0; i < nobjects; i++) {
            el = (TStreamerElement*)fElements->UncheckedAt(i);
            if( el != 0 && (el->IsA() == TStreamerArtificial::Class() || el->TestBit(TStreamerElement::kRepeat))) {
               fElements->RemoveAt( i );
            }
         }
         fElements->Compress();
         R__b << fElements;
         R__ASSERT(!fElements->IsOwner());
         *fElements = store;
      }
      R__b.ClassEnd(TStreamerInfo::Class());
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//______________________________________________________________________________
void TStreamerInfo::TagFile(TFile *file)
{
   // Mark the classindex of the current file as using this TStreamerInfo
   // This function is deprecated and its functionality is now done by
   // the overloads of TBuffer::TagStreamerInfo.

   if (file) {
      static Bool_t onlyonce = kFALSE;
      if (!onlyonce) {
         Warning("TagFile","This function is deprecated, use TBuffer::TagStreamerInfo instead");
         onlyonce = kTRUE;
      }
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
#ifdef DOLOOP
#undef DOLOOP
#endif
#define DOLOOP for (k = 0, pointer = arr[0]; k < narr; pointer = arr[++k])

namespace {
   static void PrintCR(int j,Int_t aleng, UInt_t ltype)
   {
      if (j == aleng-1) printf("\n");
      else {
         printf(", ");
         if (j%ltype == ltype-1) printf("\n                    ");
      }
   }
}

//______________________________________________________________________________
void TStreamerInfo::PrintValueAux(char *ladd, Int_t atype, TStreamerElement *aElement, Int_t aleng, Int_t *count)
{
   //  print value of element  in object at pointer, type atype, leng aleng or *count
   //  The function may be called in two ways:
   //    -method1  len < 0
   //           i is assumed to be the TStreamerElement number i in StreamerInfo
   //    -method2  len >= 0
   //           i is the type
   //           address of variable is directly pointer.
   //           len is the number of elements to be printed starting at pointer.
   int j;

   //assert(!((kOffsetP + kChar) <= atype && atype <= (kOffsetP + kBool) && count == 0));
   switch (atype) {
      // basic types
      case kBool:              {Bool_t    *val = (Bool_t*   )ladd; printf("%d" ,*val);  break;}
      case kChar:              {Char_t    *val = (Char_t*   )ladd; printf("%d" ,*val);  break;}
      case kShort:             {Short_t   *val = (Short_t*  )ladd; printf("%d" ,*val);  break;}
      case kInt:               {Int_t     *val = (Int_t*    )ladd; printf("%d" ,*val);  break;}
      case kLong:              {Long_t    *val = (Long_t*   )ladd; printf("%ld",*val);  break;}
      case kLong64:            {Long64_t  *val = (Long64_t* )ladd; printf("%lld",*val);  break;}
      case kFloat:             {Float_t   *val = (Float_t*  )ladd; printf("%f" ,*val);  break;}
      case kFloat16:           {Float_t   *val = (Float_t*  )ladd; printf("%f" ,*val);  break;}
      case kDouble:            {Double_t  *val = (Double_t* )ladd; printf("%g" ,*val);  break;}
      case kDouble32:          {Double_t  *val = (Double_t* )ladd; printf("%g" ,*val);  break;}
      case kUChar:             {UChar_t   *val = (UChar_t*  )ladd; printf("%u" ,*val);  break;}
      case kUShort:            {UShort_t  *val = (UShort_t* )ladd; printf("%u" ,*val);  break;}
      case kUInt:              {UInt_t    *val = (UInt_t*   )ladd; printf("%u" ,*val);  break;}
      case kULong:             {ULong_t   *val = (ULong_t*  )ladd; printf("%lu",*val);  break;}
      case kULong64:           {ULong64_t *val = (ULong64_t*)ladd; printf("%llu",*val);  break;}
      case kBits:              {UInt_t    *val = (UInt_t*   )ladd; printf("%d" ,*val);  break;}

         // array of basic types  array[8]
      case kOffsetL + kBool:    {Bool_t    *val = (Bool_t*   )ladd; for(j=0;j<aleng;j++) { printf("%c " ,val[j]); PrintCR(j,aleng,20); } break;}
      case kOffsetL + kChar:    {Char_t    *val = (Char_t*   )ladd; for(j=0;j<aleng;j++) { printf("%c " ,val[j]); PrintCR(j,aleng,20); } break;}
      case kOffsetL + kShort:   {Short_t   *val = (Short_t*  )ladd; for(j=0;j<aleng;j++) { printf("%d " ,val[j]); PrintCR(j,aleng,10); } break;}
      case kOffsetL + kInt:     {Int_t     *val = (Int_t*    )ladd; for(j=0;j<aleng;j++) { printf("%d " ,val[j]); PrintCR(j,aleng,10); } break;}
      case kOffsetL + kLong:    {Long_t    *val = (Long_t*   )ladd; for(j=0;j<aleng;j++) { printf("%ld ",val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kLong64:  {Long64_t  *val = (Long64_t* )ladd; for(j=0;j<aleng;j++) { printf("%lld ",val[j]);PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kFloat:   {Float_t   *val = (Float_t*  )ladd; for(j=0;j<aleng;j++) { printf("%f " ,val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kFloat16: {Float_t   *val = (Float_t*  )ladd; for(j=0;j<aleng;j++) { printf("%f " ,val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kDouble:  {Double_t  *val = (Double_t* )ladd; for(j=0;j<aleng;j++) { printf("%g " ,val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kDouble32:{Double_t  *val = (Double_t* )ladd; for(j=0;j<aleng;j++) { printf("%g " ,val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kUChar:   {UChar_t   *val = (UChar_t*  )ladd; for(j=0;j<aleng;j++) { printf("%u " ,val[j]); PrintCR(j,aleng,20); } break;}
      case kOffsetL + kUShort:  {UShort_t  *val = (UShort_t* )ladd; for(j=0;j<aleng;j++) { printf("%u " ,val[j]); PrintCR(j,aleng,10); } break;}
      case kOffsetL + kUInt:    {UInt_t    *val = (UInt_t*   )ladd; for(j=0;j<aleng;j++) { printf("%u " ,val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kULong:   {ULong_t   *val = (ULong_t*  )ladd; for(j=0;j<aleng;j++) { printf("%lu ",val[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kULong64: {ULong64_t *val = (ULong64_t*)ladd; for(j=0;j<aleng;j++) { printf("%llu ",val[j]);PrintCR(j,aleng, 5); } break;}
      case kOffsetL + kBits:    {UInt_t    *val = (UInt_t*   )ladd; for(j=0;j<aleng;j++) { printf("%d " ,val[j]); PrintCR(j,aleng, 5); } break;}

         // pointer to an array of basic types  array[n]
      case kOffsetP + kBool:    {Bool_t   **val = (Bool_t**  )ladd; for(j=0;j<*count;j++) { printf("%d " ,(*val)[j]);  PrintCR(j,aleng,20); } break;}
      case kOffsetP + kChar:    {Char_t   **val = (Char_t**  )ladd; for(j=0;j<*count;j++) { printf("%d " ,(*val)[j]);  PrintCR(j,aleng,20); } break;}
      case kOffsetP + kShort:   {Short_t  **val = (Short_t** )ladd; for(j=0;j<*count;j++) { printf("%d " ,(*val)[j]);  PrintCR(j,aleng,10); } break;}
      case kOffsetP + kInt:     {Int_t    **val = (Int_t**   )ladd; for(j=0;j<*count;j++) { printf("%d " ,(*val)[j]);  PrintCR(j,aleng,10); } break;}
      case kOffsetP + kLong:    {Long_t   **val = (Long_t**  )ladd; for(j=0;j<*count;j++) { printf("%ld ",(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kLong64:  {Long64_t **val = (Long64_t**)ladd; for(j=0;j<*count;j++) { printf("%lld ",(*val)[j]); PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kFloat:   {Float_t  **val = (Float_t** )ladd; for(j=0;j<*count;j++) { printf("%f " ,(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kFloat16: {Float_t  **val = (Float_t** )ladd; for(j=0;j<*count;j++) { printf("%f " ,(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kDouble:  {Double_t **val = (Double_t**)ladd; for(j=0;j<*count;j++) { printf("%g " ,(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kDouble32:{Double_t **val = (Double_t**)ladd; for(j=0;j<*count;j++) { printf("%g " ,(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kUChar:   {UChar_t  **val = (UChar_t** )ladd; for(j=0;j<*count;j++) { printf("%u " ,(*val)[j]);  PrintCR(j,aleng,20); } break;}
      case kOffsetP + kUShort:  {UShort_t **val = (UShort_t**)ladd; for(j=0;j<*count;j++) { printf("%u " ,(*val)[j]);  PrintCR(j,aleng,10); } break;}
      case kOffsetP + kUInt:    {UInt_t   **val = (UInt_t**  )ladd; for(j=0;j<*count;j++) { printf("%u " ,(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kULong:   {ULong_t  **val = (ULong_t** )ladd; for(j=0;j<*count;j++) { printf("%lu ",(*val)[j]);  PrintCR(j,aleng, 5); } break;}
      case kOffsetP + kULong64: {ULong64_t**val = (ULong64_t**)ladd; for(j=0;j<*count;j++){ printf("%llu ",(*val)[j]); PrintCR(j,aleng, 5); } break;}

         // array counter //[n]
      case kCounter:            {Int_t *val    = (Int_t*)ladd;    printf("%d",*val);  break;}
         // char *
      case kCharStar:{
         char **val = (char**)ladd;
         if (*val) printf("%s",*val);
         break;
      }
      // Class *  derived from TObject with comment field  //->
      case kObjectp: {
         TObject **obj = (TObject**)(ladd);
         TStreamerObjectPointer *el = (TStreamerObjectPointer*)aElement;
         printf("(%s*)%lx",el ? el->GetClass()->GetName() : "unknown_type",(Long_t)(*obj));
         break;
      }

      // Class*   derived from TObject
      case kObjectP: {
         TObject **obj = (TObject**)(ladd);
         TStreamerObjectPointer *el = (TStreamerObjectPointer*)aElement;
         printf("(%s*)%lx",el ? el->GetClass()->GetName() : "unknown_type",(Long_t)(*obj));
         break;
      }

      // Class    derived from TObject
      case kObject:  {
         TObject *obj = (TObject*)(ladd);
         printf("%s",obj->GetName());
         break;
      }

      // Special case for TString, TObject, TNamed
      case kTString: {
         TString *st = (TString*)(ladd);
         printf("%s",st->Data());
         break;
      }
      case kTObject: {
         TObject *obj = (TObject*)(ladd);
         printf("%s",obj->GetName());
         break;
      }
      case kTNamed:  {
         TNamed *named = (TNamed*) (ladd);
         printf("%s/%s",named->GetName(),named->GetTitle());
         break;
      }

      // Class *  not derived from TObject with comment field  //->
      case kAnyp:    {
         TObject **obj = (TObject**)(ladd);
         TStreamerObjectAnyPointer *el = (TStreamerObjectAnyPointer*)aElement;
         printf("(%s*)0x%lx",el ? el->GetClass()->GetName() : "unknown_type",(Long_t)(*obj));
         break;
      }

      // Class*   not derived from TObject
      case kAnyP:    {
         TObject **obj = (TObject**)(ladd);
         TStreamerObjectAnyPointer *el = (TStreamerObjectAnyPointer*)aElement;
         printf("(%s*)0x%lx",el ? el->GetClass()->GetName() : "unknown_type",(Long_t)(*obj));
         break;
      }
      // Any Class not derived from TObject
      case kOffsetL + kObjectp:
      case kOffsetL + kObjectP:
      case kAny:     {
         printf("printing kAny case (%d)",atype);
//         if (aElement) {
//            TMemberStreamer *pstreamer = aElement->GetStreamer();
//            if (pstreamer == 0) {
//               //printf("ERROR, Streamer is null\n");
//               //aElement->ls();
//               break;
//            }
//            //(*pstreamer)(b,ladd,0);
//         }
         break;
      }
      // Base Class
      case kBase:    {
         printf("printing kBase case (%d)",atype);
         //aElement->ReadBuffer(b,pointer);
         break;
      }

      case kOffsetL + kObject:
      case kOffsetL + kTString:
      case kOffsetL + kTObject:
      case kOffsetL + kTNamed:
      case kStreamer: {
         printf("printing kStreamer case (%d)",atype);
//         TMemberStreamer *pstreamer = aElement->GetStreamer();
//         if (pstreamer == 0) {
//            //printf("ERROR, Streamer is null\n");
//            //aElement->ls();
//            break;
//         }
//         //UInt_t start,count;
//         //b.ReadVersion(&start, &count);
//         //(*pstreamer)(b,ladd,0);
//         //b.CheckByteCount(start,count,IsA());
         break;
      }

      case kStreamLoop: {
         printf("printing kStreamLoop case (%d)",atype);
//         TMemberStreamer *pstreamer = aElement->GetStreamer();
//         if (pstreamer == 0) {
//            //printf("ERROR, Streamer is null\n");
//            //aElement->ls();
//            break;
//         }
         //Int_t *counter = (Int_t*)(count);
         //UInt_t start,count;
         ///b.ReadVersion(&start, &count);
         //(*pstreamer)(b,ladd,*counter);
         //b.CheckByteCount(start,count,IsA());
         break;
      }
      case kSTL: {
         if (aElement) {
            static TClassRef stringClass("string");
            if (ladd && aElement->GetClass() == stringClass) {
               std::string *st = (std::string*)(ladd);
               printf("%s",st->c_str());
            } else {
               printf("(%s*)0x%lx",aElement->GetClass()->GetName(),(Long_t)(ladd));
            }
         } else {
            printf("(unknown_type*)0x%lx",(Long_t)(ladd));
         }
         break;
      }   
   }
}

//______________________________________________________________________________
void TStreamerInfo::Update(const TClass *oldcl, TClass *newcl)
{
   //function called by the TClass constructor when replacing an emulated class
   //by the real class

   TStreamerElement *element;
   TIter nextElement(GetElements());
   while ((element = (TStreamerElement*)nextElement())) {
      element->Update(oldcl,newcl);
   }
   for (Int_t i=0;i < fNdata;i++) {
      fComp[i].Update(oldcl,newcl);
   }
}

//______________________________________________________________________________
void TStreamerInfo::TCompInfo::Update(const TClass *oldcl, TClass *newcl)
{
   // Update the TClass pointer cached in this object.

   if (fClass == oldcl)
      fClass = newcl;
   else if (fClass == 0)
      fClass =TClass::GetClass(fClassName);
}


//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
TVirtualCollectionProxy*
TStreamerInfo::GenEmulatedProxy(const char* class_name)
{
   // Generate emulated collection proxy for a given class.

   return TCollectionProxyFactory::GenEmulatedProxy(class_name);
}

//______________________________________________________________________________
TClassStreamer*
TStreamerInfo::GenEmulatedClassStreamer(const char* class_name)
{
   // Generate emulated class streamer for a given collection class.

   return TCollectionProxyFactory::GenEmulatedClassStreamer(class_name);
}

//______________________________________________________________________________
TVirtualCollectionProxy*
TStreamerInfo::GenExplicitProxy( const ::ROOT::TCollectionProxyInfo &info, TClass *cl )
{
   // Generate proxy from static functions.

   return TCollectionProxyFactory::GenExplicitProxy(info, cl);
}

//______________________________________________________________________________
TClassStreamer*
TStreamerInfo::GenExplicitClassStreamer( const ::ROOT::TCollectionProxyInfo &info, TClass *cl )
{
   // Generate class streamer from static functions.

   return TCollectionProxyFactory::GenExplicitClassStreamer(info, cl);
}
