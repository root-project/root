// @(#)root/io:$Id$
// Author: Philippe Canal 05/2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TStreamerInfo.h"
#include "TROOT.h"
#include "TStreamerElement.h"
#include "TVirtualMutex.h"
#include "TObjArray.h"
#include "TInterpreter.h"

static const Int_t kRegrouped = TStreamerInfo::kOffsetL;

//______________________________________________________________________________
void TStreamerInfo::Compile()
{
   // loop on the TStreamerElement list
   // regroup members with same type
   // Store predigested information into local arrays. This saves a huge amount
   // of time compared to an explicit iteration on all elements.

   R__LOCKGUARD(gCINTMutex);

   // fprintf(stderr,"Running Compile for %s %d %d req=%d,%d\n",GetName(),fClassVersion,fOptimized,CanOptimize(),TestBit(kCannotOptimize));

   // if (IsCompiled() && (!fOptimized || (CanOptimize() && !TestBit(kCannotOptimize)))) return;

   fOptimized = kFALSE;
   fNdata = 0;

   TObjArray* infos = (TObjArray*) gROOT->GetListOfStreamerInfo();
   if (fNumber >= infos->GetSize()) {
      infos->AddAtAndExpand(this, fNumber);
   } else {
      if (!infos->At(fNumber)) {
         infos->AddAt(this, fNumber);
      }
   }

   delete[] fType;
   fType = 0;
   delete[] fNewType;
   fNewType = 0;
   delete[] fOffset;
   fOffset = 0;
   delete[] fLength;
   fLength = 0;
   delete[] fElem;
   fElem = 0;
   delete[] fMethod;
   fMethod = 0;
   delete[] fComp;
   fComp = 0;


   Int_t ndata = fElements->GetEntries();

   fOffset = new Int_t[ndata+1];
   fType   = new Int_t[ndata+1];

   SetBit(kIsCompiled);

   if (!ndata) {
      // This may be the case for empty classes (e.g., TAtt3D).
      // We still need to properly set the size of emulated classes (i.e. add the virtual table)
      if (fClass->TestBit(TClass::kIsEmulation) && fNVirtualInfoLoc!=0) {
         fSize = sizeof(TStreamerInfo*);
      }
      return;
   }

   fComp = new TCompInfo[ndata];
   fNewType = new Int_t[ndata];
   fLength = new Int_t[ndata];
   fElem = new ULong_t[ndata];
   fMethod = new ULong_t[ndata];

   TStreamerElement* element;
   TStreamerElement* previous = 0;
   Int_t keep = -1;
   Int_t i;

   if (!CanOptimize()) {
      SetBit(kCannotOptimize);
   }

   Bool_t isOptimized = kFALSE;

   for (i = 0; i < ndata; ++i) {
      element = (TStreamerElement*) fElements->At(i);
      if (!element) {
         break;
      }
      if (element->GetType() < 0) {
         // -- Skip an ignored TObject base class.
         // Note: The only allowed negative value here is -1,
         // and signifies that Build() has found a TObject
         // base class and TClass::IgnoreTObjectStreamer() was
         // called.  In this case the compiled version of the
         // elements omits the TObject base class element,
         // which has to be compensated for by TTree::Bronch()
         // when it is making branches for a split object.
         continue;
      }
      if (TestBit(kCannotOptimize) && element->IsBase()) 
      {
         // Make sure the StreamerInfo for the base class is also
         // not optimized.
         TClass *bclass = element->GetClassPointer();
         Int_t clversion = ((TStreamerBase*)element)->GetBaseVersion();
         TStreamerInfo *binfo = ((TStreamerInfo*)bclass->GetStreamerInfo(clversion));
         binfo->SetBit(kCannotOptimize);
         if (binfo->IsOptimized())
         {
            // Optimizing does not work with splitting.
            binfo->Compile();
         }      
      }
      Int_t asize = element->GetSize();
      if (element->GetArrayLength()) {
         asize /= element->GetArrayLength();
      }
      fType[fNdata] = element->GetType();
      fNewType[fNdata] = element->GetNewType();
      fOffset[fNdata] = element->GetOffset();
      fLength[fNdata] = element->GetArrayLength();
      fElem[fNdata] = (ULong_t) element;
      fMethod[fNdata] = element->GetMethod();
      // try to group consecutive members of the same type
      if (!TestBit(kCannotOptimize) 
          && (keep >= 0) 
          && (element->GetType() < 10) 
          && (fType[fNdata] == fNewType[fNdata]) 
          && (fMethod[keep] == 0) 
          && (element->GetType() > 0) 
          && (element->GetArrayDim() == 0) 
          && (fType[keep] < kObject) 
          && (fType[keep] != kCharStar) /* do not optimize char* */ 
          && (element->GetType() == (fType[keep]%kRegrouped)) 
          && ((element->GetOffset()-fOffset[keep]) == (fLength[keep])*asize)
          && ((fOldVersion<6) || !previous || /* In version of TStreamerInfo less than 6, the Double32_t were merged even if their annotation (aka factor) were different */
              ((element->GetFactor() == previous->GetFactor())
               && (element->GetXmin() == previous->GetXmin())
               && (element->GetXmax() == previous->GetXmax())
              )
             )
         ) 
      {
         if (fLength[keep] == 0) {
            fLength[keep]++;
         }
         fLength[keep]++;
         fType[keep] = element->GetType() + kRegrouped;
         isOptimized = kTRUE;
      } else {
         if (fNewType[fNdata] != fType[fNdata]) {
            if (fNewType[fNdata] > 0) {
               if (fType[fNdata] != kCounter) {
                  fType[fNdata] += kConv;
               }
            } else {
               if (fType[fNdata] == kCounter) {
                  Warning("Compile", "Counter %s should not be skipped from class %s", element->GetName(), GetName());
               }
               fType[fNdata] += kSkip;
            }
         }
         keep = fNdata;
         if (fLength[keep] == 0) {
            fLength[keep] = 1;
         }
         fNdata++;
      }
      previous = element;
   }

   for (i = 0; i < fNdata; ++i) {
      element = (TStreamerElement*) fElem[i];
      if (!element) {
         continue;
      }
      fComp[i].fClass = element->GetClassPointer();
      fComp[i].fNewClass = element->GetNewClass();
      fComp[i].fClassName = TString(element->GetTypeName()).Strip(TString::kTrailing, '*');
      fComp[i].fStreamer = element->GetStreamer();
   }
   ComputeSize();

   fOptimized = isOptimized;

   if (gDebug > 0) {
      ls();
   }
}
