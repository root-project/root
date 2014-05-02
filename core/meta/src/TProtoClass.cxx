// @(#)root/meta:$
// Author: Axel Naumann 2014-05-02

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Persistent version of a TClass.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProtoClass.h"

#include "TBaseClass.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TList.h"
#include "TListOfDataMembers.h"
#include "TRealData.h"

//______________________________________________________________________________
TProtoClass::TProtoClass(TClass* cl):
   TNamed(*cl), fBase(cl->GetListOfBases()), fData(cl->GetListOfDataMembers()),
   fSizeof(cl->Size()), fCanSplit(cl->fCanSplit), fProperty(cl->fProperty)
{
   // Initialize a TProtoClass from a TClass.
   fPRealData = new TList();

   // Build the list of RealData before we access it:
   cl->BuildRealData(0, true /*isTransient*/);
   // The data members are ordered as follows:
   // - this class's data members,
   // - foreach base: base class's data members.
   // fPRealData encodes all TProtoRealData objects with a
   // TObjString to signal a new class.
   TClass* clCurrent = cl;
   for (auto realDataObj: *cl->GetListOfRealData()) {
      TRealData *rd = (TRealData*)realDataObj;
      TClass* clRD = rd->GetDataMember()->GetClass();
      if (clRD != clCurrent) {
         clCurrent = clRD;
         fPRealData->AddLast(new TObjString(clRD->GetName()));
      }
      fPRealData->AddLast(new TProtoRealData(rd));
   }

   cl->CalculateStreamerOffset();
   fOffsetStreamer = cl->fOffsetStreamer;
}

//______________________________________________________________________________
TProtoClass::~TProtoClass()
{
   // Destructor.
   Delete();
}

//______________________________________________________________________________
void TProtoClass::Delete(Option_t* opt /*= ""*/) {
   // Delete the containers that are usually owned by their TClass.
   if (fPRealData) fPRealData->Delete(opt);
   delete fPRealData; fPRealData = 0;
   if (fBase) fBase->Delete(opt);
   delete fBase; fBase = 0;
   if (fData) fData->Delete(opt);
   delete fData; fData = 0;
}

//______________________________________________________________________________
void TProtoClass::FillTClass(TClass* cl) {
   // Move data from this TProtoClass into cl.
   if (cl->fRealData || cl->fBase || cl->fData || cl->fSizeof != -1 || cl->fCanSplit
       || cl->fProperty) {
      Error("AdoptProto", "TClass %s already initialized!", cl->GetName());
      return;
   }
   *((TNamed*)cl) = *this;
   cl->fBase = fBase;
   cl->fData = (TListOfDataMembers*)fData;
   cl->fRealData = new TList(); // FIXME: this should really become a THashList!
   TClass* currentRDClass = cl;
   for (TObject* element: *fPRealData) {
      if (element->IsA() == TObjString::Class()) {
         currentRDClass = TClass::GetClass(element->GetName());
         if (!currentRDClass) {
            Error("TProtoClass::FillTClass()", "Cannot find TClass for %s; skipping its members.",
                  element->GetName());
         }
      } else {
         if (!currentRDClass) continue;
         TProtoRealData* prd = (TProtoRealData*)element;
         if (TRealData* rd = prd->CreateRealData(currentRDClass)) {
            cl->fRealData->AddLast(rd);
         }
      }
   }
   cl->fSizeof = fSizeof;
   cl->fCanSplit = fCanSplit;
   cl->fProperty = fProperty;

   // Update pointers to TClass
   for (auto base: *cl->fBase) {
      ((TBaseClass*)base)->SetClass(cl);
   }
   for (auto dm: *cl->fData) {
      ((TDataMember*)dm)->SetClass(cl);
   }
   ((TListOfDataMembers*)cl->fData)->SetClass(cl);

   fBase = 0;
   fData = 0;
   fPRealData->Delete();
   delete fPRealData;
   fPRealData = 0;
}

//______________________________________________________________________________
TProtoClass::TProtoRealData::TProtoRealData(const TRealData* rd):
   TNamed(rd->GetDataMember()->GetName(), rd->GetName()),
   fOffset(rd->GetThisOffset())
{
   // Initialize this from a TRealData object.
   SetBit(kIsObject, rd->IsObject());
}

//______________________________________________________________________________
TProtoClass::TProtoRealData::~TProtoRealData()
{
   // Destructor to pin vtable.
}

//______________________________________________________________________________
TRealData* TProtoClass::TProtoRealData::CreateRealData(TClass* dmClass) const
{
   // Create a TRealData from this, with its data member coming from dmClass.
   TDataMember* dm = (TDataMember*)dmClass->GetListOfDataMembers()->FindObject(GetName());
   if (!dm) {
      Error("TProtoClass::TProtoRealData::CreateRealData()",
            "Cannot find data member %s::%s!", dmClass->GetName(), GetName());
      return 0;
   }
   TRealData* rd = new TRealData(GetTitle(), fOffset, dm);
   rd->SetIsObject(TestBit(kIsObject));
   return rd;
}
