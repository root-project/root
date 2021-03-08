// @(#)root/meta:$Id$
// Author: Rene Brun   05/03/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRealData.h"
#include "TDataMember.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TStreamer.h"

ClassImp(TRealData);

/** \class TRealData
The TRealData class manages the effective list of all data members
for a given class. For example for an object of class TLine that inherits
from TObject and TAttLine, the TRealData object for a line contains the
complete list of all data members of the 3 classes.

The list of TRealData members in TClass is built when functions like
object.Inspect or object.DrawClass are called.
*/

////////////////////////////////////////////////////////////////////////////////
/// RealData default constructor.

TRealData::TRealData() : TObject(), fDataMember(nullptr), fThisOffset(-1),
   fStreamer(nullptr), fIsObject(kFALSE)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor to define one persistent data member.
/// datamember is the pointer to the data member descriptor.

TRealData::TRealData(const char *name, Long_t offset, TDataMember *datamember)
   : TObject(), fDataMember(datamember), fThisOffset(offset), fName(name),
     fStreamer(nullptr), fIsObject(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// RealData default destructor.

TRealData::~TRealData()
{
   delete fStreamer;
}

////////////////////////////////////////////////////////////////////////////////
/// Write one persistent data member on output buffer.
/// pointer points to the current persistent data member

void TRealData::WriteRealData(void *, char *&)
{
}

////////////////////////////////////////////////////////////////////////////////

void TRealData::AdoptStreamer(TMemberStreamer *str)
{
// fDataMember->SetStreamer(str);
// delete fStreamer;
   fStreamer = str;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the name of the data member as represented in the list of
/// real data.

void TRealData::GetName(TString &output, TDataMember *dm)
{
   output.Clear();
   const char* dmType  = dm->GetTypeName();
   if (TClassEdit::IsStdArray(dmType)) {
      std::string typeNameBuf;
      Int_t ndim = dm->GetArrayDim();
      std::array<Int_t, 5> maxIndices; // 5 is the maximum supported in TStreamerElement::SetMaxIndex
      TClassEdit::GetStdArrayProperties(dmType,
                                        typeNameBuf,
                                        maxIndices,
                                        ndim);
      output = dm->GetName();
      for (Int_t idim = 0; idim < ndim; ++idim) {
         output += TString::Format("[%d]",maxIndices[idim] );
      }
      return;
   }

   // keep an empty name if data member is not found
   if (dm) output = dm->GetName();
   if (dm->IsaPointer())
      output = TString("*")+output;
   else {
      if (dm && dm->GetArrayDim() > 0) {
         // in case of array (like fMatrix[2][2] we need to add max index )
         // this only in case of it os not a pointer
         for (int idim = 0; idim < dm->GetArrayDim(); ++idim)
            output += TString::Format("[%d]",dm->GetMaxIndex(idim) );
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Return the associate streamer object.

TMemberStreamer *TRealData::GetStreamer() const
{
   return fStreamer; // return fDataMember->GetStreamer();
}

