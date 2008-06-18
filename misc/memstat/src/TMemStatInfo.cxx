// @(#)root/new:$Name$:$Id$
// Author: D.Bertini and M.Ivanov   10/08/2000 -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//****************************************************************************//

/*
      Mem Stat information -
.     TInfoStamp - counter
                   fTotalAllocCount;  //total number of allocation for stack sequence
                   fTotalAllocSize;   //total size of allocated memory
                   fAllocCount;       //current number of allocation-deallocation
                   fAllocSize;        //current allocated size
     TCodeInfo
     base code inormation - strings - fFunction and fLib (function desription)
                          - stamps  -
     TInfoStamp  fLastStamp;     // last time stamp info
     TInfoStamp  fCurrentStamp;  // current  time stamp info
     TInfoStamp  fMaxStampSize;  // max current size stamp
     TInfoStamp  fMaxStamp;      // max current size stamp
     TStackInfo
     base stack inormation - array of pointers to Code informations
                          - stamps  -
     TInfoStamp  fLastStamp;     // last time stamp info
     TInfoStamp  fCurrentStamp;  // current  time stamp info
     TInfoStamp  fMaxStampSize;  // max current size stamp
     TInfoStamp  fMaxStamp;      // max current size stamp
*/
//
//****************************************************************************//
// ROOT
#include "Riostream.h"
#include "TObject.h"
// Memstat
#include "TMemStatInfo.h"
#include "TMemStatManager.h"
#include "TMemStatDepend.h"


ClassImp(TCodeInfo)
ClassImp(TInfoStamp)
ClassImp(TStackInfo)

//****************************************************************************//
//                                 TInfoStamp
//****************************************************************************//

//______________________________________________________________________________
TInfoStamp::TInfoStamp():
      fTotalAllocCount(0),  //total number of allocation for stack sequence
      fTotalAllocSize(0),   //total size of allocated memory
      fAllocCount(0),       //current number of allocation-deallocation
      fAllocSize(0),        //current allocated size
      fStampNumber(0),      //index of previous
      fID(0),               //code ID
      fStampType(0)
{
}

//______________________________________________________________________________
TInfoStamp::~TInfoStamp()
{
}

//______________________________________________________________________________
void TInfoStamp::Print(Option_t* /*option*/) const
{
   cout << *this << endl;
}

//______________________________________________________________________________
std::ostream& operator << (std::ostream &_ostream, const TInfoStamp &_this)
{
   _ostream
   << std::setw(fields_length[1]) << "ID"
   << std::setw(fields_length[2]) << "Sort"
   << std::setw(fields_length[3]) << "TotalCount"
   << std::setw(fields_length[4]) << "TotalSize"
   << std::setw(fields_length[5]) << "Count"
   << std::setw(fields_length[6]) << "Size" << std::endl;

   // Setting a bit nicer formating
   // example: instead of 20600000 print 20,600,000
   std::locale loc("");
   std::locale loc_previous = _ostream.imbue(loc);
   _ostream.precision(2);
   _ostream << std::fixed;

   _ostream
   << std::setw(fields_length[1]) << _this.fStampNumber
   << std::setw(fields_length[2]) << _this.fID
   << std::setw(fields_length[3]) << _this.fTotalAllocCount
   << std::setw(fields_length[4]) << Memstat::dig2bytes(_this.fTotalAllocSize)
   << std::setw(fields_length[5]) << _this.fAllocCount
   << std::setw(fields_length[6]) << Memstat::dig2bytes(_this.fAllocSize);

   _ostream.imbue(loc_previous);

   return _ostream;
}

//****************************************************************************//
//                                 TCodeInfo
//****************************************************************************//

//______________________________________________________________________________
TCodeInfo::TCodeInfo():
      fLastStamp(),
      fCurrentStamp(),
      fMaxStampSize(),
      fMaxStamp(),
      fCode(0),
      fInfo(0),
      fFunction(),
      fLib(),
      fCodeID(0)
{
   fLastStamp.fStampType    = TInfoStamp::kCode;
   fCurrentStamp.fStampType = TInfoStamp::kCode;
   fMaxStampSize.fStampType = TInfoStamp::kCode;
   fMaxStamp.fStampType     = TInfoStamp::kCode;
}

//______________________________________________________________________________
void TCodeInfo::Inc(Int_t memSize)
{
   fCurrentStamp.Inc(memSize);
   if (fCurrentStamp.fAllocCount > fMaxStamp.fAllocCount)
      fMaxStamp = fCurrentStamp;
   if (fCurrentStamp.fAllocSize > fMaxStampSize.fAllocSize)
      fMaxStampSize = fCurrentStamp;
}

//______________________________________________________________________________
void TCodeInfo::SetInfo(void *info)
{
   //  Get function realname from info descriptor

   char *zero = 0;
   fCode = (Long64_t)((char*)info - zero);
   TString strLine;
   TMemStatDepend::GetSymbols(info, fInfo, fLib, fFunction, strLine);
}

//______________________________________________________________________________
void TCodeInfo::Print(Option_t * /*option*/) const
{
   StreemCurrAndMax(cout, *this) << endl;

   cout << fCodeID << "\t" << fInfo.Data() << endl;
   cout << fCodeID << "\t" <<  fLib.Data() << '\t' << fFunction.Data() << endl;
}

//______________________________________________________________________________
void TCodeInfo::MakeStamp(Int_t stampNumber)
{
   // make time stamp - only if change

   if (fCurrentStamp.Equal(fLastStamp))
      return;

   TInfoStamp &newStamp = TMemStatManager::GetInstance()->AddStamp();
   fCurrentStamp.fStampNumber = stampNumber;
   newStamp = fCurrentStamp;
   fLastStamp = newStamp;
}

//______________________________________________________________________________
std::ostream& operator << (std::ostream &_ostream, const TCodeInfo &_this)
{
   _ostream
   << _this.fFunction.Data()
   << '\t' << _this.fLib.Data();

   return _ostream;
}


//****************************************************************************//
//                                 Storage of Stack information
//****************************************************************************//

//______________________________________________________________________________
TStackInfo::TStackInfo():
      TObject(),
      fSize(0),
      fLastStamp(),
      fCurrentStamp(),
      fMaxStampSize(),
      fMaxStamp(),
      fNextHash(-1),
      fStackSymbols(0),
      fSymbolIndexes(0),
      fStackID(0)
{
   fLastStamp.fStampType    = TInfoStamp::kStack;
   fCurrentStamp.fStampType = TInfoStamp::kStack;
   fMaxStampSize.fStampType = TInfoStamp::kStack;
   fMaxStamp.fStampType     = TInfoStamp::kStack;
}

//______________________________________________________________________________
void TStackInfo::Init(int stacksize, void **stackptrs, TMemStatManager *manager, Int_t ID)
{
   //Initialize the stack
   fStackID = ID;
   fSize = stacksize;
   fLastStamp.fID   = fStackID;     // last time stamp info
   fCurrentStamp.fID = fStackID;    // current  time stamp info

   fStackSymbols  = new void*[stacksize];
   memcpy(fStackSymbols, stackptrs, stacksize * sizeof(void *));
   fSymbolIndexes = new UInt_t[stacksize];

   for (Int_t i = 0; i < stacksize; ++i) {
      TCodeInfo & cinfo =  manager->GetCodeInfo(stackptrs[i]);
      if (cinfo.fCode == 0)
         cinfo.SetInfo(stackptrs[i]);

      fSymbolIndexes[i] = cinfo.fCodeID;
   }
}

//______________________________________________________________________________
int TStackInfo::Equal(unsigned int size, void **ptr)
{
   // Return 0 if stack information not equal otherwise return 1.

   if (size != fSize)
      return 0;
   for (size_t i = 0; i < size; ++i)
      if (ptr[i] != fStackSymbols[i])
         return 0;
   return 1;
}

//______________________________________________________________________________
Bool_t TInfoStamp::Equal(TInfoStamp&stamp)
{
   if (fTotalAllocCount != stamp.fTotalAllocCount)
      return kFALSE;
   if (fAllocCount != stamp.fAllocCount)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
void TStackInfo::MakeStamp(Int_t stampNumber)
{
   // make time stamp - only if change

   if (fCurrentStamp.Equal(fLastStamp))
      return;

   TInfoStamp &newStamp = TMemStatManager::GetInstance()->AddStamp();
   fCurrentStamp.fStampNumber = stampNumber;
   newStamp = fCurrentStamp;
   fLastStamp = newStamp;
}

//______________________________________________________________________________
void TStackInfo::Inc(Int_t memSize, TMemStatManager *manager)
{
   fCurrentStamp.Inc(memSize);
   if (fCurrentStamp.fAllocCount > fMaxStamp.fAllocCount)
      fMaxStamp = fCurrentStamp;
   if (fCurrentStamp.fAllocSize > fMaxStampSize.fAllocSize)
      fMaxStampSize = fCurrentStamp;
   for (UInt_t i = 0; i < fSize; ++i)
      manager->fCodeInfoArray[fSymbolIndexes[i]].Inc(memSize);
}

//______________________________________________________________________________
void TStackInfo::Dec(int memSize, TMemStatManager *manager)
{
   if (fCurrentStamp.fAllocCount)
      fCurrentStamp.fAllocCount -= 1;
   fCurrentStamp.fAllocSize  -= memSize;
   for (UInt_t i = 0; i < fSize; ++i)
      manager->fCodeInfoArray[fSymbolIndexes[i]].Dec(memSize);
}

//______________________________________________________________________________
std::ostream& operator << (std::ostream &_ostream, const TStackInfo &_this)
{
   return StreemCurrAndMax(_ostream, _this);
}
