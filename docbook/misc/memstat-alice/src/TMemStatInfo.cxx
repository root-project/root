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
.     TMemStatInfoStamp - counter
                   fTotalAllocCount;  //total number of allocation for stack sequence
                   fTotalAllocSize;   //total size of allocated memory
                   fAllocCount;       //current number of allocation-deallocation
                   fAllocSize;        //current allocated size
     TMemStatCodeInfo
     base code information - strings - fFunction and fLib (function description)
                          - stamps  -
     TMemStatInfoStamp  fLastStamp;     // last time stamp info
     TMemStatInfoStamp  fCurrentStamp;  // current  time stamp info
     TMemStatInfoStamp  fMaxStampSize;  // max current size stamp
     TMemStatInfoStamp  fMaxStamp;      // max current size stamp
     TMemStatStackInfo
     base stack information - array of pointers to Code informations
                          - stamps  -
     TMemStatInfoStamp  fLastStamp;     // last time stamp info
     TMemStatInfoStamp  fCurrentStamp;  // current  time stamp info
     TMemStatInfoStamp  fMaxStampSize;  // max current size stamp
     TMemStatInfoStamp  fMaxStamp;      // max current size stamp
*/
//****************************************************************************//

// ROOT
#include "Riostream.h"
#include "TObject.h"
// Memstat
#include "TMemStatInfo.h"
#include "TMemStatManager.h"
#include "TMemStatDepend.h"


ClassImp(TMemStatCodeInfo)
ClassImp(TMemStatInfoStamp)
ClassImp(TMemStatStackInfo)

//****************************************************************************//
//                                 TMemStatInfoStamp
//****************************************************************************//

//______________________________________________________________________________
TMemStatInfoStamp::TMemStatInfoStamp():
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
TMemStatInfoStamp::~TMemStatInfoStamp()
{
}

//______________________________________________________________________________
void TMemStatInfoStamp::Print(Option_t* /*option*/) const
{
   // TODO: Comment me

   cout << *this << endl;
}

//______________________________________________________________________________
std::ostream& operator << (std::ostream &_ostream, const TMemStatInfoStamp &_this)
{
   // TODO: Comment me

   _ostream
   << std::setw(fields_length[1]) << "TotalCount"
   << std::setw(fields_length[2]) << "TotalSize"
   << std::setw(fields_length[3]) << "Count"
   << std::setw(fields_length[4]) << "Size" << std::endl;

   // Setting a bit nicer formating
   // example: instead of 20600000 print 20,600,000
   std::locale loc("");
   std::locale loc_previous = _ostream.imbue(loc);
   _ostream.precision(2);
   _ostream << std::fixed;

   _ostream
   << std::setw(fields_length[1]) << _this.fTotalAllocCount
   << std::setw(fields_length[2]) << Memstat::dig2bytes(_this.fTotalAllocSize)
   << std::setw(fields_length[3]) << _this.fAllocCount
   << std::setw(fields_length[4]) << Memstat::dig2bytes(_this.fAllocSize);

   _ostream.imbue(loc_previous);

   return _ostream;
}

//****************************************************************************//
//                                 TMemStatCodeInfo
//****************************************************************************//

//______________________________________________________________________________
TMemStatCodeInfo::TMemStatCodeInfo():
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
   // a ctor

   fLastStamp.fStampType    = TMemStatInfoStamp::kCode;
   fCurrentStamp.fStampType = TMemStatInfoStamp::kCode;
   fMaxStampSize.fStampType = TMemStatInfoStamp::kCode;
   fMaxStamp.fStampType     = TMemStatInfoStamp::kCode;
}

//______________________________________________________________________________
void TMemStatCodeInfo::Inc(Int_t memSize)
{
   // TODO: Comment me

   fCurrentStamp.Inc(memSize);
   if (fCurrentStamp.fAllocCount > fMaxStamp.fAllocCount)
      fMaxStamp = fCurrentStamp;
   if (fCurrentStamp.fAllocSize > fMaxStampSize.fAllocSize)
      fMaxStampSize = fCurrentStamp;
}

//______________________________________________________________________________
void TMemStatCodeInfo::SetInfo(void *info)
{
   //  Get function's real name from info descriptor

   char *zero = 0;
   fCode = (Long64_t)((char*)info - zero);
   TString strLine;
   TMemStatDepend::GetSymbols(info, fInfo, fLib, fFunction, strLine);
}

//______________________________________________________________________________
void TMemStatCodeInfo::Print(Option_t * /*option*/) const
{
   // TODO: Comment me

   StreemCurrAndMax(cout, *this) << endl;

   cout << fCodeID << "\t" << fInfo.Data() << endl;
   cout << fCodeID << "\t" <<  fLib.Data() << '\t' << fFunction.Data() << endl;
}

//______________________________________________________________________________
void TMemStatCodeInfo::MakeStamp(Int_t stampNumber)
{
   // make time stamp - only if change

   if (fCurrentStamp.Equal(fLastStamp))
      return;

   TMemStatInfoStamp &newStamp = TMemStatManager::GetInstance()->AddStamp();
   fCurrentStamp.fStampNumber = stampNumber;
   newStamp = fCurrentStamp;
   fLastStamp = newStamp;
}

//______________________________________________________________________________
std::ostream& operator << (std::ostream &_ostream, const TMemStatCodeInfo &_this)
{
   // TODO: Comment me
   _ostream
   << _this.fFunction.Data()
   << '\t' << _this.fLib.Data();

   return _ostream;
}


//****************************************************************************//
//                                 Storage of Stack information
//****************************************************************************//

//______________________________________________________________________________
TMemStatStackInfo::TMemStatStackInfo():
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
   // default ctor

   fLastStamp.fStampType    = TMemStatInfoStamp::kStack;
   fCurrentStamp.fStampType = TMemStatInfoStamp::kStack;
   fMaxStampSize.fStampType = TMemStatInfoStamp::kStack;
   fMaxStamp.fStampType     = TMemStatInfoStamp::kStack;
}

//______________________________________________________________________________
void TMemStatStackInfo::Init(int stacksize, void **stackptrs, TMemStatManager *manager, Int_t ID)
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
      TMemStatCodeInfo & cinfo =  manager->GetCodeInfo(stackptrs[i]);
      if (cinfo.fCode == 0)
         cinfo.SetInfo(stackptrs[i]);

      fSymbolIndexes[i] = cinfo.fCodeID;
   }
}

//______________________________________________________________________________
int TMemStatStackInfo::Equal(unsigned int size, void **ptr)
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
Bool_t TMemStatInfoStamp::Equal(TMemStatInfoStamp&stamp)
{
   // TODO: Comment me
   if (fTotalAllocCount != stamp.fTotalAllocCount)
      return kFALSE;
   if (fAllocCount != stamp.fAllocCount)
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
void TMemStatStackInfo::MakeStamp(Int_t stampNumber)
{
   // make time stamp - only if change

   if (fCurrentStamp.Equal(fLastStamp))
      return;

   TMemStatInfoStamp &newStamp = TMemStatManager::GetInstance()->AddStamp();
   fCurrentStamp.fStampNumber = stampNumber;
   newStamp = fCurrentStamp;
   fLastStamp = newStamp;
}

//______________________________________________________________________________
void TMemStatStackInfo::Inc(Int_t memSize, TMemStatManager *manager)
{
   // TODO: Comment me
   fCurrentStamp.Inc(memSize);
   if (fCurrentStamp.fAllocCount > fMaxStamp.fAllocCount)
      fMaxStamp = fCurrentStamp;
   if (fCurrentStamp.fAllocSize > fMaxStampSize.fAllocSize)
      fMaxStampSize = fCurrentStamp;
   for (UInt_t i = 0; i < fSize; ++i)
      manager->fCodeInfoArray[fSymbolIndexes[i]].Inc(memSize);
}

//______________________________________________________________________________
void TMemStatStackInfo::Dec(int memSize, TMemStatManager *manager)
{
   // TODO: Comment me
   if (fCurrentStamp.fAllocCount)
      fCurrentStamp.fAllocCount -= 1;
   fCurrentStamp.fAllocSize  -= memSize;
   for (UInt_t i = 0; i < fSize; ++i)
      manager->fCodeInfoArray[fSymbolIndexes[i]].Dec(memSize);
}

//______________________________________________________________________________
std::ostream& operator << (std::ostream &_ostream, const TMemStatStackInfo &_this)
{
   // TODO: Comment me
   return StreemCurrAndMax(_ostream, _this);
}
