// @(#)root/memstat:$Name$:$Id$
// Author: D.Bertini and M.Ivanov   18/06/2007 -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TMEMSTATINFO
#define ROOT_TMEMSTATINFO


//****************************************************************************//
//
//
//  Memory statistic information
//                   TInfoStamp
//                   TCodeInfo
//                   TStackInfo
//****************************************************************************//

// STD
#include <iosfwd>
#include <iomanip>
#include <sstream>
//ROOT
#include "TROOT.h"
#include "TObject.h"
// Memstat
#include "TMemStatHelpers.h"

class TStackInfo;
class TMemStatManager;



const int fields_length[] = {16, 7, 9, 15, 19, 12, 8};


struct TInfoStamp: public TObject {
   enum StampType { kCode, kStack };
   TInfoStamp();              //stamp of memory usage information
   virtual ~TInfoStamp();
   void        Print(Option_t* option = "") const;
   Bool_t      Equal(TInfoStamp&stamp);
   void     Inc(Int_t memSize);  //increment counters -when memory allocated
   void     Dec(Int_t memSize);  //decrement counters -when memory deallocated
   friend std::ostream& operator<< (std::ostream &_ostream, const TInfoStamp &_this);

   Long64_t    fTotalAllocCount;  //total number of allocation for stack sequence
   Long64_t    fTotalAllocSize;   //total size of allocated memory
   Int_t       fAllocCount;       //current number of allocation-deallocation
   Int_t       fAllocSize;        //current allocated size
   Int_t       fStampNumber;      //stamp number
   Int_t       fID;               //code ID number
   Short_t     fStampType;        //stamp Type

   ClassDef(TInfoStamp, 1) // information about stamps
};


struct TCodeInfo: public TObject {
   TCodeInfo();              // store information about line of code
   void SetInfo(void *info);
   virtual ~TCodeInfo() {
   }
   void MakeStamp(Int_t stampNumber);
   void Print(Option_t* option = "") const;
   void Inc(Int_t memSize);  //increment counters -when memory allocated
   void Dec(Int_t memSize);  //decrement counters -when memory deallocated
   friend std::ostream& operator<< (std::ostream &_ostream, const TCodeInfo &_this);

   TInfoStamp  fLastStamp;     // last time stamp info
   TInfoStamp  fCurrentStamp;  // current  time stamp info
   TInfoStamp  fMaxStampSize;  // max current size stamp
   TInfoStamp  fMaxStamp;      // max current size stamp
   Long64_t    fCode;          //pointer to the code
   TString     fInfo;          //pointer desription
   TString     fFunction;      //function
   TString     fLib;           //library
   UInt_t      fCodeID;        //ID number

   ClassDef(TCodeInfo, 1) // a code information structure
};


struct TStackInfo: public TObject {
   enum {kStackHistorySize = 50};
   UInt_t      fSize;               // size of the stack
   TInfoStamp  fLastStamp;          // last time stamp info
   TInfoStamp  fCurrentStamp;       // current  time stamp info
   TInfoStamp  fMaxStampSize;       // max current size stamp
   TInfoStamp  fMaxStamp;           // max current size stamp
   Int_t       fNextHash;           // index  to the next info for given hash value
   void      **fStackSymbols;       //!Stack Symbols
   UInt_t     *fSymbolIndexes;      //[fSize]symbol indexes
   UInt_t      fStackID;            //ID number

   TStackInfo();
   virtual ~TStackInfo() {}
   void     Init(Int_t stacksize, void **stackptrs,  TMemStatManager *manager, Int_t ID); //initialization
   void     Inc(Int_t memSize, TMemStatManager *manager);  //increment counters -when memory allocated
   void     Dec(Int_t memSize, TMemStatManager *manager);  //decrement counters -when memory deallocated
   ULong_t  Hash() const;
   Int_t    Equal(UInt_t size, void **ptr);
   void    *StackAt(UInt_t i);
   //   TStackInfo *Next();    //index of the next entries
   void     MakeStamp(Int_t stampNumber);
   static inline ULong_t HashStack(UInt_t size, void **ptr) {
      return  TString::Hash(ptr, size*sizeof(void*));
   }
   friend std::ostream& operator << (std::ostream &_ostream, const TStackInfo &_this);

   ClassDef(TStackInfo, 1) // a stack information structure
};


struct TMemInfo {
   void   *fAddress;    //mem address
   size_t  fSize;       //size of the allocated memory
   Int_t   fStackIndex; //index of the stack info
};

struct TMemTable {
   Int_t     fAllocCount;    //number of memory allocation blocks
   Int_t     fMemSize;       //total memory allocated size
   Int_t     fTableSize;     //amount of entries in the below array
   Int_t     fFirstFreeSpot; //where is the first free spot in the leaks array?
   TMemInfo *fLeaks;         //leak table
};

struct TDeleteTable {
   Int_t     fAllocCount;    //how many memory blocks do we have
   Int_t     fTableSize;     //amount of entries in the below array
   TMemInfo *fLeaks;         //leak table
};



inline void TInfoStamp::Inc(int memSize)
{
   fTotalAllocCount += 1;
   fTotalAllocSize  += memSize;
   fAllocCount += 1;
   fAllocSize  += memSize;
}
inline void TInfoStamp::Dec(int memSize)
{
   fAllocCount -= 1;
   fAllocSize  -= memSize;
}
inline void TCodeInfo::Dec(int memSize)
{
   fCurrentStamp.Dec(memSize);
}

inline ULong_t TStackInfo::Hash() const
{
   return HashStack(fSize, fStackSymbols);
}

inline void *TStackInfo::StackAt(UInt_t i)
{
   return i < fSize ? fStackSymbols[i] : 0;
}


//______________________________________________________________________________
template<class T>
std::ostream& StreemCurrAndMax(std::ostream &_ostream, const T &_this)
{
   std::ios::fmtflags old_flags(_ostream.flags(std::ios::left));

   _ostream << "\n"
   << std::setw(fields_length[0]) << ""
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

   _ostream << std::setw(fields_length[0]) << "Current stamp";
   _ostream
   << std::setw(fields_length[1]) << _this.fCurrentStamp.fStampNumber
   << std::setw(fields_length[2]) << _this.fCurrentStamp.fID
   << std::setw(fields_length[3]) << _this.fCurrentStamp.fTotalAllocCount
   << std::setw(fields_length[4]) << Memstat::dig2bytes(_this.fCurrentStamp.fTotalAllocSize)
   << std::setw(fields_length[5]) << _this.fCurrentStamp.fAllocCount
   << std::setw(fields_length[6]) << Memstat::dig2bytes(_this.fCurrentStamp.fAllocSize) << std::endl;

   _ostream << std::setw(fields_length[0]) << "Max Alloc stamp";
   _ostream
   << std::setw(fields_length[1]) << _this.fMaxStamp.fStampNumber
   << std::setw(fields_length[2]) << _this.fMaxStamp.fID
   << std::setw(fields_length[3]) << _this.fMaxStamp.fTotalAllocCount
   << std::setw(fields_length[4]) << Memstat::dig2bytes(_this.fMaxStamp.fTotalAllocSize)
   << std::setw(fields_length[5]) << _this.fMaxStamp.fAllocCount
   << std::setw(fields_length[6]) << Memstat::dig2bytes(_this.fMaxStamp.fAllocSize) << std::endl;

   _ostream << std::setw(fields_length[0]) << "Max Size stamp";
   _ostream
   << std::setw(fields_length[1]) << _this.fMaxStampSize.fStampNumber
   << std::setw(fields_length[2]) << _this.fMaxStampSize.fID
   << std::setw(fields_length[3]) << _this.fMaxStampSize.fTotalAllocCount
   << std::setw(fields_length[4]) << Memstat::dig2bytes(_this.fMaxStampSize.fTotalAllocSize)
   << std::setw(fields_length[5]) << _this.fMaxStampSize.fAllocCount
   << std::setw(fields_length[6]) << Memstat::dig2bytes(_this.fMaxStampSize.fAllocSize);

   _ostream.imbue(loc_previous);
   _ostream.flags(old_flags);

   return _ostream;
}


#endif
