// @(#)root/memstat:$Name$:$Id$
// Author: D.Bertini and M.Ivanov   18/06/2007 -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemStat
#define ROOT_TMemStat

// STD
#include <memory>
#include <vector>
#include <set>
// ROOT
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TObjString
#include "TObjString.h"
#endif

class TArrayI;
class TBits;
class TTree;
class TMemStatManager;
class TGraph;
class TMemStatCodeInfo;
class TMemStatInfoStamp;

typedef std::vector<UInt_t> UIntVector_t;
typedef std::vector<Int_t> IntVector_t;
typedef std::auto_ptr<TFile> TFilePtr;

class TMemStat: public TObject
{
public:
   typedef std::set<std::string> Selection_t;

   enum ESelection { kFunction, kLibrary };
   enum StatType  { kTotalAllocCount = 0, kAllocCount = 2, kTotalAllocSize = 1, kAllocSize = 3, kUndef = 4};
   enum StampType { kCurrent = 0, kMaxSize = 1, kMaxCount = 2};
   enum OperType { kAND = 0, kOR = 1, kNOT = 2};

private:
   StatType           fSortStat;              // sorting statistic type
   StampType          fSortStamp;             // sorting stamp type
   Double_t           fMaximum;               // maximum value of all graphs
   UInt_t             fSortDeep;              // Deepness of the information to be print - draw
   UInt_t             fStackDeep;             // Deepness of the stack information
   UInt_t             fMaxStringLength;       // max length of information string
   Int_t              fSelected;              // index of selected object
   Bool_t             fIsActive;              // is object attached to MemStat
   Bool_t             fOrder;                 // sorting order
   UIntVector_t       fSelectedCodeIndex;     // vector of indexes of selected items - code
   UIntVector_t       fSelectedStackIndex;    // vector of indexes of selected items - stack
   IntVector_t        fArrayIndexes;          // indexes of selected objects
   TBits*             fSelectedCodeBitmap;    // bitmask   of selected items        - code
   TBits*             fSelectedStackBitmap;   // bitmask   of selected items - stack
   TFilePtr           fFile;                  // current file with information  - stamps
   TObjArray*         fStampArray;            // array of stamp names
   TObjArray*         fArray;                 // array of objects to draw
   TObjArray*         fArrayGraphics;         // array of graphic objects
   TObjArray          fDisablePrintLib;       // disable printing for libraries
   TObjArray          fDisablePrintCode;      // disable printing for libraries
   TString            fOption;                // current options
   TTree*             fTree;                  // current tree with information  - stamps
   TTree*             fTreeSys;               // tree with system information
   TMemStatInfoStamp* fStackSummary;          // summary information for selected stack
   TMemStatManager*   fManager;               // current MemStatManager

private :
   Int_t      DistancetoPrimitive(Int_t px, Int_t py);
   Bool_t     GetMemStat(const char * fname, Int_t entry);
   Bool_t     EnabledCode(const TMemStatCodeInfo &info) const;
   void       ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void       MakeCodeArray();
   TGraph*    MakeGraph(StatType statType, Int_t id, Int_t type, Double_t &xmax, Double_t &ymax);
   TObjArray* MakeGraphCode(StatType statType, Int_t nentries);
   TObjArray* MakeGraphStack(StatType statType, Int_t nentries);
   void       MakeStampsText();
   void       MakeStackArray();
   void       ProcessOption(Option_t *option);
   void       RefreshSelect();

public:
   TMemStat(Option_t* option = "read");
   virtual ~TMemStat();

public:
   void             AddStamp(const char*stampName);
   void             Draw(Option_t *option = "");
   void             GetFillSelection(Selection_t *_Container, ESelection _Selection) const;
   const Option_t*  GetOption() const {
      return fOption.Data();
   }
   TObjArray*       GetStampList();
   UInt_t           GetStackDeep() const
   {
      return fStackDeep;
   }
   UInt_t           GetSortDeep() const
   {
      return fSortDeep;
   }
   UInt_t           GetMaxStringLength() const
   {
      return fMaxStringLength;
   }
   virtual char    *GetObjectInfo(Int_t px, Int_t py) const;
   void             MakeReport(const char * lib = "", const char *fun = "",
                               Option_t* option = NULL, const char *fileName = "");
   void             MakeHisMemoryStamp(Int_t topDiff);
   void             MakeHisMemoryTime();
   void             Paint(Option_t *option = "");
   void             PrintCode(Int_t nentries = 10) const;
   void             PrintCodeWithID(UInt_t index) const;
   void             PrintStack(Int_t nentries = 10, UInt_t deep = 1) const;
   void             PrintStackWithID(UInt_t _id, UInt_t _deep = 0) const;
   void             Report(Option_t* option = "");
   void             ResetSelection();
   void             SetAutoStamp(Int_t autoStampSize = 2000000, Int_t autoStampAlloc = 200000);
   void             SetCurrentStamp(const char *stampName);
   void             SetCurrentStamp(const TObjString &stampName);
   void             SetSortStat(StatType NewVal)
   {
      fSortStat = NewVal;
   }
   void             SetSortStamp(StampType NewVal)
   {
      fSortStamp= NewVal;
   }
   void             SetStackDeep(UInt_t NewVal)
   {
      fStackDeep = NewVal;
   }
   void             SetSortDeep(UInt_t NewVal)
   {
      fSortDeep = NewVal;
   }
   void             SelectCode(const char *contlib = "",
                               const char *contfunction = "", OperType oType = kOR);
   void             SelectStack(OperType oType = kOR);
   void             SortCode(StatType sortType, StampType stampType);
   void             SortStack(StatType sortType, StampType stampType);

   ClassDef(TMemStat, 0) // a user interface class of memstat
};

#endif
