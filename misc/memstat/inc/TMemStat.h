// @(#)root/memstat:$Name$:$Id$
// Author: D.Bertini and M.Ivanov   18/06/2007 -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TMEMSTAT
#define ROOT_TMEMSTAT

// STD
#include <memory>
#include <vector>
#include <set>
// ROOT
#include "TString.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TObjString.h"

class TArrayI;
class TBits;
class TTree;
class TMemStatManager;
class TGraph;
class TCodeInfo;
class TStackInfo;
class TInfoStamp;

typedef std::vector<UInt_t> UIntVector_t;
typedef std::vector<Int_t> IntVector_t;
typedef std::auto_ptr<TFile> TFilePtr;

class TMemStat: public TObject
{
public:
   typedef std::set<std::string> Selection_t;
   enum ESelection{ kFunction, kLibrary };

   enum StatType  { kTotalAllocCount = 0, kAllocCount = 2, kTotalAllocSize = 1, kAllocSize = 3, kUndef = 4};
   enum StampType { kCurrent = 0, kMaxSize = 1, kMaxCount = 2};
   enum OperType { kAND = 0, kOR = 1, kNOT = 2};

public:
   TMemStat(Option_t* option = "read");
   virtual ~TMemStat();

public:
   void SetAutoStamp(Int_t autoStampSize = 2000000, Int_t autoStampAlloc = 200000);
   void SetCurrentStamp(const char *stampName);
   void SetCurrentStamp(const TObjString &stampName);
   TObjArray * GetStampList();
   void AddStamp(const char*stampName);
   void Report(Option_t* option = "");
   void Paint(Option_t *option = "");
   void Draw(Option_t *option = "");
   void MakeReport(const char * lib = "", const char *fun = "",
                   Option_t* option = NULL, const char *fileName = "");        // make report for library
   void MakeHisMemoryTime();                                                   //draw histogram of memory usage
   void MakeHisMemoryStamp(Int_t topDiff);                                     //draw histogram of memory usage
   Option_t *GetOption() const {
      return fOption.Data();
   }

   void ResetSelection();
   void PrintCodeWithID(UInt_t index) const;
   // if _deep is 0, then we use fStackDeep
   void PrintStackWithID(UInt_t _id, UInt_t _deep = 0) const;
   void SelectCode(const char * contlib = "", const char * contfunction = "", OperType oType = kOR);
   void SelectStack(OperType oType = kOR);
   void SortCode(StatType sortType, StampType stampType);
   void SortStack(StatType sortType, StampType stampType);
   void PrintCode(Int_t nentries = 10) const;
   void PrintStack(Int_t nentries = 10, UInt_t deep = 1) const;
   void GetFillSelection(Selection_t *_Container, ESelection _Selection) const;

public:
   StatType fSortStat;                  // sorting statistic type
   StampType fSortStamp;                // sorting stamp type
   UInt_t fSortDeep;                    // deepnes of the information to be print - draw
   UInt_t fStackDeep;                   // deepnes of the stack information
   Int_t fSelected;                     // index of selected object

private :
   Bool_t GetMemStat(const char * fname, Int_t entry);
   Bool_t EnabledCode(const TCodeInfo &info) const;
   void MakeCodeArray();
   void MakeStackArray();
   void ProcessOption(Option_t *option);
   TGraph* MakeGraph(StatType statType, Int_t id, Int_t type, Double_t &xmax, Double_t &ymax);
   void MakeStampsText();
   void RefreshSelect();
   TObjArray *MakeGraphCode(StatType statType, Int_t nentries);
   TObjArray *MakeGraphStack(StatType statType, Int_t nentries);
   Int_t DistancetoPrimitive(Int_t px, Int_t py);
   void ExecuteEvent(Int_t event, Int_t px, Int_t py);

private:
   Bool_t fIsActive;                    // is object attached to MemStat
   UIntVector_t fSelectedCodeIndex;     // vector of indexes of selected items - code
   UIntVector_t fSelectedStackIndex;    // vector of indexes of selected items - stack
   TBits *fSelectedCodeBitmap;          // bitmask   of selected items        - code
   TBits *fSelectedStackBitmap;         // bitmask   of selected items - stack
   TInfoStamp* fStackSummary;           // summary information for selected stack
   TFilePtr fFile;                      // current file with information  - stamps
   TTree *fTree;                        // current tree with inforamtion  - stamps
   TTree *fTreeSys;                     // tree with system information
   TObjArray * fStampArray;             // array of stamp names
   TMemStatManager * fManager;          // current MemStatManager
   TObjArray *fArray;                   // array of objects to draw
   TObjArray *fArrayGraphics;           // array of graphic objects
   IntVector_t fArrayIndexes;           // indexes of selected objects
   UInt_t    fMaxStringLength;          // max length of information string
   Bool_t    fOrder;                    // sorting order
   TString   fOption;                   // current options
   TObjArray fDisablePrintLib;          // disable printing for libraries
   TObjArray fDisablePrintCode;         // disable printing for libraries

   ClassDef(TMemStat, 0) // a user interface class of memstat
};

#endif
