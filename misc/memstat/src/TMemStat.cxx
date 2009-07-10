// @(#)root/memstat:$Name$:$Id$
// Author: M.Ivanov   18/06/2007 -- Anar Manafov (A.Manafov@gsi.de) 29/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/* BEGIN_HTML
<center><h2>The TMemStat class</h2></center>
TMemStat  - a class for visualization of the memory usage.
<br/>
Principle:
Hook functions for malloc and free are used.
All calls to malloc and free are caught and the statistical information is collected.
The information is collected per stack trace (Unique identifier and per functions).<br/>
Following information are collected as function of time:
<ul>
    <li>Total number of allocations</li>
    <li>Total allocation size</li>
    <li>Allocation count</li>
    <li>Allocation size</li>
</ul>
How to use it:<br/>
To create a memstat object use the following:
<pre>
    TMemStat memstat("new");
</pre>
Possible options:
<ul>
    <li> "new" - Start new data collection. In the current release a hard-coded "memstat.root" file will be created or overwritten if exists.</li>
    <li> "read" - "read" - analyze previously collected data.</li>
    <li>"gnubuildin" - if declared, then TMemStat will use gcc build-in function, otherwise glibc backtrace will be used.</li>
</ul>
A user may want to request for adding the named memory stamp:
<pre>
    memstat.AddStamp("STEMPNAME")
</pre>
The file "memstat.root" is created after destruction of object.<br/>
This class supports following  functions for standard user:
<ul>
    <li> Report </li>
    <li> Draw </li>
    <li> SelectCode(libname, functionname) </li>
    <li> SelectStack() </li>
</ul>
The various format options to draw a Graph  can be accessed calling
<pre>
    TMemStat memstat;
    memstat.Report("?")
</pre>

Supported options for TMemStat::Report:
<ul>
    <li> order     : 0=increasing,  1=decreasing</li>
    <li> sortstat  : 0=TotalAllocCount, 1=TotalAlocSize,  2=AllocCount, 3=AllocSize</li>
    <li> sortstamp : 0=Current, 1=MaxSize,  2=MaxCount</li>
    <li> sortdeep  : (0-inf) number of info to print</li>
    <li> stackdeep : (0-inf) deepness of stack</li>
    <li> Example   : order 0 sortstat 3 sortstamp 0 sortdeep 10 stackdeep 5 maxlength 50 </li>
</ul>


The picture below gives an example:
END_HTML
BEGIN_MACRO(source)
{
   {
      TMemStat memstat("new,gnubuildin");

      for (Int_t i=0;i<11000;i++) {
         TObjString * object = new TObjString(Form("Object%d",i));
         if (i%2) delete object;
      }
      for (Int_t i=0;i<12000;i++) {
         TString * object2 = new TString(i);
         if (i%2) delete object2;
      }
      for (Int_t i=0;i<1300;i++) {
         TClonesArray  *array = new TClonesArray("TVectorD");
         //array.ExpandCreatFast(i);
      }
   }

   TMemStat report;
   report->Report();
}
END_MACRO */
////////////////////////////////////////////////////////////////////////////////

// STD
#include <map>
#include <vector>
#include <iostream>
// ROOT
#include "TTree.h"
#include "TMath.h"
#include "TArrayI.h"
#include "TBits.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TText.h"
#include "TLine.h"
#include "THStack.h"
#include "TSystem.h"

#include "TMemStat.h"
#include "TMemStatInfo.h"
#include "TMemStatManager.h"
#include "TMemStatHelpers.h"


ClassImp(TMemStat)

using namespace std;
using namespace Memstat;

static char *ginfo = 0;

typedef vector<Long64_t> Long64Vector_t;
typedef vector<Double_t> DoubleVector_t;

_INIT_TOP_STECK;

struct SFillSelection: public binary_function<
         TMemStatManager::CodeInfoContainer_t::value_type,
         TMemStat::ESelection,
         TMemStat::Selection_t::value_type>
{
   TMemStat::Selection_t::value_type operator() (
      const TMemStatManager::CodeInfoContainer_t::value_type &_code_info,
      TMemStat::ESelection _Selection ) const
   {
      switch ( _Selection )
      {
      case TMemStat::kFunction:
         return _code_info.fFunction.Data();
      case TMemStat::kLibrary:
         // we need only names of libraries
         return gSystem->BaseName(_code_info.fLib.Data());
      default:
         return string();
      }
   }
};

//______________________________________________________________________________
TMemStat::TMemStat(Option_t* option):
      TObject(),
      fSortStat(kAllocSize),
      fSortStamp(kCurrent),
      fMaximum(0),
      fSortDeep(10),
      fStackDeep(20),
      fMaxStringLength(50),
      fSelected(0),
      fIsActive(kFALSE),
      fOrder(kFALSE),
      fSelectedCodeBitmap(NULL),
      fSelectedStackBitmap(NULL),
      fStampArray(NULL),
      fArray(NULL),
      fArrayGraphics(NULL),
      fTree(NULL),
      fTreeSys(NULL),
      fStackSummary(NULL),
      fManager(NULL)
{
   // Supported options:
   //    "new" - start to collect memory allocations stamps
   //    "read" - analyze previously collected data
   //    "gnubuildin" - if declared, then TMemStat will use gcc build-in function,
   //                      otherwise glibc backtrace will be used
   //
   // Default: "read"
   // Note: Currently TMemStat uses a hard-coded output file name (for writing/reading) = "memstat.root";

   // It marks the highest used stack address.
   _GET_TO_STECK;

   //preserve context. When exiting will restore the current directory
   TDirectory::TContext context(gDirectory);

   string opt(option);
   transform( opt.begin(), opt.end(), opt.begin(),
              Memstat::ToLower_t() );

   if ( opt.find("new") != string::npos) // Processing "NEW"
   {
      SetAutoStamp();
      const Bool_t useBuildin = (opt.find("gnubuildin") != string::npos)? kTRUE: kFALSE;
      TMemStatManager::GetInstance()->SetUseGNUBuildinBacktrace(useBuildin);
      TMemStatManager::GetInstance()->Enable();
      // set this variable only if "NEW" mode is active
      fIsActive = kTRUE;
   }
   else if ( opt.find("read") != string::npos) // Processing "READ"
   {
      // GetData from existing file
      GetMemStat("memstat.root", -1);

      // default disabling
      fDisablePrintLib.SetOwner();
      fDisablePrintLib.AddLast(new TObjString("libRIO"));
      fDisablePrintCode.SetOwner();
      fDisablePrintCode.AddLast(new TObjString("TClass::Streamer"));
      // set default option
      ProcessOption("order 0 sortstat 3 sortstamp 0 sortdeep 30 stackdeep 15 maxlength 50");
   }
   else
   {
      Error("TMemStat", "Invalid option");
   }
}

//______________________________________________________________________________
TMemStat::~TMemStat()
{
   // destructor

   if (fIsActive) {
      TMemStatManager::GetInstance()->Disable();
      TMemStatManager::GetInstance()->Close();
   }

   delete fStackSummary;
   delete fSelectedCodeBitmap;
   delete fSelectedStackBitmap;
}

//______________________________________________________________________________
void TMemStat::AddStamp(const char*stampName)
{
   // Add named stamp to the file

   TMemStatManager::GetInstance()->AddStamps(stampName);
}

//______________________________________________________________________________
Int_t TMemStat::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Return distance of the mouse to the TMemStat object
   
   const Int_t big = 9999;
   if (!fArray) return big;

   Int_t mindist = big;
   for (Int_t i = 0; i < fArray->GetSize(); ++i) {
      TObject *obj = fArray->At(i);
      if (!obj) continue;
      Int_t dist = obj->DistancetoPrimitive(px, py);
      if (dist < mindist) mindist = dist;
   }
   return mindist;
}

//______________________________________________________________________________
void TMemStat::Draw(Option_t *option)
{
   // Draw the memory statistic
   // call ::Report("?") to see possible options and meaning

     TString opt(option);
     opt.ToLower();
     if (opt.Contains("?"))
        return;
     
     TLegend *legend = 0;
     if (!gPad) {
        TCanvas *c = new TCanvas;
        c->SetGrid();
        if (gROOT->IsBatch()) {
           c->SetTopMargin(0.2);
           c->SetRightMargin(0.3);
           c->SetLeftMargin(0.10);
           legend = new TLegend(0.75, 0.1, 0.99, 0.9, "Memory statistic");
        } else {
           c->ToggleToolTips();
        }   
     } else {
        gPad->GetListOfPrimitives()->Remove(this);
        gPad->Clear();
     }

     ProcessOption(option);

     RefreshSelect();

     if (!opt.Contains("code")) {
        SortStack(fSortStat, fSortStamp);
        fArray = MakeGraphStack(fSortStat, fSortDeep);
     } else {
        SortCode(fSortStat, fSortStamp);
        fArray = MakeGraphCode(fSortStat, fSortDeep);
     }

     
     Bool_t gDone = kFALSE;
     MakeStampsText();
     if (gPad) {
        for (Int_t i = 0;i < fArray->GetEntries();i++) {
           TObject *obj = fArray->At(i);
           if (!obj) continue;
           if (!gDone) {
              obj->Draw("alp");
              ((TGraph*)obj)->SetMaximum(1.1*fMaximum);
              gDone = kTRUE;
           } else {
              obj->Draw("lp");
           }
           cout << i << '\t' << obj->GetName() << endl;
           if (legend) legend->AddEntry(obj, obj->GetName());
        }
        if (!gROOT->IsBatch()) {AppendPad(); gPad->Update(); return;}
        
        gPad->Update();
        if (legend) legend->Draw();
        fArray->AddLast(legend);
        Int_t ng = 0;
        if (fArrayGraphics) ng = fArrayGraphics->GetEntries();
        
        for (Int_t i = 0; i < ng; i++) {
           TText *ptext = dynamic_cast<TText*>(fArrayGraphics->At(i));
           if (ptext) {
              ptext->SetY(gPad->GetUymax());
              ptext->SetTextAngle(45);
              ptext->SetTextSizePixels(12);
              ptext->SetTextAlign(13);
              ptext->Draw("");
           }
           TLine *pline = dynamic_cast<TLine*>(fArrayGraphics->At(i));
           if (pline) {
              pline->SetY2(gPad->GetUymax());
              pline->SetLineStyle(2);
              pline->Draw("");
           }
        }
     }
     AppendPad();
}

//______________________________________________________________________________
Bool_t TMemStat::EnabledCode(const TMemStatCodeInfo &info) const
{
   // Private function
   // disable printing of the predefined code sequence
   if (info.fLib.Contains("libMemStat.so"))
      return kFALSE;
   if (info.fFunction.Contains("operator new"))
      return kFALSE;
   if (info.fFunction.Contains("TMethodCall::Execute"))
      return kFALSE;
   if (info.fFunction.Contains("Cint::G__CallFunc::Exec"))
      return kFALSE;
   if (info.fFunction.Contains("Cint::G__ExceptionWrapper"))
      return kFALSE;
   if (info.fFunction.Sizeof() <= 1)
      return kFALSE;

   for (Int_t i = 0; i < fDisablePrintLib.GetEntries(); ++i) {
      TObjString * str = (TObjString*)fDisablePrintLib.At(i);
      if (str && info.fLib.Contains(str->String().Data()))
         return kFALSE;
   }

   for (Int_t i = 0; i < fDisablePrintCode.GetEntries(); ++i) {
      TObjString * str = (TObjString*)fDisablePrintCode.At(i);
      if (str && info.fFunction.Contains(str->String().Data()))
         return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TMemStat::ExecuteEvent(Int_t /*event*/, Int_t /* px*/, Int_t /*py*/)
{
   // function called when clicking with the mouse on a TMemStat object

   // TODO: this method needs to be revised
   /* switch (event) {
    case kButton1Down:
       if (fArray && fSelected >= 0) {
          const Int_t uid = fArrayIndexes[fSelected];
          cout << endl;
          (uid >= 0) ? PrintStackWithID(uid) : PrintCodeWithID(-uid);
       }
       break;
    case kMouseMotion:
       break;
    case kButton1Motion:
       break;
    case kButton1Up:
       break;
    }*/
}

//______________________________________________________________________________
void TMemStat::GetFillSelection(Selection_t *_Container, ESelection _Selection) const
{
   // TODO: Comment me

   if ( !_Container || !fManager)
      return;

   transform( fManager->fCodeInfoArray.begin(),
              fManager->fCodeInfoArray.end(),
              inserter(*_Container, _Container->begin()),
              bind2nd(SFillSelection(), _Selection) );
}

//______________________________________________________________________________
Bool_t TMemStat::GetMemStat(const char * fname, Int_t entry)
{
   // Get memstat from tree

   if (fname != 0) {
      fFile.reset(TFile::Open(fname));
      if (!fFile.get() || fFile->IsZombie())
         return kFALSE;

      fTree = dynamic_cast<TTree*>(fFile->Get("MemStat"));
      if (!fTree)
         return kFALSE;

      fTreeSys = dynamic_cast<TTree*>(fFile->Get("MemSys"));
      if (!fTreeSys)
         return kFALSE;
   }

   TMemStatManager *man(NULL);
   // TODO: needs to be investigated.
   // There was a crash, happens when user reselect stamps list after TMemStat Draw has been called.
   // The crash (SEGFAULT) happens in fTree->GetEntry(entry) in access of its buffer.
   // ResetBranchAddresses helped, but it is not clear whether this fix correct or not.
   fTree->ResetBranchAddresses();
   fTree->SetBranchAddress("Manager", &man);

   if ( (entry < 0) || (entry >= fTree->GetEntries()) )
      entry = fTree->GetEntries() - 1;

   fTree->GetEntry(entry);
   fManager = man;
   return kTRUE;
}


//______________________________________________________________________________
char *TMemStat::GetObjectInfo(Int_t px, Int_t py) const
{
   //Display the stack trace info corresponding to the graph at cursor position px,py.

   if (!gPad || !fArray) return (char*)"";
   if (!ginfo) ginfo = new char[10000];
   const Int_t big = 9999;

   Int_t mindist = big;
   TObject *objmin = 0;
   Int_t imin = 0;
   for (Int_t i = 0; i < fArray->GetSize(); ++i) {
      TObject *obj = fArray->At(i);
      if (!obj) continue;
      Int_t dist = obj->DistancetoPrimitive(px, py);
      if (dist < mindist) {mindist = dist; objmin = obj;imin=i;}
   }

   if (objmin) {
      sprintf(ginfo,"-");
      const TMemStatStackInfo &infoStack = fManager->fStackVector[objmin->GetUniqueID()];
      for (UInt_t icode = 0, counter = 0; icode < infoStack.fSize; ++icode) {
         const TMemStatCodeInfo &infoCode(fManager->fCodeInfoArray[infoStack.fSymbolIndexes[icode]]);
         if (!EnabledCode(infoCode)) continue;
         strcat(ginfo,infoCode.fFunction.Data()); strcat(ginfo,"\n");
         ++counter;
         if (counter >= 5) break;
      }
      return ginfo;
   }
   return (char*)"";
}

//______________________________________________________________________________
TObjArray* TMemStat::GetStampList()
{
   // TODO: Comment me
   if (fStampArray)
      return fStampArray;

   if (!fTreeSys)
      return NULL;

   TObjString str;
   TObjString *pstr = &str;
   fStampArray = new TObjArray;
   fTreeSys->SetBranchAddress("StampName.", &pstr);
   for (Int_t i = 0; i < fTreeSys->GetEntries(); ++i) {
      fTreeSys->GetEntry(i);
      fStampArray->AddLast(str.Clone());
   }
   return fStampArray;
}

//______________________________________________________________________________
void TMemStat::MakeCodeArray()
{
   //   PRIVATE: make code index accoring tbit mask

   if (!fManager)
      return;

   Int_t nselected = 0;
   size_t csize = fManager->fCodeInfoArray.size();

   for (UInt_t i = 0; i < csize; ++i)
      if (fSelectedCodeBitmap->TestBitNumber(i))
         ++nselected;

   fSelectedCodeIndex.clear();
   fSelectedCodeIndex.reserve(nselected);
   for (UInt_t i = 0; i < csize; ++i) {
      if (fSelectedCodeBitmap->TestBitNumber(i))
         fSelectedCodeIndex.push_back(i);
   }
}

//______________________________________________________________________________
void TMemStat::MakeStackArray()
{
   //   PRIVATE: make code index according tbit mask

   if (!fManager)
      return;

   delete fStackSummary;
   fStackSummary = new TMemStatInfoStamp();

   fSelectedStackIndex.clear();

   const size_t csize = fManager->fStackVector.size();
   for (size_t i = 0; i < csize; ++i) {
      if (fSelectedStackBitmap->TestBitNumber(i)) {
         fSelectedStackIndex.push_back(i);
         const TMemStatStackInfo &info = fManager->fStackVector[i];
         fStackSummary->fTotalAllocCount += info.fCurrentStamp.fTotalAllocCount;
         fStackSummary->fTotalAllocSize += info.fCurrentStamp.fTotalAllocSize;
         fStackSummary->fAllocCount += info.fCurrentStamp.fAllocCount;
         fStackSummary->fAllocSize += info.fCurrentStamp.fAllocSize;
      }
   }
}



//______________________________________________________________________________
TObjArray *TMemStat::MakeGraphCode(StatType statType, Int_t nentries)
{
   // make array of graphs

   if (fArray) {
      fArray->Delete();
      delete fArray;
   }
   fArray  = new TObjArray(nentries);

   fArrayIndexes.clear();
   fArrayIndexes.resize(nentries);

   Int_t count = 0;
   Int_t first = TMath::Max(static_cast<Int_t>(fSelectedCodeIndex.size()) - nentries, 0);
   Double_t cxmax, cymax;
   for (Int_t i = fSelectedCodeIndex.size() - 1; i > first; --i) {
      TGraph * gr = MakeGraph(statType, fSelectedCodeIndex[i], TMemStatInfoStamp::kCode, cxmax, cymax);
      if (!gr)
         continue;
      TMemStatCodeInfo  &cinfo =  fManager->fCodeInfoArray[fSelectedCodeIndex[i]];
      if (cinfo.fFunction.Length() > 0) {
         TString str(cinfo.fFunction);
         if ((UInt_t)(str.Length()) > fMaxStringLength)
            str.Resize(fMaxStringLength);
         gr->SetName(str);
      }
      ++count;
      gr->SetLineColor(count % 5 + 1);

      fArrayIndexes[fArray->GetEntries()] = -fSelectedCodeIndex[i];
      fArray->AddLast(gr);
   }
   return fArray;
}

//______________________________________________________________________________
TObjArray *TMemStat::MakeGraphStack(StatType statType, Int_t nentries)
{
   // make array of graphs

   if (fArray) {
      fArray->Delete();
      delete fArray;
   }
   fArray = new TObjArray(nentries);

   fArrayIndexes.clear();
   fArrayIndexes.resize(nentries);

   Int_t count = 0;
   const Int_t first = TMath::Max(static_cast<int>(fSelectedStackIndex.size()) - nentries, 0);
   Double_t cxmax(0);
   Double_t cymax(0);
   fMaximum = 0;
   for (Int_t i = fSelectedStackIndex.size() - 1; i > first; --i) {
      TGraph * gr = MakeGraph(statType, fSelectedStackIndex[i], TMemStatInfoStamp::kStack, cxmax, cymax);
      if (!gr)
         continue;
      if (cymax > fMaximum) fMaximum = cymax;
      TMemStatStackInfo &infoStack = fManager->fStackVector[(fSelectedStackIndex[i])];
      for (UInt_t icode = 0; icode < infoStack.fSize; icode++) {
         TMemStatCodeInfo &infoCode = fManager->fCodeInfoArray[infoStack.fSymbolIndexes[icode]];
         if (EnabledCode(infoCode)) {
            if (infoCode.fFunction) {
               TString str(infoCode.fFunction);
               if ((UInt_t)(str.Length()) > fMaxStringLength) str.Resize(fMaxStringLength);
               gr->SetName(str);
               gr->SetUniqueID(fSelectedStackIndex[i]);
            }
            break;
         }
      }
      ++count;
      gr->SetLineColor(count % 5 + 1);
      gr->SetMarkerColor(count % 5 + 1);
      gr->SetMarkerStyle(20 + count % 5);
      gr->SetMarkerSize(0.15);
      fArrayIndexes[fArray->GetEntries()] = fSelectedStackIndex[i];
      fArray->AddLast(gr);
   }

   return fArray;
}

//______________________________________________________________________________
TGraph *TMemStat::MakeGraph(StatType statType, Int_t id, Int_t type, Double_t &xmax, Double_t &ymax)
{
   // Make graph

   if (!fTree)
      return 0;

   string sWhat;
   string sWhatName;
   switch (statType) {
   case kTotalAllocCount:
      sWhat = "fStampVector.fTotalAllocCount:fStampVector.fStampNumber";
      sWhatName = "TotalAllocCount";
      break;
   case kAllocCount:
      sWhat = "fStampVector.fAllocCount:fStampVector.fStampNumber";
      sWhatName = "AllocCount";
      break;
   case kTotalAllocSize:
      sWhat = "fStampVector.fTotalAllocSize/1000000.:fStampVector.fStampNumber";
      sWhatName = "TotalAllocSize (MBy)";
      break;
   case kAllocSize:
      sWhat = "fStampVector.fAllocSize/1000000.:fStampVector.fStampNumber";
      sWhatName = "AllocSize (MBy)";
      break;
   case kUndef:
      // TODO: check this case; in original code it wasn't handled
      break;
   }
   ostringstream ssWhere;
   ssWhere << "fStampVector.fID==" << id << "&&fStampVector.fStampType==" << type;

   const Int_t entries = fTree->Draw(sWhat.c_str(), ssWhere.str().c_str(), "goff");
   if (entries <= 0)
      return 0;

   const Int_t maxStamp = fManager->fStampNumber;

   Float_t *x = new Float_t[maxStamp];
   Float_t *y = new Float_t[maxStamp];
   xmax = 0;
   ymax = 0;
   Float_t last = 0;
   for (Int_t i = 0, counter = 0; i < maxStamp; ++i) {
      x[i] = i;
      y[i] = last;
      if (y[i] > ymax) ymax = y[i];
      if (x[i] > xmax) xmax = x[i];
      if (counter >= entries)
         continue;
      if (fTree->GetV2()[counter] > i) {
         y[i] = last;
      } else {
         y[i] = fTree->GetV1()[counter];
         last = y[i];
         ++counter;
      }
   }
   TGraph * graph  = new TGraph(maxStamp, x, y);
   graph->GetXaxis()->SetTitle("StampNumber");
   graph->GetYaxis()->SetTitle(sWhatName.c_str());
   return graph;
}

//______________________________________________________________________________
void TMemStat::MakeHisMemoryStamp(Int_t /*topDiff*/)
{
   //draw histogram of memory usage as function of stamp name
   // NOT IMPLEMENTED YET

   const Int_t entries = fTreeSys->Draw("Mem3", "Mem3>0", "");
   DoubleVector_t diff(entries - 1);
   for (Int_t i = 0; i < entries - 1; ++i) {
      diff[i] = fTreeSys->GetV1()[i+1] - fTreeSys->GetV1()[i];
   }
   IntVector_t indexes(entries - 1);
   TMath::Sort(entries - 1, &diff[0], &indexes[0], kFALSE);
}

//______________________________________________________________________________
void TMemStat::MakeHisMemoryTime()
{
   // Make dump of memory usage versus time

   fTreeSys->Draw("Mem3:StampTime.fSec>>hhh", "", "goff*");
   if (!gROOT->FindObject("hhh"))
      return ;

   TH1* his3 = (TH1*)gROOT->FindObject("hhh")->Clone("Virtual Memory");
   his3->SetDirectory(0);
   delete gROOT->FindObject("hhh");

   fTreeSys->Draw("Mem2:StampTime.fSec>>hhh", "", "goff*");
   if (!gROOT->FindObject("hhh"))
      return ;

   TH1* his2 = (TH1*)gROOT->FindObject("hhh")->Clone("Residual Memory");
   his2->SetDirectory(0);
   delete gROOT->FindObject("hhh");

   fTreeSys->Draw("CurrentStamp.fAllocSize/1000000.:StampTime.fSec>>hhh", "", "goff*");
   if (!gROOT->FindObject("hhh"))
      return ;

   TH1* hism = (TH1*)gROOT->FindObject("hhh")->Clone("Allocated Memory");
   hism->SetDirectory(0);
   delete gROOT->FindObject("hhh");

   his3->GetXaxis()->SetTimeDisplay(1);
   his3->SetMarkerColor(2);
   his2->SetMarkerColor(3);
   hism->SetMarkerColor(4);
   his3->Draw();
   his2->Draw("same");
   hism->Draw("same");
}

//______________________________________________________________________________
void TMemStat::MakeReport(const char * lib, const char *fun, Option_t* option, const char *fileName)
{
   // make report for library

   // reset selection
   SelectCode(NULL, NULL, TMemStat::kOR);
   SelectStack(TMemStat::kOR);
   //
   SelectCode(lib, fun, TMemStat::kAND);
   SelectStack(TMemStat::kAND);
   if (option)
      ProcessOption(option);
   SortCode(fSortStat, fSortStamp);
   SortStack(fSortStat, fSortStamp);

   // Redirecting the output if needed
   if (strlen(fileName) > 0)
      gSystem->RedirectOutput(fileName, "w");

   Report();

   if (strlen(fileName) > 0)
      gSystem->RedirectOutput(0);
}

//______________________________________________________________________________
void TMemStat::MakeStampsText()
{
   // Make a text description of the stamps
   // create a array of TText objects

   // TODO: see TMemStat::Draw
   /*if (!fArrayGraphics)
       fArrayGraphics = new TObjArray();

    const Int_t nentries = fTree->GetEntries();
    Int_t stampNumber(0);
    TObjString *pnameStamp(NULL);
    TTimeStamp *ptimeStamp(NULL);
    fTree->SetBranchAddress("StampTime.", &ptimeStamp);
    fTree->SetBranchAddress("StampName.", &pnameStamp);
    fTree->SetBranchAddress("StampNumber", &stampNumber);

    for (Int_t i = 0; i < nentries; ++i) {
       fTree->GetBranch("StampTime.")->GetEntry(i);
       fTree->GetBranch("StampName.")->GetEntry(i);
       fTree->GetBranch("StampNumber")->GetEntry(i);
       char chname[1000];
       if (pnameStamp->GetString().Contains("autoStamp")) {
          sprintf(chname, " ");
       } else {
          sprintf(chname, "%s  %d", pnameStamp->GetString().Data(), ptimeStamp->GetTime());
       }
       TText *ptext = new TText(stampNumber, 0, chname);
       fArrayGraphics->AddLast(ptext);
       TLine * pline = new TLine(stampNumber, 0, stampNumber, 1);
       fArrayGraphics->AddLast(pline);
    }*/
}

//______________________________________________________________________________
void TMemStat::Paint(Option_t * /* option */)
{
   // Paint function
}

//______________________________________________________________________________
void TMemStat::PrintCode(Int_t nentries) const
{
   // Print information about n selected functions
   // If the number of function selected is bigger than number n
   // the only top (sorted accoring some creteria,) n are displayed
   // e.g draw.SortCode(TMemStat::kAllocSize,TMemStat::kCurrent);
   if (fSelectedCodeIndex.empty() || !fManager)
      return;

   UIntVector_t::const_iterator iter = max((fSelectedCodeIndex.end() - nentries), fSelectedCodeIndex.begin());
   UIntVector_t::const_iterator iter_end = fSelectedCodeIndex.end();
   for (; iter != iter_end; ++iter)
      fManager->fCodeInfoArray[*iter].Print();
}

//______________________________________________________________________________
void TMemStat::PrintCodeWithID(UInt_t id) const
{
   // print information for code with ID

   if (!fManager)
      return;
   if (id > fManager->fCodeInfoArray.size())
      return;
   fManager->fCodeInfoArray[id].Print();
}

//______________________________________________________________________________
void TMemStat::PrintStack(Int_t nentries, UInt_t deep) const
{
   // Print information about n selected stack traces
   // If the number of function selected is bigger than number n
   // the only top (sorted according some criteria) n are displayed
   // e.g draw.SortCode(TMemStat::kAllocSize,TMemStat::kCurrent);
   if (fSelectedStackIndex.empty())
      return;

   UIntVector_t::const_iterator iter = max((fSelectedStackIndex.end() - nentries), fSelectedStackIndex.begin());
   UIntVector_t::const_iterator iter_end = fSelectedStackIndex.end();
   for (; iter != iter_end; ++iter)
      PrintStackWithID(*iter, deep);

   cout << "Summary for selected:" << endl;
   ios::fmtflags old_flags(cout.flags(ios::left));
   fStackSummary->Print();
   cout.flags(old_flags);
}


//______________________________________________________________________________
void TMemStat::PrintStackWithID(UInt_t _id, UInt_t _deep) const
{
   // print stack information for code with ID
   // NOTE: if _deep is 0, then we use fStackDeep
   if (!fManager)
      return;

   _deep = !_deep ? fStackDeep : _deep;

   const TMemStatStackInfo &infoStack = fManager->fStackVector[_id];
   cout << infoStack << endl;

   ios::fmtflags old_flags(cout.flags(ios::left));
   for (UInt_t icode = 0, counter = 0; icode < infoStack.fSize; ++icode) {
      const TMemStatCodeInfo &infoCode(fManager->fCodeInfoArray[infoStack.fSymbolIndexes[icode]]);
      if (!EnabledCode(infoCode))
         continue;
      cout
      << setw(5) << icode
      << infoCode << endl;
      ++counter;
      if (counter >= _deep)
         break;
   }
   cout.flags(old_flags);
}

//______________________________________________________________________________
void TMemStat::ProcessOption(Option_t *option)
{
   // PRIVATE function
   // process user option string for printing

   TString str(option);
   TString delim(" ");
   TObjArray *tokens = str.Tokenize(delim);
   for (Int_t i = 0; i < tokens->GetEntriesFast() - 1; ++i) {
      TObjString *strTok = (TObjString*)tokens->At(i);
      TObjString *strNum = (i < tokens->GetEntriesFast()) ? (TObjString*)tokens->At(i + 1) : 0;

      if (strNum && strNum->String().IsDigit()) {
         if (strTok->String().Contains("sortstat")) {
            Int_t val = strNum->String().Atoi();
            if (val > 3) {
               Error("SetOption", Form("Invalid value for sortstat %d", val));
               val = 3;
            }
            fSortStat = (TMemStat::StatType)val;
         }
         if (strTok->String().Contains("sortstamp")) {
            Int_t val = strNum->String().Atoi();
            if (val > 2) {
               Error("SetOption", Form("Invalid value for sortstamp %d", val));
               val = 0;
            }
            fSortStamp = (TMemStat::StampType)val;
         }

         if (strTok->String().Contains("order")) {
            Int_t val = strNum->String().Atoi();
            if (val > 1) {
               Error("SetOption", Form("Invalid sorting value", val));
               val = 0;
            }
            fOrder = (val > 0);
         }
         if (strTok->String().Contains("sortdeep")) {
            fSortDeep = strNum->String().Atoi();
         }
         if (strTok->String().Contains("stackdeep")) {
            fStackDeep  = strNum->String().Atoi();
         }
         if (strTok->String().Contains("maxlength")) {
            fMaxStringLength  = strNum->String().Atoi();
         }
      }
   }
   char currentOption[1000];
   sprintf(currentOption, "order %d sortstat %d sortstamp %d sortdeep %d stackdeep %d maxlength %d",
           fOrder, fSortStat, fSortStamp, fSortDeep, fStackDeep, fMaxStringLength);
   fOption = currentOption;
   if (str.Contains("?")) {
      printf("Options   : %s\n", fOption.Data());
      printf("order     : 0 - increasing 1 - decreasing\n");
      printf("sortstat  : 0 - TotalAllocCount 1 -  TotalAlocSize  2 - AllocCount 3 - AllocSize\n");
      printf("sortstamp : 0 - Current 1 -  MaxSize  2 - MaxCount\n");
      printf("sortdeep  : (0-inf) number of info to print\n");
      printf("stackdeep : (0-inf) deepnes of stack\n");
      printf("maxlength : (0-inf) maximal length of function (truncation after maxlength)");
   }

   delete tokens;
}

//______________________________________________________________________________
void TMemStat::RefreshSelect()
{
   // TODO: Comment me
   if (fSelectedCodeIndex.empty())
      SelectCode(NULL, NULL, TMemStat::kOR);

   if (fSelectedStackIndex.empty())
      SelectStack(TMemStat::kOR);
}

//______________________________________________________________________________
void TMemStat::Report(Option_t* option)
{
   // Report function
   // Supported options:
   //
   // order     : 0 - increasing 1 - decreasing
   // sortstat  : 0 - TotalAllocCount 1 -  TotalAlocSize  2 - AllocCount 3 - AllocSize
   // sortstamp : 0 - Current 1 -  MaxSize  2 - MaxCount
   // sortdeep  : (0-inf) number of info to print
   // stackdeep : (0-inf) deepness of stack
   // Example   : order 0 sortstat 3 sortstamp 0 sortdeep 10 stackdeep 5 maxlength 50

   ProcessOption(option);

   TString opt(option);
   opt.ToLower();
   if (opt.Contains("?"))
      return;

   RefreshSelect();

   if (!(opt.Contains("code"))) {
      SortStack(fSortStat, fSortStamp);
      PrintStack(fSortDeep, fStackDeep);
   } else {
      SortCode(fSortStat, fSortStamp);
      PrintCode(fSortDeep);
   }
}

//______________________________________________________________________________
void TMemStat::ResetSelection()
{
   // reset all selections

   fSelectedCodeIndex.clear();
   fSelectedStackIndex.clear();

   delete fSelectedCodeBitmap;
   fSelectedCodeBitmap = NULL;
   delete fSelectedStackBitmap;
   fSelectedStackBitmap = NULL;
   delete fStackSummary;
   fStackSummary = NULL;
}

//______________________________________________________________________________
void TMemStat::SetAutoStamp(Int_t autoStampSize, Int_t autoStampAlloc)
{
   // Change default values of the auto stamping
   //   autoStampSize  [in] - size of invoking STAMPs
   //   autoStampAlloc [in] - a number of allocations

   if (autoStampSize > 0)
      TMemStatManager::GetInstance()->SetAutoStamp(autoStampSize, autoStampAlloc, 10000);
}

//______________________________________________________________________________
void TMemStat::SetCurrentStamp(const char *stampName)
{
   // Getvalues for iven stamp

   GetStampList();

   const Int_t entry = find_string(*fStampArray, stampName);
   GetMemStat(0, entry);
}

//______________________________________________________________________________
void TMemStat::SetCurrentStamp(const TObjString &stampName)
{
   // TODO: Comment me
   SetCurrentStamp(stampName.GetString());
}

//______________________________________________________________________________
void TMemStat::SelectCode(const char *contlib, const char *contfunction, OperType oType)
{
   // select code with given mask
   // contlib       - select only code containing contlib in library path
   // contfunction  - select only code with function name containing contfunction
   // oType         - logical operation - AND and Or is supported
   // By default all code is enabled

   if (!fManager) {
      Error("SelectCode", "MemStat Manager is the NULL object.");
      return;
   }

   const size_t entries = fManager->fCodeInfoArray.size();

   fSelectedCodeIndex.clear();

   if (!fSelectedCodeBitmap) {
      fSelectedCodeBitmap = new TBits(entries);
      for (UInt_t i = 0; i < entries; ++i)
         fSelectedCodeBitmap->SetBitNumber(i, kFALSE);
   }

   switch (oType) {
   case kOR:
      for (UInt_t i = 0; i < entries; ++i) {
         if (fSelectedCodeBitmap->TestBitNumber(i))
            continue;
         const TMemStatCodeInfo &info = fManager->fCodeInfoArray[i];
         if (contlib && (!(info.fLib.Contains(contlib))))
            continue;
         if (contfunction && (!(info.fFunction.Contains(contfunction))))
            continue;
         if (info.fFunction.Contains("TObject::operator new"))
            continue;
         fSelectedCodeBitmap->SetBitNumber(i);
      }
      break;
   case kAND:
      for (UInt_t i = 0; i < entries; i++) {
         if (!(fSelectedCodeBitmap->TestBitNumber(i)))
            continue;
         const TMemStatCodeInfo&info = fManager->fCodeInfoArray[i];
         fSelectedCodeBitmap->SetBitNumber(i, kFALSE);
         if (contlib && (!(info.fLib.Contains(contlib))))
            continue;
         if (contfunction && (!(info.fFunction.Contains(contfunction))))
            continue;
         if (info.fFunction.Contains("TObject::operator new"))
            continue;
         fSelectedCodeBitmap->SetBitNumber(i, kTRUE);
      }
      break;
   case kNOT:
      break;
   }

   MakeCodeArray();
}

//______________________________________________________________________________
void TMemStat::SelectStack(OperType oType)
{
   // select stack containing the selected code
   // oType - And and OR is supported (AND, OR in respect with previous selection)

   if (!fSelectedCodeBitmap || !fManager)
      return;

   const size_t entries = fManager->fStackVector.size();

   fSelectedStackIndex.clear();

   if (!fSelectedStackBitmap) {
      fSelectedStackBitmap = new TBits(entries);
      for (UInt_t i = 0; i < entries; ++i)
         fSelectedStackBitmap->SetBitNumber(i, kFALSE);
   }

   switch (oType) {
   case kOR:
      for (UInt_t i = 0; i < entries; ++i) {
         if (fSelectedStackBitmap->TestBitNumber(i))
            continue;
         const TMemStatStackInfo &info = fManager->fStackVector[i];
         for (UInt_t icode = 0; icode < info.fSize; ++icode) {
            if (fSelectedCodeBitmap->TestBitNumber(info.fSymbolIndexes[icode])) {
               fSelectedStackBitmap->SetBitNumber(i, kTRUE);
            }
         }
      }
      break;
   case kAND:
      for (UInt_t i = 0; i < entries; ++i) {
         if (!(fSelectedStackBitmap->TestBitNumber(i)))
            continue;
         const TMemStatStackInfo &info = fManager->fStackVector[i];
         fSelectedStackBitmap->SetBitNumber(i, kFALSE);
         for (UInt_t icode = 0; icode < info.fSize; ++icode) {
            if (fSelectedCodeBitmap->TestBitNumber(info.fSymbolIndexes[icode])) {
               fSelectedStackBitmap->SetBitNumber(i, kTRUE);
            }
         }
      }
      break;
   case kNOT:
      break;
   }

   MakeStackArray();
}

//______________________________________________________________________________
void TMemStat::SortCode(StatType sortType, StampType stampType)
{
   // sort code according statistical criteria
   // sortType
   // stampType

   if (fSelectedCodeIndex.empty() || !fManager)
      return;

   const Int_t size = fSelectedCodeIndex.size();
   Long64Vector_t values(size);
   TArrayI indexes(size);

   const size_t entries = fManager->fCodeInfoArray.size();
   Int_t iselected = 0;
   for (UInt_t icode = 0; icode < entries; ++icode) {
      if (!(fSelectedCodeBitmap->TestBitNumber(icode)))
         continue;
      TMemStatInfoStamp * info = 0;
      switch (stampType) {
      case kCurrent:
         info = &(fManager->fCodeInfoArray[icode].fCurrentStamp);
         break;
      case kMaxSize:
         info = &(fManager->fCodeInfoArray[icode].fMaxStampSize);
         break;
      case kMaxCount:
         info = &(fManager->fCodeInfoArray[icode].fMaxStamp);
         break;
      }

      if (!info)
         break;

      indexes[iselected] = icode;

      switch (sortType) {
      case kTotalAllocCount:
         values[iselected] = info->fTotalAllocCount;
         break;
      case kAllocCount:
         values[iselected] = info->fAllocCount;
         break;
      case kTotalAllocSize:
         values[iselected] = info->fTotalAllocSize;
         break;
      case kAllocSize:
         values[iselected] = info->fAllocSize;
         break;
      case kUndef:
         break;
      }

      ++iselected;
   }

   TArrayI sortIndexes(size);
   TMath::Sort(iselected, &values[0], &sortIndexes[0], fOrder);

   fSelectedCodeIndex.clear();
   fSelectedCodeIndex.reserve(iselected);
   for (Int_t i = 0; i < iselected; ++i)
      fSelectedCodeIndex.push_back(indexes[sortIndexes[i]]);
}

//______________________________________________________________________________
void TMemStat::SortStack(StatType sortType, StampType stampType)
{
   // sort code according statistical criteria
   // sortType
   // stampType

   if (!fSelectedCodeBitmap || !fManager)
      return;

   const size_t entries = fManager->fStackVector.size();
   Long64Vector_t values(entries);
   TArrayI indexes(entries);

   UInt_t iselected = 0;
   for (UInt_t istack = 0; istack < entries; ++istack) {
      if (!(fSelectedStackBitmap->TestBitNumber(istack)))
         continue;
      TMemStatInfoStamp * info(NULL);
      switch (stampType) {
      case kCurrent:
         info = &(fManager->fStackVector[istack].fCurrentStamp);
         break;
      case kMaxSize:
         info = &(fManager->fStackVector[istack].fMaxStampSize);
         break;
      case kMaxCount:
         info = &(fManager->fStackVector[istack].fMaxStamp);
         break;
      }

      indexes[iselected] = istack;

      switch (sortType) {
      case kTotalAllocCount:
         values[iselected] = info->fTotalAllocCount;
         break;
      case kAllocCount:
         values[iselected] = info->fAllocCount;
         break;
      case kTotalAllocSize:
         values[iselected] = info->fTotalAllocSize;
         break;
      case kAllocSize:
         values[iselected] = info->fAllocSize;
         break;
      case kUndef:
         break;
      }
      if (values[iselected] == 0) continue;
      ++iselected;
   }
   TArrayI  sortIndexes(entries);
   TMath::Sort(static_cast<Int_t>(iselected), &values[0], &sortIndexes[0], fOrder);
   const Int_t sizeOut = TMath::Min(fSortDeep, iselected);
   fSelectedStackIndex.clear();
   fSelectedStackIndex.reserve(sizeOut);
   if (fOrder) {
      for (Int_t i = 0; i < sizeOut; ++i)
         fSelectedStackIndex.push_back(indexes[sortIndexes[i]]);
   } else {
      const Int_t first = (iselected < fSortDeep) ? 0 : iselected - fSortDeep;
      for (UInt_t i = first; i < (first + fSortDeep) && i < iselected; ++i) {
         const UInt_t indexS = sortIndexes[i];
         if (indexS >= entries) {
            cerr << "Error 0 \n";
            continue;
         }
         if (static_cast<size_t>(indexes[indexS]) >= entries) {
            cerr << "Error 1 \n";
            continue;
         }
         const Long64_t value = values[indexS];
         if (0 == value) {
            cerr << "Error 2\n";
            continue;
         }
         fSelectedStackIndex.push_back(indexes[indexS]);
      }
   }
}
