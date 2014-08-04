// @(#)root/treeviewer:$Id$
// Author: Rene Brun   21/09/2010

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMemStatShow
#define ROOT_TMemStatShow



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMemStatShow                                                         //
//                                                                      //
// class to visualize the results of TMemStat                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TTree;
class TH1D;
class TH1I;
class TGToolTip;
class TObjArray;
class TCanvas;

class TMemStatShow : public TObject {

protected:
   static TTree     *fgT;        //TMemStat Tree
   static TH1D      *fgHalloc;   //histogram with allocations
   static TH1D      *fgHfree;    //histogram with frees
   static TH1D      *fgH;        //histogram with allocations - frees
   static TH1I      *fgHleaks;   //histogram with leaks
   static TH1I      *fgHentry;   //histogram with entry numbers in the TObjArray
   static TH1I      *fgHdiff;    //histogram with diff of entry number between alloc/free

   static TGToolTip *fgTip1;     //pointer to tool tip for canvas 1
   static TGToolTip *fgTip2;     //pointer to tool tip for canvas 2
   static TObjArray *fgBtidlist; //list of back trace ids
   static Double_t  *fgV1;       //pointer to V1 array of TTree::Draw (pos)
   static Double_t  *fgV2;       //pointer to V2 array of TTree::Draw (nbytes)
   static Double_t  *fgV3;       //pointer to V3 array of TTree::Draw (time)
   static Double_t  *fgV4;       //pointer to V4 array of TTree::Draw (btid)
   static TCanvas   *fgC1;       //pointer to canvas showing allocs/deallocs vs time
   static TCanvas   *fgC2;       //pointer to canvas with leaks in decreasing order
   static TCanvas   *fgC3;       //pointer to canvas showing the main leaks

   static Long64_t fgAddressFirst; //first address to process
   static Long64_t fgAddressN;     //number of addresses in bytes to process
   static Long64_t fgEntryFirst;   //first entry to process
   static Long64_t fgEntryN;       //number of entries to process

public:
   TMemStatShow() {;}
   virtual   ~TMemStatShow() {;}
   static void EventInfo1(Int_t event, Int_t px, Int_t py, TObject *selected);
   static void EventInfo2(Int_t event, Int_t px, Int_t py, TObject *selected);
   static void FillBTString(Int_t bin, Int_t mode, TString &btstring);

   static void SetAddressRange(Long64_t nbytes=0, Long64_t first=0);
   static void SetEntryRange(Long64_t nentries=0, Long64_t first=0);
   static void Show(Double_t update=0.1, Int_t nbigleaks=20, const char* fname="*");

   ClassDef(TMemStatShow,0)  //class to visualize the results of TMemStat
};

#endif
