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

class TMemStatShow : public TObject {
   
protected:
   static Long64_t fgAddressFirst; //first address to process
   static Long64_t fgAddressN;     //number of addresses in bytes to process
   static Long64_t fgEntryFirst;   //first entry to process
   static Long64_t fgEntryN;       //number of entries to process

public:
   TMemStatShow() {;}
   virtual   ~TMemStatShow() {;}
   static void EventInfo(Int_t event, Int_t px, Int_t py, TObject *selected);
   static void FillBTString(Int_t bin, Int_t mode, TString &btstring);
   
   static void SetAddressRange(Long64_t nbytes=0, Long64_t first=0);
   static void SetEntryRange(Long64_t nentries=0, Long64_t first=0);
   static void Show(Double_t update=0.1, Int_t nbigleaks=20, const char* fname="*");

   ClassDef(TMemStatShow,0)  //class to visualize the results of TMemStat 
};

#endif
