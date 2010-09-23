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
   
public:
   TMemStatShow() {;}
   virtual   ~TMemStatShow() {;}
   static void EventInfo(Int_t event, Int_t px, Int_t py, TObject *selected);

   static void Show(Double_t update=0.01, const char* fname="*");

   ClassDef(TMemStatShow,0)  //class to visualize the results of TMemStat 
};

#endif
