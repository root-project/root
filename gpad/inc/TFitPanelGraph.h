// @(#)root/gpad:$Name:  $:$Id: TFitPanelGraph.h,v 1.3 2005/11/23 11:03:12 couet Exp $
// Author: Rene Brun   28/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitPanelGraph
#define ROOT_TFitPanelGraph


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitPanelGraph                                                       //
//                                                                      //
// Class used to control graphs fit panel                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////



#ifndef ROOT_TFitPanel
#include "TFitPanel.h"
#endif

class TSlider;
class TH1;

class TFitPanelGraph : public TFitPanel {

public:
   TFitPanelGraph();
   TFitPanelGraph(const char *name, const char *title, UInt_t ww, UInt_t wh, const TVirtualPad *pad, const TObject *obj);
   virtual ~TFitPanelGraph();
   virtual void  Apply(const char *action="");
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TFitPanelGraph,1)  //Class used to control graphs fit panel
};

#endif

