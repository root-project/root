// @(#)root/sessionviewer:$Id$
// Author: Anna Kreshuk 18/07/2008

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofProgressMemoryPlot
#define ROOT_TProofProgressMemoryPlot

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressMemoryPlot                                             //
//                                                                      //
// This class implements a dialog, used to display the memory footprint //
// on the proof workers and master. For the workers, memory is plotted  //
// as a function of number of events processed. For the master, it is   //
// plotted as a function of number of objects merged                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGListBox;
class TGTextButton;
class TRootEmbeddedCanvas;
class TProofProgressDialog;
class TGSplitButton;
class TProofLog;
class TMultiGraph;
class TGraph;
class TProofLogElem;

class TProofProgressMemoryPlot : public TGTransientFrame {

 protected:
   TProofProgressDialog *fDialog;
   TGListBox            *fWorkers;
   TGTextButton         *fPlot;
   TGSplitButton        *fAllWorkers; // display all workers button
   TRootEmbeddedCanvas  *fWorkersPlot;
   TRootEmbeddedCanvas  *fMasterPlot;
   TProofLog            *fProofLog;
   TMultiGraph          *fWPlot;
   TMultiGraph          *fMPlot;
   TMultiGraph          *fAPlot;
   Bool_t               fFullLogs;

   TGListBox* BuildLogList(TGFrame *parent);
   TGraph*    DoWorkerPlot(TProofLogElem *ple);
   TGraph*    DoMasterPlot(TProofLogElem *ple);
   TGraph*    DoAveragePlot(Int_t &max_el, Int_t &min_el);
   TGraph*    DoAveragePlotOld(Int_t &max_el, Int_t &min_el);

   Int_t      ParseLine(TString l, Long64_t &v, Long64_t &r, Long64_t &e);

 public:
   TProofProgressMemoryPlot(TProofProgressDialog *d, Int_t w = 700, Int_t h = 300);
   virtual ~TProofProgressMemoryPlot();

   void       Clear(Option_t * = 0);
   void       DoPlot();
   void       Select(Int_t id);

   ClassDef(TProofProgressMemoryPlot,0) //PROOF progress memory plots
};

#endif
