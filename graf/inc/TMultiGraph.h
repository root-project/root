// @(#)root/graf:$Name:  $:$Id: TMultiGraph.h,v 1.4 2000/10/12 10:00:59 brun Exp $
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMultiGraph
#define ROOT_TMultiGraph


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMultiGraph                                                          //
//                                                                      //
// A collection of TGraph objects                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif


class TH1F;
class TAxis;
class TBrowser;
class TGraph;

class TMultiGraph : public TNamed {

protected:
    TList      *fGraphs;     //Pointer to list of TGraphs
    TH1F       *fHistogram;  //Pointer to histogram used for drawing axis
    Double_t    fMaximum;    //Maximum value for plotting along y
    Double_t    fMinimum;    //Minimum value for plotting along y

public:

        TMultiGraph();
        TMultiGraph(const char *name, const char *title);
        virtual ~TMultiGraph();
        virtual void     Add(TGraph *graph);
        virtual void     Browse(TBrowser *b);
        virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
        virtual void     Draw(Option_t *chopt="");
        TH1F            *GetHistogram();
        TList           *GetListOfGraphs() { return fGraphs; }
        TAxis           *GetXaxis();
        TAxis           *GetYaxis();
        virtual void     Paint(Option_t *chopt="");
        virtual void     Print(Option_t *chopt="");
        virtual void     SavePrimitive(ofstream &out, Option_t *option);
        virtual void     SetMaximum(Double_t maximum=-1111);
        virtual void     SetMinimum(Double_t minimum=-1111);

        ClassDef(TMultiGraph,1)  //A collection of TGraph objects
};

#endif


