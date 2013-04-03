// @(#)root/hist:$Id$
// Author: Olivier Couet 13/07/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphEdge
#define ROOT_TGraphEdge

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#if !defined(__CINT__)
#include <types.h>
#else
struct Agraph_t;
struct Agedge_t;
#endif

class  TGraphNode;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphEdge                                                           //
//                                                                      //
// Interface to the graphviz package.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


class TGraphEdge: public TObject, public TAttLine {

protected:

   TGraphNode *fNode1;  // First node
   TGraphNode *fNode2;  // Second node
   Agedge_t   *fGVEdge; // Graphviz edge
   Double_t   *fX;      // X edge points (GV)
   Double_t   *fY;      // X edge points (GV)
   Int_t      *fN;      // number of edge points (GV)
                        // fN[0] = number of splines
			// fN[1...n] = number of points in each spline
   Double_t    fArrX;   // Arrow X position
   Double_t    fArrY;   // Arrow Y position

public:

   TGraphEdge();
   TGraphEdge(TGraphNode *n1, TGraphNode *n2);
   virtual ~TGraphEdge();

   void           CreateGVEdge(Agraph_t *gv);
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);
   virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
   void           SetGVEdge(Agedge_t *gve) {fGVEdge = gve;}
   Agedge_t      *GetGVEdge() {return fGVEdge;}
   TGraphNode    *GetNode1() {return fNode1;}
   TGraphNode    *GetNode2() {return fNode2;}
   void           Layout();            
   virtual void   Paint(Option_t *option="");
   virtual void   SavePrimitive(ostream &, Option_t *);
   void           SaveAttributes(ostream &, const char*);                  


   ClassDef(TGraphEdge,1)  //Graph edge class
};

#endif
