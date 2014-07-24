/*
 * $Header$
 * $Log$
 */

#ifndef __XSGRAPH_H
#define __XSGRAPH_H

#include <TGraph.h>
#include <TString.h>

#include "NdbMTReactionXS.h"

/* =================== XSGraph ===================== */
class XSGraph : public TObject
{
protected:
   TString   desc;
   Int_t     Z;
   Int_t     A;

   Int_t     N;
   Float_t  *X;
   Float_t  *Y;

   TGraph   *graph;

public:
   XSGraph()
   {
      N = 0;
      X = Y = NULL;
      graph = NULL;
   }

   XSGraph( NdbMTReactionXS *reac );
   ~XSGraph();

   inline TGraph*   GetGraph()   { return graph; }

   //ClassDef(XSGraph,1)
}; // XSGraph

#endif
