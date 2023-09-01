#include "XSGraph.h"

/* ===================== XSGraph ================= */
XSGraph::XSGraph( NdbMTReactionXS *reac )
{
   N = reac->Pairs();

   X = new Float_t[N];
   Y = new Float_t[N];

   for (int i=0; i<N; i++) {
      X[i] = reac->Energy(i);
      Y[i] = reac->XS(i);
   }

   graph = new TGraph(N,X,Y);
} // XSGraph

/* ---------- ~XSGraph --------- */
XSGraph::~XSGraph()
{
   if (N) {
      delete [] X;
      delete [] Y;
      delete graph;
   }
} // ~XSGraph
