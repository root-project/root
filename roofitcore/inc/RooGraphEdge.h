/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGraphEdge.rdl $
 * Authors:
 *   Ali Hanks, University of Washington, livelife@u.washington.edu
 * History:
 *   30-Aug-2002 Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#ifndef ROO_GRAPH_EDGE
#define ROO_GRAPH_EDGE

#include "RooFitCore/RooGraphNode.rdl"
#include "TObject.h"

class RooGraphEdge : public TObject
{
private:
  RooGraphNode *fn1;              //nodes the edge
  RooGraphNode *fn2;              //connects together
  TString fes;
  TString ffirstnode;
  TString fsecondnode;

public:
  RooGraphEdge();
  RooGraphEdge(RooGraphNode *n1, RooGraphNode *n2);
  RooGraphEdge(RooGraphNode *n1, RooGraphNode *n2, TString es);
  void Print();
  void Read(ifstream &file);
  void Set1stNode(RooGraphNode *n1);
  void Set2ndNode(RooGraphNode *n2);
  void SetType(TString es) { fes = es; }
  void Connect();
  void Connect(int color);
  void Connect(RooGraphNode *n1, RooGraphNode *n2);
  double GetInitialDistance();
  TObject *GetType(TList *padlist);
  const char* Get1stNode() const { return ffirstnode.Data(); }
  const char* Get2ndNode() const { return fsecondnode.Data(); }
  void SwitchNodes();
  TString GetStyle() const { return fes; }
  double GetX1();
  double GetY1();
  double GetX2();
  double GetY2();
  RooGraphNode *GetStart() { return fn1; }
  RooGraphNode *GetEnd() { return fn2; }

  ClassDef(RooGraphEdge,2)

};
#endif









