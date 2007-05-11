/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGraphEdge.rdl,v 1.9 2005/06/20 15:44:53 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_GRAPH_EDGE
#define ROO_GRAPH_EDGE

#include "RooGraphNode.h"
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
  void print();
  void read(ifstream &file);
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









