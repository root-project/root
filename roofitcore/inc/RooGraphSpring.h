/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGraphSpring.rdl,v 1.9 2005/06/20 15:44:53 wverkerke Exp $
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

#ifndef ROO_GRAPH_SPRING
#define ROO_GRAPH_SPRING

#include "RooGraphNode.h"
#include "TObject.h"

class RooGraphSpring : public TObject
{
private:
  RooGraphNode *fn1;              //nodes the edge
  RooGraphNode *fn2;              //connects together
  double fgraphlength;

public:
  RooGraphSpring();
  RooGraphSpring(RooGraphNode *n1, RooGraphNode *n2);
  void print();
  void read(ifstream &file);
  void Set1stNode(RooGraphNode *n1);
  void Set2ndNode(RooGraphNode *n2);
  void Connect(RooGraphNode *n1, RooGraphNode *n2);
  double GetX1();
  double GetY1();
  double GetX2();
  double GetY2();
  RooGraphNode *GetStart() { return fn1; }
  RooGraphNode *GetEnd() { return fn2; }
  void SwitchNodes();
  double GetInitialDistance();
  void SetGraphLength(double length);
  double GetGraphLength() { return fgraphlength; }
  double GetLength();
  double GetSpringConstant();
  double GetSpringDEnergy(char m);
  double GetSpringD2Energy(char m);
  double GetSpringDxyEnergy();

  ClassDef(RooGraphSpring,2)
};
#endif




