/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGraphSpring.rdl $
 * Authors:
 *   Ali Hanks, University of Washington, livelife@u.washington.edu
 * History:
 *   30-Aug-2002 Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#ifndef ROO_GRAPH_SPRING
#define ROO_GRAPH_SPRING

#include "RooFitCore/RooGraphNode.rdl"
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
  void Print();
  void Read(ifstream &file);
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




