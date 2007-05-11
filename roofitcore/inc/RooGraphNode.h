/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGraphNode.rdl,v 1.10 2005/12/08 13:19:55 wverkerke Exp $
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

#ifndef ROO_GRAPH_NODE
#define ROO_GRAPH_NODE

#include "Riostream.h"
#include <fstream>
#include "TString.h"
#include "TObject.h"
class TEllipse ;

class RooGraphNode : public TObject {
private:
  double fX1;                  //x coordinate
  double fY1;                  //y coordinate
  double fR1;                  //node width
  double fR2;                  //node length
  TString ftext;               //string of text
  double fnumber;

public:
  RooGraphNode();
  RooGraphNode(double x, double y);
  RooGraphNode(double x, double y, double w, double l, TString text);
  void paint();
  void draw();
  void draw(int color);
  void SetCoords(double x, double y);
  void SetSize(double w, double l);
  void SetText(TString text);
  void print() const;
  void read(ifstream & file);
  void ReadPDF(ifstream & file);
  double GetX1() const { return fX1; }
  double GetY1() const { return fY1; }
  double GetRadius() const { return fR1; }
  virtual const char* GetName() const { return ftext.Data(); }
  void GetNumber(double number);
  void GetValue(double number, TList *padlist, TList *edges);
  TEllipse *GetEllipse(TList *padlist);
  void RemoveE(TList *padlist);
  void RemoveT(TList *padlist);
  void RemoveN(TList *padlist);
  void RemoveEdges(TList *edges, TList *padlist);
  void RedrawEdges(TList *edges);
  void RedrawEdges(TList *edges, int color);
  double GetTotalEChange(TList *nodessprings);
  void GetDxDy(double &dx, double &dy, TList *nodessprings);
  void NodesSprings(TList *springs, TList *nodessprings);
  double GetTotalE(TList *nodessprings, char m);
  double GetTotalE2(TList *nodessprings, char m);
  double GetTotalExy(TList *nodessprings);

  ClassDef(RooGraphNode,1)  //one-D circular node
};
#endif






