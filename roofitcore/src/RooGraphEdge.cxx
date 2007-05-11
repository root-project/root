/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGraphEdge.cc,v 1.13 2005/12/08 13:19:55 wverkerke Exp $
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

#include "RooFit.h"

#include "RooGraphEdge.h"
#include "RooGraphEdge.h"
#include "TLine.h"
#include "TArrow.h"
#include "TList.h"
#include "TMath.h"

#include "Riostream.h"
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
//RooGraphEdge:                                                              //
//                                                                           //
//The edge class is a subset of the node class and an edge can only be made  //
//if there are two nodes to connect it to. It is used to show how the nodes  //
//depend on each other, and to help display the topology of the graph.  They //
//are also necessary for finding the graph lenght for the springs.           //
///////////////////////////////////////////////////////////////////////////////

ClassImp(RooGraphEdge)

RooGraphEdge::RooGraphEdge()
{
  //Default constructor for an edge
  fn1 = 0;
  fn2 = 0;
  fes = "aLine";
}

RooGraphEdge::RooGraphEdge(RooGraphNode *n1, RooGraphNode *n2)
{
  //Edge standard constructor with default edge type
  fn1 = n1;
  fn2 = n2;
  fes = "aLine";
}

RooGraphEdge::RooGraphEdge(RooGraphNode *n1, RooGraphNode *n2, TString es)
{
  //Edge standard constructor
  fn1 = n1;
  fn2 = n2;
  fes = es;
}

void RooGraphEdge::print()
{
  //prints the names of the nodes the edge is connected to to the screen
  cout << fn1->GetName() << ", " << fn2->GetName() << endl;
}

void RooGraphEdge::read(ifstream &file)
{
  //gets the information needed to draw an edge from a file of a special format
  TString ies;
  TString firstnode;
  TString secondnode;

  file.seekg(4, ios::cur);
  file >> ies >> firstnode >> secondnode;
  char eol;
  file.get(eol);
  fes = ies;
  ffirstnode = firstnode;
  fsecondnode = secondnode;
}

void RooGraphEdge::Set1stNode(RooGraphNode *n1)
{
  //sets the first node for the edge
  fn1 = n1;
}

void RooGraphEdge::Set2ndNode(RooGraphNode *n2)
{
  //sets the second node for the edge
  fn2 = n2;
}

void RooGraphEdge::Connect()
{
  //draw the edge to the canvas in the form of either a line, or arrow
  double x1 = fn1->GetX1();
  double y1 = fn1->GetY1();
  double x2 = fn2->GetX1();
  double y2 = fn2->GetY1();
  if (fes=="aLine"){
    TLine *l = new TLine(x1,y1,x2,y2);
    l->Draw();
  }
  if (fes=="Arrow"){
    TArrow *a = new TArrow(x1,y1,x2,y2,0.02F,"|>");
    a->Draw();
  }
}
void RooGraphEdge::Connect(int color)
{
  //draws the egde to the canvas with the given color
  double x1 = fn1->GetX1();
  double y1 = fn1->GetY1();
  double x2 = fn2->GetX1();
  double y2 = fn2->GetY1();
  if (fes=="aLine"){
    TLine *l = new TLine(x1,y1,x2,y2);
    l->SetLineColor(color);
    l->Draw();
  }
  if (fes=="Arrow"){
    TArrow *a = new TArrow(x1,y1,x2,y2,0.02F,"|>");
    a->SetLineColor(color);
    a->Draw();
  }
}

void RooGraphEdge::Connect(RooGraphNode *n1, RooGraphNode *n2)
{
  //draws the edge to the screen with the given imput nodes
  fn1 = n1;
  fn2 = n2;
  double x1 = fn1->GetX1();
  double y1 = fn1->GetY1();
  double x2 = fn2->GetX1();
  double y2 = fn2->GetY1();
  if (fes=="aLine"){
    TLine *l = new TLine(x1,y1,x2,y2);
    l->Draw();
  }
  if (fes=="Arrow"){
    TArrow *a = new TArrow(x1,y1,x2,y2,0.02F,"|>");
    a->Draw();
  }
}

double RooGraphEdge::GetInitialDistance()
{
  //Returns the length of the edge
  const double x1 = fn1->GetX1();
  const double y1 = fn1->GetY1();
  const double x2 = fn2->GetX1();
  const double y2 = fn2->GetY1();

  double ilength = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
  return ilength;
}

TObject *RooGraphEdge::GetType(TList *padlist)
{
  //returns the object that represents the edge on the canvas
  TObject *obj = padlist->First();
  while(obj != 0)
    {
      if (obj->InheritsFrom("TLine"))
	{
	  TLine *line = dynamic_cast<TLine*>(obj);
	  double x1 = line->GetX1();
	  double y1 = line->GetY1();
	  double x2 = line->GetX2();
	  double y2 = line->GetY2();
	  double X1 = fn1->GetX1();
	  double Y1 = fn1->GetY1();
	  double X2 = fn2->GetX1();
	  double Y2 = fn2->GetY1();
	  if (x1==X1&&y1==Y1&&x2==X2&&y2==Y2)
	    { obj = line; break; }
	}
      if (obj->InheritsFrom("TArrow"))
	{
	  TArrow *arrow = dynamic_cast<TArrow*>(obj);
	  double x1 = arrow->GetX1();
	  double y1 = arrow->GetY1();
	  double x2 = arrow->GetX2();
	  double y2 = arrow->GetY2();
	  double X1 = fn1->GetX1();
	  double Y1 = fn1->GetY1();
	  double X2 = fn2->GetX1();
	  double Y2 = fn2->GetY1();
	  if (x1==X1&&y1==Y1&&x2==X2&&y2==Y2)
	    { obj = arrow; break; }
	}
      obj = padlist->After(obj);
    }

	return obj;
}

double RooGraphEdge::GetX1()
{
  //returns the value for x1
  const double x1 = fn1->GetX1();
  return x1;
}

double RooGraphEdge::GetY1()
{
  //returns the value for y1
  const double y1 = fn1->GetY1();
  return y1;
}

double RooGraphEdge::GetX2()
{
  //returns the value for x2
  const double x2 = fn2->GetX1();
  return x2;
}

double RooGraphEdge::GetY2()
{
  //returns the value for y2
  const double y2 = fn2->GetY1();
  return y2;
}

void RooGraphEdge::SwitchNodes()
{
  //switches the nodes of the edge
  RooGraphNode *n1 = fn1;
  RooGraphNode *n2 = fn2;
  fn1 = n2;
  fn2 = n1;
}
































