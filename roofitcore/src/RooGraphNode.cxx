/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooGraphNode.cxx,v 1.17 2007/05/11 09:11:58 verkerke Exp $
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

#include "RooGraphNode.h"
#include "RooGraphNode.h"
#include "TEllipse.h"
#include "TText.h"
#include "TString.h"
#include "RooGraphEdge.h"
#include "RooGraphSpring.h"
#include "TStyle.h"
#include "TList.h"

#include "Riostream.h"
#include <fstream>
#include <iomanip>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

///////////////////////////////////////////////////////////////////////////////
//RooGraphNode:                                                              //
//                                                                           //
//The purpose of this class is to create nodes, based from the TEllipse class//
//that have a position, name, and energy that is drawn on the graph with the //
//node.  The energy is determined by the topology of the system of nodes the //
//node is drawn with and the edges it is attached with and is calculated from//
//a system of strings that attach all the nodes in the system.               //
///////////////////////////////////////////////////////////////////////////////

ClassImp(RooGraphNode)

RooGraphNode::RooGraphNode()
{
  //Node default constructor
  fX1 = 0;
  fY1 = 0;
  fR1 = .03;
  fR2 = .03;
  ftext = "0";
}

RooGraphNode::RooGraphNode(double x, double y)
{
  //Node constructor with default radius and text
  fX1 = x;
  fY1 = y;
  fR1 = .03;
  fR2 = .03;
  ftext = "0";
  fnumber = 0;
}

RooGraphNode::RooGraphNode(double x, double y, double w, double l, TString text) 
{
  //Node standard constructor
  fX1=x;
  fY1=y;
  fR1=w;
  fR2=l;
  ftext=text;
  fnumber= 0;
}

void RooGraphNode::paint()
{
  //Paints Node to canvas by painting an ellipse and text at that location
  TEllipse *e = new TEllipse(fX1,fY1,fR1,fR2);
  e->Paint();
  TText *t = new TText(fX1,fY1,ftext);
  t->SetTextSize(0.02F);
  t->Paint();
  char text[20];
  //int precision = 5;
  sprintf(text,"%.7f",fnumber) ;
  TText *n = new TText(fX1,fY1,text);
  n->SetTextSize(0.03F);
  n->Draw();
}

void RooGraphNode::draw()
{
  //Draws Node to canvas
  TEllipse *e = new TEllipse(fX1,fY1,fR1,fR2);
  e->Draw();
  double y = fY1 + fR1;
  if (ftext != "0")
    {
      TText *t = new TText(fX1,y,ftext);
      t->SetTextSize(0.03F);
      t->Draw();
    }
  if (fnumber != 0)
    { char text[20];
    //int precision = 5;
    sprintf(text,"%.7f",fnumber) ;
    TText *n = new TText(fX1,fY1,text);
    n->SetTextSize(0.03F);
    n->Draw();
    }
}

void RooGraphNode::draw(int color)
{
  //Draws Node to canvas with given color
  TEllipse *e = new TEllipse(fX1,fY1,fR1,fR2);
  e->Draw();
  e->SetLineColor(color);
  double y = fY1 + fR1;
  if (ftext != "0")
    { TText *t = new TText(fX1,y,ftext);
    t->SetTextSize(0.03F);
    t->Draw();
    }
  if (fnumber != 0)
    { char text[20];
    //int precision = 5;
    sprintf(text,"%.7f",fnumber) ;
    TText *n = new TText(fX1,fY1,text);
    n->SetTextSize(0.03F);
    n->Draw();
    }
}

void RooGraphNode::print() const
{
  //outputs location and size of the node to screen
  cout << "x = " << fX1 << ", y = " << fY1 << endl;
  cout << "radius = " << fR1 << endl;
}

void RooGraphNode::read(ifstream &file)
{
  //Reads the properties of the node from a file of a set format
  double ix;
  double iy;
  double iw;
  double il;
  TString itext;
  char eol;
  file.seekg(4, ios::cur);
  file >> ix >> iy >> iw >> il >> itext;
  file.get(eol);

  fX1 = ix;
  fY1 = iy;
  fR1 = iw;
  fR2 = il;
  ftext = itext;
}

void RooGraphNode::ReadPDF(ifstream &file)
{
  //Reads the properties of the node from a file created from the PDF info
  TString itext;
  double value;
  char equals;
  file >> itext >> equals >> value;
  char line[50];
  file.getline(line, 50);
  ftext = itext;
  fnumber = value;
}

void RooGraphNode::SetCoords(double x, double y)
{
  //Changes the coordinates of the node to the given values
  fX1 = x;
  fY1 = y;
}

void RooGraphNode::SetSize(double w, double l)
{
  //Sets the size of the node to the given values
  fR1 = w;
  fR2 = l;
}

void RooGraphNode::SetText(TString text)
{
  //Changes the name and attached string of the node to given string
  ftext = text;
}

void RooGraphNode::GetNumber(double /*number*/)
{
  //draws the given number value to the screen at the node location.
  //This is a number or value associated with the node.
  char text[20];
  // int precision = 5;
  sprintf(text,"%.7f",fnumber) ;
  TText *t = new TText(fX1,fY1,text);
  t->SetTextSize(0.03F);
  t->Draw();
}

void RooGraphNode::GetValue(double number, TList *padlist, TList *edges)
{
  RemoveE(padlist);
  RemoveN(padlist);
  RemoveEdges(edges, padlist);
  draw(4);
  RedrawEdges(edges, 4);
  GetNumber(number);
}

TEllipse *RooGraphNode::GetEllipse(TList *padlist)
{
  //Returns the ellipse that was created by drawing this ellipse to the current
  //canvas by finding it from the list of canvas objects.
  TObject *obj = padlist->First();
  TEllipse *e = 0;
  while(obj != 0)
    {
      if (obj->InheritsFrom("TEllipse"))
	{ 
	  e = dynamic_cast<TEllipse*>(obj);
	  double x = e->GetX1();
	  double y = e->GetY1();
	  if (x==fX1&&y==fY1)
	    { break; }
	}
      obj = padlist->After(obj);
    }

	return e;
}

void RooGraphNode::RemoveT(TList *padlist)
{
  //Finds and removes the text associated with this node from the list of
  //canvas objects
  TObject *obj = padlist->First();
  while(obj != 0)
    {
      if (obj->InheritsFrom("TText"))
	{ 
	  TText *txt = dynamic_cast<TText*>(obj);
	  double x = txt->GetX();
	  double y = txt->GetY();
	  double y1 = fY1 + fR1;
	  if ((x==fX1)&&(y==y1))
	    { cout << y;
	    cout << ", " << y1 << endl;
	    padlist->Remove(txt); }
	}
      obj = padlist->After(obj);
    }
}

void RooGraphNode::RemoveN(TList *padlist)
{
  //Finds the number associated with this node, created from GetNumber, and
  //removes it from the canvas by finding it in the list of canvas objects
  TObject *obj = padlist->First();
  while(obj != 0)
    {
      if (obj->InheritsFrom("TText"))
	{ 
	  TText *txt = dynamic_cast<TText*>(obj);
	  double x = txt->GetX();
	  double y = txt->GetY();
	  if (x==fX1&&y==fY1)
	    { padlist->Remove(txt); }
	}
      obj = padlist->After(obj);
    }
}

void RooGraphNode::RemoveE(TList *padlist)
{
  //Removes the ellipse associated with this node from the canvas
  //This is the only way to redraw the Node in a new location
  TEllipse *e = GetEllipse(padlist);
  padlist->Remove(e);
}

void RooGraphNode::RemoveEdges(TList *edges, TList *padlist)
{
  //Finds all the edges associated with this node by searching a list of all
  //the edges on the canvas, and then finds them in the list of canvas objects
  //and removes them from the canvas.
  //This is the only way the edges can be redrawn in new location
  RooGraphEdge *edge = dynamic_cast<RooGraphEdge*>(edges->First());
  while (edge != 0)
    {
      if ((fX1==edge->GetX1()&&fY1==edge->GetY1())||(fX1==edge->GetX2()&&fY1==edge->GetY2()))
	{
	  TObject *obj = edge->GetType(padlist);
	  padlist->Remove(obj);
	}
      edge = dynamic_cast<RooGraphEdge*>(edges->After(edge));
    }
}

void RooGraphNode::RedrawEdges(TList *edges)
{
  //Redraws the edges associated with this node.
  RooGraphEdge *edge = dynamic_cast<RooGraphEdge*>(edges->First());
  while(edge != 0)
    {
      double x1 = edge->GetX1();
      double y1 = edge->GetY1();
      double x2 = edge->GetX2();
      double y2 = edge->GetY2();
      if ((x1==fX1||x2==fX1)&&(y1==fY1||y2==fY1))
	{ edge->Connect(); }
      edge = dynamic_cast<RooGraphEdge*>(edges->After(edge));
    }
}

void RooGraphNode::RedrawEdges(TList *edges, int color)
{
  //Redraws the edges associated with this node in the given color
  RooGraphEdge *edge = dynamic_cast<RooGraphEdge*>(edges->First());
  while(edge != 0)
    {
      double x1 = edge->GetX1();
      double y1 = edge->GetY1();
      double x2 = edge->GetX2();
      double y2 = edge->GetY2();
      if ((x1==fX1||x2==fX1)&&(y1==fY1||y2==fY1))
	{ edge->Connect(color); }
      edge = dynamic_cast<RooGraphEdge*>(edges->After(edge));
    }
}

void RooGraphNode::NodesSprings(TList *springs, TList *nodessprings)
{
  //Finds all the springs associated witht this node from the list of springs
  //existing and puts them into a list.
  RooGraphSpring *spring = dynamic_cast<RooGraphSpring*>(springs->First());
  while (spring != 0)
    {
      if (fX1==spring->GetX1()&&fY1==spring->GetY1())
	{
	  nodessprings->AddLast(spring);     //adds this spring to the list
	}
      if (fX1==spring->GetX2()&&fY1==spring->GetY2())
	{
	  spring->SwitchNodes();             //makes this node first node
	  nodessprings->AddLast(spring);     //adds this spring to the list
	}
      spring = dynamic_cast<RooGraphSpring*>(springs->After(spring));
    }
}

double RooGraphNode::GetTotalEChange(TList *nodessprings)
{
  //Returns the total change in energy for this node
  RooGraphSpring *spring=dynamic_cast<RooGraphSpring*>(nodessprings->First());
  double tex = 0;
  double tey = 0;
  while (spring != 0)
    {
      double ex = spring->GetSpringDEnergy('x');
      double ey = spring->GetSpringDEnergy('y');
      tex = tex + ex;
      tey = tey + ey;
      spring = dynamic_cast<RooGraphSpring*>(nodessprings->After(spring));
    }
  double echange = sqrt(tex*tex + tey*tey);
  return echange;
}

double RooGraphNode::GetTotalE(TList *nodessprings, char m)
{
  //Returns the derivative of the energy of this node with respect to either 
  //x or y, which is determined by the input character m
  RooGraphSpring *spring=dynamic_cast<RooGraphSpring*>(nodessprings->First());
  double tex = 0;
  double tey = 0;
  while (spring != 0)
    {
      double ex = spring->GetSpringDEnergy('x');
      double ey = spring ->GetSpringDEnergy('y');
      tex = tex + ex;
      tey = tey + ey;
      spring = dynamic_cast<RooGraphSpring*>(nodessprings->After(spring));
    }

  double return_value = 0;
  if (m=='x')
    {
      return_value = tex;
    }
  if (m=='y')
    {
      return_value = tey;
    }

	return return_value;
}

double RooGraphNode::GetTotalE2(TList *nodessprings, char m)
{
  //Returns the second derivative of the energy of the node with respect to
  //x or y, determined by the input character m
  RooGraphSpring *spring=dynamic_cast<RooGraphSpring*>(nodessprings->First());
  double tex = 0;
  double tey = 0;
  while (spring != 0)
    {
      double ex = spring->GetSpringD2Energy('x');
      tex = tex + ex;
      double ey = spring->GetSpringD2Energy('y');
      tey = tey + ey;
      spring = dynamic_cast<RooGraphSpring*>(nodessprings->After(spring));
    }

  double return_value = 0;
  if (m=='x')
    {
      return_value = tex;	  
    }
  if (m=='y')
    {
      return_value = tey;	  
    }

	return return_value;
}

double RooGraphNode::GetTotalExy(TList *nodessprings)
{
  //Returns the double derivative of the energy of the node with respect to
  //both x and y
  RooGraphSpring *spring=dynamic_cast<RooGraphSpring*>(nodessprings->First());
  double texy = 0;
  while (spring != 0)
    {
      double exy = spring->GetSpringDxyEnergy();
      texy = texy + exy;
      spring = dynamic_cast<RooGraphSpring*>(nodessprings->After(spring));
    }
  return texy;
}

void RooGraphNode::GetDxDy(double &dx, double &dy, TList *nodessprings)
{
  //Solves for two parameters, dx and dy, which are the change in the possition
  //of the node necessary to reduce it's energy
  double ex = GetTotalE(nodessprings, 'x');
  double ey = GetTotalE(nodessprings, 'y');
  double exd = GetTotalE2(nodessprings, 'x');
  double eyd = GetTotalE2(nodessprings, 'y');
  double exy = GetTotalExy(nodessprings);
  dx = ((exy*ey-eyd*ex)/(eyd*exd-exy*exy));
  dy = ((exd*ey-exy*ex)/(exy*exy-exd*eyd));
}












