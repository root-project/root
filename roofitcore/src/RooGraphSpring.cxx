/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooGraphSpring.cxx,v 1.14 2007/05/11 09:11:58 verkerke Exp $
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

#include "RooGraphSpring.h"
#include "RooGraphSpring.h"
#include "TList.h"
#include "TMath.h"
#include "RooGraphEdge.h"

#include "Riostream.h"
#include <fstream>
#include <math.h>

///////////////////////////////////////////////////////////////////////////////
//RooGraphSpring:                                                            //
//                                                                           //
//The spring class is a subset of node class and a spring can only be created//
//if there are two nodes to connect it to. It is not drawn to the canvas, but//
//is used to determine the energy of each node and thus can be used to reduce//
//the energy of the node/edge system to its lowest possible value.           //
///////////////////////////////////////////////////////////////////////////////

ClassImp(RooGraphSpring)

RooGraphSpring::RooGraphSpring()
{
  //Default constructor for a spring
  fn1 = 0;
  fn2 = 0;
}

RooGraphSpring::RooGraphSpring(RooGraphNode *n1, RooGraphNode *n2)
{
  //Spring standard constructor
  fn1 = n1;
  fn2 = n2;
}

void RooGraphSpring::print()
{
  cout << fn1->GetName() << ", " << fn2->GetName() << endl;
}

void RooGraphSpring::read(ifstream &/*file*/)
{
}

void RooGraphSpring::Set1stNode(RooGraphNode *n1)
{
  //Changes the first node the spring is connected to to the given value
  fn1 = n1;
}

void RooGraphSpring::Set2ndNode(RooGraphNode *n2)
{
  //Changes the second node the spring is connected to to the given value
  fn2 = n2;
}

void RooGraphSpring::Connect(RooGraphNode *n1, RooGraphNode *n2)
{
  //Causes the spring to connect the two given nodes
  fn1 = n1;
  fn2 = n2;
  fgraphlength = 1000.0;
}

double RooGraphSpring::GetX1()
{
  //Returns the x value for one endpoint of the node
  const double x1 = fn1->GetX1();
  return x1;
}

double RooGraphSpring::GetY1()
{
  //Returns the y value for one endpoint of the node
  const double y1 = fn1->GetY1();
  return y1;
}

double RooGraphSpring::GetX2()
{
  //Returns the x value for the other endpoint of the node
  const double x2 = fn2->GetX1();
  return x2;
}

double RooGraphSpring::GetY2()
{
  //Returns the y value for the other endpoint fo the node
  const double y2 = fn2->GetY1();
  return y2;
}

void RooGraphSpring::SwitchNodes()
{
  //Changes which node is the first and which is second
  RooGraphNode *n1 = fn1;
  RooGraphNode *n2 = fn2;
  fn1 = n2;
  fn2 = n1;
}

double RooGraphSpring::GetInitialDistance()
{
  //Returns the length of the string
  const double x1 = fn1->GetX1();
  const double y1 = fn1->GetY1();
  const double x2 = fn2->GetX1();
  const double y2 = fn2->GetY1();

  double ilength = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
  return ilength;
}

void RooGraphSpring::SetGraphLength(double length)
{
  fgraphlength = length;
}

double RooGraphSpring::GetLength()
{
  //Returns the spring length used to calculate the energy in the spring
  double L = .5;
  double d = fgraphlength;
  double length = L*d;
  return length;
}

double RooGraphSpring::GetSpringConstant()
{
  //Returns the spring constant for this spring
  double K = 1.0;
  double k;
  double d = fgraphlength;
  k = K/(d*d);
  return k;
}

double RooGraphSpring::GetSpringDEnergy(char m)
{
  //Returns the first derivative of the energy of this spring with respect to
  //either x or y, determined by the input parameter m
  double l = GetLength();
  double k = GetSpringConstant();
  double x1(0);
  double x2(0);
  double y1(0);
  double y2(0);
  if (m == 'x')
    {
      x1 = fn1->GetX1();
      y1 = fn1->GetY1();
      x2 = fn2->GetX1();
      y2 = fn2->GetY1();
    } 
  if (m == 'y')
    {
      y1 = fn1->GetX1();
      x1 = fn1->GetY1();
      y2 = fn2->GetX1();
      x2 = fn2->GetY1();
    }
  double energy = k*((x1-x2)-l*(x1-x2)/sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)));
  return energy;
}

double RooGraphSpring::GetSpringD2Energy(char m)
{
  //Returns the second derivative of the spring energy with respect to either
  //x or y, determined by the input parameter m
  double l = GetLength();
  double k = GetSpringConstant();
  double x1(0);
  double x2(0);
  double y1(0);
  double y2(0);
  if (m == 'y')
    {
      x1 = fn1->GetX1();
      y1 = fn1->GetY1();
      x2 = fn2->GetX1();
      y2 = fn2->GetY1();
    }
  if (m == 'x')
    {
      y1 = fn1->GetX1();
      x1 = fn1->GetY1();
      y2 = fn2->GetX1();
      x2 = fn2->GetY1();
    }
  double n = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
  double energy = k*(1-l*(x1-x2)*(x1-x2)/TMath::Power(n,1.5));
  return energy;
}

double RooGraphSpring::GetSpringDxyEnergy()
{
  //Returns the double derivative of the spring energy with repect to both x&y
  double l = GetLength();
  double k = GetSpringConstant();
  double x1 = fn1->GetX1();
  double y1 = fn1->GetY1();
  double x2 = fn2->GetX1();
  double y2 = fn2->GetY1();
  double n = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
  double energy = k*(l*(x1-x2)*(y1-y2)/TMath::Power(n,1.5));
  return energy;
}
















