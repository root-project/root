// @(#)root/unuran:$Name:  $:$Id: src/TUnuranDistr.cxx,v 1.0 2006/01/01 12:00:00 moneta Exp $
// Author: L. Moneta Wed Sep 27 11:53:27 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TUnuranDistr

#include "TUnuranDistr.h"



// TUnuranDistr::TUnuranDistr() 
// {
//    // Default constructor implementation.
// }

// TUnuranDistr::~TUnuranDistr() 
// {
//    // Destructor implementation.
// }

TUnuranDistr::TUnuranDistr(const TUnuranDistr & rhs) : 
   fFunc(rhs.fFunc),
   fCdf(rhs.fCdf),
   fDeriv(rhs.fDeriv),
   fXmin(rhs.fXmin), fXmax(rhs.fXmax), 
   fHasDomain(rhs.fHasDomain)
{
   // Implementation of copy constructor.
}

TUnuranDistr & TUnuranDistr::operator = (const TUnuranDistr &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
   fFunc = rhs.fFunc;
   fCdf= rhs.fCdf;
   fDeriv = rhs.fDeriv;
   fXmin= rhs.fXmin; 
   fXmax = rhs.fXmax;
   fHasDomain = rhs.fHasDomain;
   return *this;
}

