/* @(#)root/hist:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Foption
#define ROOT_Foption


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Foption                                                              //
//                                                                      //
// Histogram fit options structure.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


struct Foption_t {
//*-*      chopt may be the concatenation of the following options:
//*-*      =======================================================
//*-*
//*-*   The following structure members are set to 1 if the option is selected:
   int Quiet;       // "Q"  Quiet mode. No print
   int Verbose;     // "V"  Verbose mode. Print results after each iteration
   int Bound;       // "B"  Some or all parameters are bounded
   int Like;        // "L"  Use Log Likelihood. Default is chisquare method
   int User;        // "U"  Use a User specified fitting algorithm (via SetFCN)
   int W1;          // "W"  Set all the weights to 1. Ignore error bars
   int Errors;      // "E"  Performs a better error evaluation, calling HESSE and MINOS
   int More;        // "M"  Improve fit results.
   int Range;       // "R"  Use the range stored in function
   int Gradient;    // "G"  Option to compute derivatives analytically
   int Nostore;     // "N"  If set, do not store the function graph
   int Nograph;     // "0"  If set, do not display the function graph
   int Plus;        // "+"  Add new function (default is replace)
   int Integral;    // "I"  Use function integral instead of function in center of bin
};

#endif
