// @(#)root/memstat:$Name$:$Id: run_test.C 143 2008-06-17 16:15:42Z anar $
// Author: Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This is a simple demo of TMemStat.                                   //
//                                                                      //
// Usage:                                                               //
//      root [0] .x MemStat.C                                           //
//                                                                      //
// This subroutine calls the "MemStatLeak.C" script, which produces a   //
// data file with all collected memory-wise information.                //
// Then the TMemStat object is created and a report is called.          //
// The report will show all detected memory allocations according  to   //
// stack traces.                                                        //
// At the end the TMemStat GUI is called.                               //
//                                                                      //
// This script works ONLY on Linux for the time being                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void MemStat()
{
   // calling a "leaker"
   gROOT->Macro("MemStatLeak.C+");
   // Creating the TMemStat object in "read" mode
   TMemStat mem;
   // Printing a report
   mem.Report();
   // a GUI of TMemStat
   TMemStatViewerGUI::ShowGUI();
}
