/* @(#)root/treeplayer:$Name:  $:$Id: LinkDef.h,v 1.9 2003/01/15 18:43:45 brun Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TTreePlayer+;
#pragma link C++ class TTreeFormula-;
#pragma link C++ class TSelectorDraw;
#pragma link C++ class TTreeFileMap+;
#pragma link C++ class TFileMap+;
#pragma link C++ class TTreeFormulaManager;

#endif
