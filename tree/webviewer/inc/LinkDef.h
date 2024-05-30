// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006 - 2019

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//==============================================================================
// LinkDef.h - REve objects and services.
//==============================================================================

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


// Geometry viewer
#pragma link C++ class ROOT::RTreeViewer+;
#pragma link C++ struct ROOT::RTreeViewer::RBranchInfo+;
#pragma link C++ struct ROOT::RTreeViewer::RConfig+;

#endif
