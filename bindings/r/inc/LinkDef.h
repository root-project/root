// @(#)root/mpi:$Id: LinkDef.h  -- :: $
// Author: Omar Zapata  Omar.Zapata@cern.ch   29/05/2013

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

//classes
#pragma link C++ class ROOT::R::TRFunctionExport+;

#pragma link C++ class ROOT::R::TRFunctionImport+;

#pragma link C++ class ROOT::R::TRInterface+;

#pragma link C++ class ROOT::R::TRObject+;

#pragma link C++ class ROOT::R::TRDataFrame+;

#endif
