/* @(#)root/treeplayer:$Name:  $:$Id: LinkDef.h,v 1.3 2000/11/21 20:52:54 brun Exp $ */

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
#pragma link C++ class TPacketGenerator;
#pragma link C++ class TTreeFormula-;
#pragma link C++ class TPlayer+;
#pragma link C++ class TPlayerLocal+;
#pragma link C++ class TPlayerRemote+;
#pragma link C++ class TPlayerSlave+;
#pragma link C++ class TEventIter+;
#pragma link C++ class TDSet+;

#endif
