/* @(#)root/proof:$Name:  $:$Id: LinkDef.h,v 1.2 2002/01/18 14:24:09 rdm Exp $ */

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

#pragma link C++ global gProof;
#pragma link C++ global gProofServ;

#pragma link C++ class TProof;
#pragma link C++ class TProofServ;
#pragma link C++ class TSlave;
#pragma link C++ class TProofPlayer+;
#pragma link C++ class TProofPlayerLocal+;
#pragma link C++ class TProofPlayerRemote+;
#pragma link C++ class TProofPlayerSlave+;
#pragma link C++ class TEventIter+;
#pragma link C++ class TEventIterLocal+;
#pragma link C++ class TEventIterSlave+;

#endif
