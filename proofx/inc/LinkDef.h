/* @(#)root/netx:$Name:  $:$Id: LinkDef.h,v 1.2 2006/04/18 10:34:35 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TXHandler;
#pragma link C++ class TXProofMgr;
#pragma link C++ class TXSlave;
#pragma link C++ class TXSocket;
#ifndef WIN32
#pragma link C++ class TXProofServ;
#pragma link C++ class TXUnixSocket;
#endif

#endif
