/* @(#)root/win32:$Name:  $:$Id: LinkDef.h,v 1.5 2003/11/18 18:41:55 fine Exp $ */

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
 
#pragma link C++ class TGQt;
#pragma link C++ class TQtThread;
#pragma link C++ global gQt;
// #pragma link C++ class TQtGUIFactory;

#endif
