/* @(#)root/html:$Name:  $:$Id: LinkDef.h,v 1.1.1.1 2000/05/16 17:00:43 rdm Exp $ */

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

#pragma link C++ global gHtml;

#pragma link C++ nestedclass;
#pragma link C++ class THtml;
#pragma link C++ class THtml::TParseStack;
#pragma link C++ class THtml::TParseStack::TParseElement;
#pragma link C++ class THtml::TDocElement;
#pragma link C++ enum THtml::TParseStack::EContext;
#pragma link C++ enum THtml::TParseStack::EBlockSpec;
#pragma link C++ class THtml::TLocalType;

#endif
