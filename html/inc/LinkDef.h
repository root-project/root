/* @(#)root/html:$Name:  $:$Id: LinkDef.h,v 1.4 2007/02/07 20:40:38 brun Exp $ */

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

#pragma link C++ class THtml;
#pragma link C++ class TDocParser;
#pragma link C++ class TDocOutput;
#pragma link C++ class TDocDirective;
#pragma link C++ class TDocHtmlDirective;
#pragma link C++ class TDocMacroDirective;
#pragma link C++ class TDocLatexDirective;
#pragma link C++ class TClassDocOutput;
#pragma link C++ class TClassDocInfo;
#pragma link C++ class TModuleDocInfo;
#pragma link C++ class TLibraryDocInfo;
#endif
