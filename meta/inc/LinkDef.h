/* @(#)root/meta:$Name:  $:$Id: LinkDef.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $ */

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


#pragma link C++ enum EProperty;

#pragma link C++ global gInterpreter;

#pragma link C++ class TBaseClass;
#pragma link C++ class TClass;
#pragma link C++ class TDataMember;
#pragma link C++ class TDataType;
#pragma link C++ class TDictionary;
#pragma link C++ class TFunction;
#pragma link C++ class TGlobal;
#pragma link C++ class TMethod;
#pragma link C++ class TMethodArg;
#pragma link C++ class TMethodCall;
#pragma link C++ class TCint;
#pragma link C++ class TInterpreter;

#pragma link C++ class TToggle;
#pragma link C++ class TToggleGroup;

#endif
