// @(#)root/mathcore:$Name:  $:$Id: LinkDef.h,v 1.1 2005/09/18 17:33:47 brun Exp $


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;


#include "LinkDef_Func.h" 
#include "LinkDef_GenVector.h" 

#endif
