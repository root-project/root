// @(#)root/io:$Id$
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMakeProject
#define ROOT_TMakeProject


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMakeProject                                                         //
//                                                                      //
// Helper class implementing the TFile::MakeProject.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#ifndef ROOT_TOBJECT_H
#include "TString.h"
#endif


class TMakeProject
{
public:
   static void AddUniqueStatement(FILE *fp, const char *statement, char *inclist);
   static void AddInclude(FILE *fp, const char *header, Bool_t system, char *inclist);
   static TString GetHeaderName(const char *name, Bool_t includeNested = kFALSE);
   static UInt_t GenerateClassPrefix(FILE *fp, const char *clname, Bool_t top, TString &protoname, UInt_t *numberOfClasses, Bool_t implementEmptyClass = kFALSE); 
   static UInt_t GenerateEmptyNestedClass(FILE *fp, const char *topclass, const char *clname);
   static UInt_t GenerateForwardDeclaration(FILE *fp, const char *clname, char *inclist, Bool_t implementEmptyClass = kFALSE);
   static UInt_t GenerateIncludeForTemplate(FILE *fp, const char *clname, char *inclist, Bool_t forward);

};

#endif // ROOT_TMakeProject

