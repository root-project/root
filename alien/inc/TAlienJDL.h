// @(#)root/alien:$Name:  $:$Id: TAlienJDL.h,v 1.1 2005/05/20 11:13:30 rdm Exp $
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienJDL
#define ROOT_TAlienJDL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJDL                                                            //
//                                                                      //
// Class which creates JDL files for the alien middleware.              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJDL
#include "TGridJDL.h"
#endif


class TAlienJDL : public TGridJDL {

public:
   TAlienJDL() { }
   virtual ~TAlienJDL() { };

   virtual void SetExecutable(const char* value=0);
   virtual void SetArguments(const char* value=0);
   virtual void SetEMail(const char* value=0);
   virtual void SetOutputDirectory(const char* value=0);
   virtual void SetPrice(UInt_t price=1);
   virtual void SetTTL(UInt_t ttl=72000);
   virtual void SetJobTag(const char* jobtag=0);
   virtual void SetInputDataListFormat(const char* format="xml-single");
   virtual void SetInputDataList(const char* list="collection.xml");

   virtual void SetSplitMode(const char* value, UInt_t maxnumberofinputfiles=0, UInt_t maxinputfilesize=0);
   virtual void SetSplitArguments(const char* splitarguments=0);
   virtual void SetValidationCommand(const char* value);

   virtual void AddToInputSandbox(const char* value=0);
   virtual void AddToOutputSandbox(const char* value=0);
   virtual void AddToInputData(const char* value=0);
   virtual void AddToInputDataCollection(const char* value=0);
   virtual void AddToRequirements(const char* value=0);
   virtual void AddToPackages(const char* name="AliRoot", const char* version="newest",const char* type="VO_ALICE");
   virtual void AddToOutputArchive(const char* value=0);

   Bool_t SubmitTest();

   ClassDef(TAlienJDL,1)  // Creates JDL files for the AliEn middleware
};

#endif
