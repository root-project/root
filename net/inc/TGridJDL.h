// @(#)root/net:$Name:  $:$Id: TGridJDL.h,v 1.2 2006/05/16 06:28:23 brun Exp $
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridJDL
#define ROOT_TGridJDL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridJDL                                                             //
//                                                                      //
// Abstract base class to generate JDL files for job submission to the  //
// Grid.                                                                //
//                                                                      //
// Related classes are TGLiteJDL                                        //
//                              .                                       //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif


class TGridJDL : public TObject {
protected:
   TMap    fMap;  // stores the key, value pairs of the JDL

public:
   TGridJDL() : fMap() { }
   virtual ~TGridJDL();

   void             SetValue(const char *key, const char *value);
   const char      *GetValue(const char *key);
   TString          AddQuotes(const char *value);
   void             AddToSet(const char *key, const char *value);
   virtual TString  Generate();
   virtual void     Clear(const Option_t* = 0);

   virtual void SetExecutable(const char* value=0) = 0;
   virtual void SetArguments(const char* value=0) = 0;
   virtual void SetEMail(const char* value=0) = 0;
   virtual void SetOutputDirectory(const char* value=0) = 0;
   virtual void SetPrice(UInt_t price=1) = 0;
   virtual void SetTTL(UInt_t ttl=72000) = 0;
   virtual void SetJobTag(const char* jobtag=0) = 0;
   virtual void SetInputDataListFormat(const char* format="xml-single") = 0;
   virtual void SetInputDataList(const char* list="collection.xml") = 0;

   virtual void SetSplitMode(const char* value, UInt_t maxnumberofinputfiles=0, UInt_t maxinputfilesize=0) = 0;
   virtual void SetSplitArguments(const char* splitarguments=0) = 0;
   virtual void SetValidationCommand(const char* value) = 0;

   virtual void AddToInputSandbox(const char* value=0) = 0;
   virtual void AddToOutputSandbox(const char* value=0) = 0;
   virtual void AddToInputData(const char* value=0) = 0;
   virtual void AddToInputDataCollection(const char* value=0) = 0;
   virtual void AddToRequirements(const char* value=0) = 0;
   virtual void AddToPackages(const char* name="AliRoot", const char* version="default",const char* type="VO_ALICE") = 0;
   virtual void AddToOutputArchive(const char* value=0) = 0;
   ClassDef(TGridJDL,1)  // ABC defining interface JDL generator
};

#endif
