// @(#)root/net:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004
// Jancurova.lucia@cern.ch Slovakia  29/9/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
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

#include "TObject.h"
#include "TString.h"
#include "TMap.h"


class TGridJDL : public TObject {
protected:
   TMap    fMap;              // stores the key, value pairs of the JDL
   TMap    fDescriptionMap;   // stores the key, value pairs of the JDL
public:
   TGridJDL() : fMap(), fDescriptionMap() { }
   virtual ~TGridJDL();

   void             SetValue(const char *key, const char *value);
   const char      *GetValue(const char *key);
   void             SetDescription(const char *key, const char *description);
   const char      *GetDescription(const char *key);
   TString          AddQuotes(const char *value);
   void             AddToSet(const char *key, const char *value);
   void             AddToSetDescription(const char *key, const char *description);
   virtual TString  Generate();
   void             Clear(const Option_t* = "") override;

   virtual void SetExecutable(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void SetArguments(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void SetEMail(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void SetOutputDirectory(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void SetPrice(UInt_t price=1, const char *description=nullptr) = 0;
   virtual void SetTTL(UInt_t ttl=72000, const char *description=nullptr) = 0;
   virtual void SetJobTag(const char *jobtag=nullptr, const char *description=nullptr) = 0;
   virtual void SetInputDataListFormat(const char *format="xml-single", const char *description=nullptr) = 0;
   virtual void SetInputDataList(const char *list="collection.xml", const char *description=nullptr) = 0;

   virtual void SetSplitMode(const char *value, UInt_t maxnumberofinputfiles=0,
                             UInt_t maxinputfilesize=0, const char *d1=nullptr,
                             const char *d2=nullptr, const char *d3=nullptr) = 0;
   virtual void SetSplitArguments(const char *splitarguments=nullptr, const char *description=nullptr) = 0;
   virtual void SetValidationCommand(const char *value, const char *description=nullptr) = 0;

   virtual void AddToInputSandbox(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void AddToOutputSandbox(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void AddToInputData(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void AddToInputDataCollection(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void AddToRequirements(const char *value=nullptr, const char *description=nullptr) = 0;
   virtual void AddToPackages(const char *name="AliRoot", const char *version="default",
                              const char *type="VO_ALICE", const char *description=nullptr) = 0;
   virtual void AddToOutputArchive(const char *value=nullptr, const char *description=nullptr) = 0;

   ClassDefOverride(TGridJDL,1)  // ABC defining interface JDL generator
};

#endif
