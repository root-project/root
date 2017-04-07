// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004
//         Lucia.Jancurova@cern.ch Slovakia 2007
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

#include "TGridJDL.h"


class TAlienJDL : public TGridJDL {

public:
   TAlienJDL() : TGridJDL() { }
   virtual ~TAlienJDL() { }

   virtual void SetExecutable(const char *value=0, const char *description=0);
   virtual void SetArguments(const char *value=0, const char *description=0);
   virtual void SetEMail(const char *value=0, const char *description=0);
   virtual void SetOutputDirectory(const char *value=0, const char *description=0);
   virtual void SetPrice(UInt_t price=1, const char *description=0);
   virtual void SetMergedOutputDirectory(const char *value=0, const char *description=0);
   virtual void SetTTL(UInt_t ttl=72000, const char *description=0);
   virtual void SetJobTag(const char *jobtag=0, const char *description=0);
   virtual void SetInputDataListFormat(const char *format="xml-single", const char *description=0);
   virtual void SetInputDataList(const char *list="collection.xml", const char *description=0);

   virtual void SetSplitMode(const char *value, UInt_t maxnumberofinputfiles=0,
                             UInt_t maxinputfilesize=0, const char *d1=0, const char *d2=0,
                             const char *d3=0);
   virtual void SetSplitModeMaxNumOfFiles(UInt_t maxnumberofinputfiles=0, const char *description=0);
   virtual void SetSplitModeMaxInputFileSize(UInt_t maxinputfilesize=0, const char *description=0);
   virtual void SetSplitArguments(const char *splitarguments=0, const char *description=0);
   virtual void SetValidationCommand(const char *value, const char *description=0);
   virtual void SetMaxInitFailed(Int_t maxInitFailed, const char *description=0);

   virtual void SetOwnCommand(const char *command=0, const char *value=0, const char *description=0);

   virtual void AddToInputSandbox(const char *value=0, const char *description=0);
   virtual void AddToOutputSandbox(const char *value=0, const char *description=0);
   virtual void AddToInputData(const char *value=0, const char *description=0);
   virtual void AddToInputDataCollection(const char *value=0, const char *description=0);
   virtual void AddToRequirements(const char *value=0, const char *description=0);
   virtual void AddToPackages(const char *name/*="AliRoot"*/, const char *version/*="newest"*/,
                              const char *type/*="VO_ALICE"*/, const char *description=0);
   virtual void AddToPackages(const char *name/*="VO_ALICE@AliRoot::newest"*/,
                              const char *description=0);
   virtual void AddToOutputArchive(const char *value=0, const char *description=0);
   virtual void AddToReqSet(const char *key, const char *value=0);

   virtual void AddToMerge(const char *filenameToMerge/*="histograms.root"*/,
                           const char *jdlToSubmit/*="/alice/jdl/mergerootfile.jdl"*/,
                           const char *mergedFile/*="histograms-merged.root"*/,
                           const char *description=0);
   virtual void AddToMerge(const char *merge="histo.root:/alice/jdl/mergerootfile.jdl:histo-merged.root",
                           const char *description=0);

   void         SetValueByCmd(TString cmd, TString value);
   virtual void Parse(const char *filename);
   void         Simulate();

   Bool_t       SubmitTest();

   ClassDef(TAlienJDL,1)  // Creates JDL files for the AliEn middleware
};

#endif
