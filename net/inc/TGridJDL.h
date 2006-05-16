// @(#)root/net:$Name:  $:$Id: TGridJDL.h,v 1.1 2005/05/12 13:19:39 rdm Exp $
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

   virtual void SetExecutable(const char *value) = 0;
   virtual void SetArguments(const char *value) = 0;
   virtual void SetRequirements(const char *value) = 0;
   virtual void SetEMail(const char *value) = 0;

   virtual void AddToInputSandbox(const char *value) = 0;
   virtual void AddToOutputSandbox(const char *value) = 0;
   virtual void AddToInputData(const char *value) = 0;
   virtual void AddToInputDataCollection(const char *value) = 0;

   virtual void SetSplitMode(const char *value) = 0;
   virtual void SetValidationCommand(const char *value) = 0;

   ClassDef(TGridJDL,1)  // ABC defining interface JDL generator
};

#endif
