// @(#)root/alien:$Name:  $:$Id: TAlienJDL.h,v 1.3 2004/11/01 17:38:08 jgrosseo Exp $
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

   virtual void SetExecutable(const char* value);
   virtual void SetArguments(const char* value);
   virtual void SetRequirements(const char* value);
   virtual void SetEMail(const char* value);

   virtual void AddToInputSandbox(const char* value);
   virtual void AddToOutputSandbox(const char* value);
   virtual void AddToInputData(const char* value);
   virtual void AddToInputDataCollection(const char* value);

   virtual void SetSplitMode(const char* value);
   virtual void SetValidationCommand(const char* value);

   Bool_t SubmitTest();

   ClassDef(TAlienJDL,1)  // Creates JDL files for the AliEn middleware
};

#endif
