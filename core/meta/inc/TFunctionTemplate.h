// @(#)root/meta:
// Author: Philippe Canal November 2013.

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFunctionTemplate
#define ROOT_TFunctionTemplate


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFunctionTemplate                                                    //
//                                                                      //
// Dictionary for function template                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif

class TFunctionTemplate : public TDictionary {
protected:
   FuncTempInfo_t *fInfo;  // pointer to Interpreter function template info
   TClass         *fClass; //pointer to the class (if any).

public:
   TFunctionTemplate(FuncTempInfo_t *info, TClass *cl);
   TFunctionTemplate(const TFunctionTemplate &orig);
   TFunctionTemplate& operator=(const TFunctionTemplate &rhs);
   virtual            ~TFunctionTemplate();
   virtual TObject   *Clone(const char *newname="") const;

   DeclId_t            GetDeclId() const;
   UInt_t              GetTemplateNargs() const;
   UInt_t              GetTemplateMinReqArgs() const;

   virtual Bool_t      IsValid();
   Long_t              Property() const;

   virtual bool        Update(FuncTempInfo_t *info);

   ClassDef(TFunctionTemplate,0)  //Dictionary for function template

};

#endif
