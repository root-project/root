// @(#)root/base:$Name:  $:$Id$
// Author: Maarten Ballintijn   21/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParameter
#define ROOT_TParameter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParameter<AParamType>                                               //
//                                                                      //
// Named parameter, streamable and storable.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


template <class AParamType>
class TParameter : public TObject {

private:
   TString     fName;
   AParamType  fVal;

public:
   TParameter() { /* leave fVal uninitialized */ }
   TParameter(const char *name, const AParamType &val)
             : fName(name), fVal(val) { }
   ~TParameter() { }

   const char        *GetName() const { return fName; }
   const AParamType   GetVal() const { return fVal; }

   ClassDef(TParameter,1)  //Basic ROOT object
};

#endif
