// @(#)root/proof:$Id$
// Author: Paul Nilsson   7/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofResources
#define ROOT_TProofResources

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofResources                                                      //
//                                                                      //
// Abstract base class for PROOF resources.                             //
// The class contains common method declarations for derived classes    //
// such as TProofResourcesStatic which reads and interprets static      //
// config files, and returns master, submaster and worker information   //
// using TProofNodeInfo objects.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TList;
class TString;
class TProofNodeInfo;


class TProofResources : public TObject {

protected:
   Bool_t  fValid;    // kTRUE if resource information was processed correctly

public:
   TProofResources() : fValid(kFALSE) { }
   virtual ~TProofResources() { }

   virtual TProofNodeInfo *GetMaster() = 0;
   virtual TList          *GetSubmasters() = 0;
   virtual TList          *GetWorkers() = 0;
   virtual Bool_t          IsValid() const { return fValid; }

   ClassDef(TProofResources,0)  // Abstract class describing PROOF resources
};

#endif
