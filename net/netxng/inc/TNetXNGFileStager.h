// @(#)root/netxng:$Id$
/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNetXNGFileStager
#define ROOT_TNetXNGFileStager

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TNetXNGFileStager                                                          //
//                                                                            //
// Authors: Lukasz Janyst, Justin Salmon                                      //
//          CERN, 2013                                                        //
//                                                                            //
// Enables access to XRootD staging capabilities using the new client.        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TFileStager.h"

class TCollection;
class TNetXNGSystem;
class TFileCollection;

class TNetXNGFileStager: public TFileStager {

private:
   TNetXNGSystem *fSystem; // Used to access filesystem interface

public:
   TNetXNGFileStager(const char *url = "");
   virtual ~TNetXNGFileStager();

   Bool_t IsStaged(const char *path);
   Int_t  Locate(const char *path, TString &endpath);
   Int_t  LocateCollection(TFileCollection *fc, Bool_t addDummyUrl = kFALSE);
   Bool_t Matches(const char *s);
   Bool_t Stage(const char *path, Option_t *opt = 0);
   Bool_t Stage(TCollection *pathlist, Option_t *opt = 0);
   Bool_t IsValid() const { return (fSystem ? kTRUE : kFALSE); }

private:
   UChar_t ParseStagePriority(Option_t *opt);

   ClassDef( TNetXNGFileStager, 0 ) //! Interface to a 'XRD' staging
};

#endif
