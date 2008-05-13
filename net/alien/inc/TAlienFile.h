// @(#)root/alien:$Id$
// Author: Andreas Peters 11/09/2003

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienFile
#define ROOT_TAlienFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienFile                                                           //
//                                                                      //
// A TAlienFile is like a normal TFile except that it reads and writes  //
// it's data via TXNetFile and gets authorization and the TXNetFile     //
// URL from an alien service.                                           //
//                                                                      //
// Filenames are standard URL format with protocol "alien".             //
// The following are valid TAlienFile URL's:                            //
//                                                                      //
//    alien:///alice/cern.ch/user/p/peters/test.root                    //
//    /alien/alice/cern.ch/user/p/peters/test.root                      //
//                                                                      //
//    - notice that URLs like /alien/alice... are converted internally  //
//      to alien://alice...                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TXNetFile
#include "TXNetFile.h"
#endif


class TUrl;

class TAlienFile : public TXNetFile {

private:
   TString fLfn;       // logical file name
   TString fAuthz;     // authorization envelope

public:
   TAlienFile() : TXNetFile(), fLfn(), fAuthz() { }
   TAlienFile(const char *purl, Option_t *option = "",
              const char *ftitle = "", Int_t compress = 1,
              Bool_t parallelopen = kFALSE, const char *lurl = 0,
              const char *authz = 0);
   virtual ~TAlienFile();

   virtual void Close(const Option_t *opt = "");

   static TAlienFile *Open(const char *lfn, const Option_t *option = "",
                           const char *title = "", Int_t compress = 1,
                           Bool_t parallelopen = kFALSE);
   static TString     SUrl(const char *lfn);

   ClassDef(TAlienFile, 3)  //A ROOT file that reads/writes via AliEn services and TXNetFile protocol
};

#endif
