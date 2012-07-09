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
   TString fGUID;      // GUID
   TString fUrl;       // url for the opened copy
   TString fPfn;       // physical file name
   TString fSE;        // Storage element
   Int_t   fImage;     // Image number
   Int_t   fNreplicas; // Number of replicas
   Long64_t fOpenedAt; // Absolute value for time when opened
   Double_t fElapsed;  // Time elapsed to opem file
public:
   TAlienFile() : TXNetFile(), fLfn(), fAuthz(), fGUID(), fUrl(), fPfn(), fSE(), fImage(0), fNreplicas(0), fOpenedAt(0), fElapsed(0) { }
   TAlienFile(const char *purl, Option_t *option = "",
              const char *ftitle = "", Int_t compress = 1,
              Bool_t parallelopen = kFALSE, const char *lurl = 0,
              const char *authz = 0);
   virtual ~TAlienFile();

   virtual void Close(const Option_t *opt = "");

   const char  *GetGUID() const  {return fGUID;}
   Double_t     GetElapsed() const {return fElapsed;}
   Int_t        GetImage() const {return fImage;}
   const char  *GetLfn() const   {return fLfn;}
   Int_t        GetNreplicas() const {return fNreplicas;}
   Long64_t     GetOpenTime() const {return fOpenedAt;}
   const char  *GetPfn() const   {return fPfn;}
   const char  *GetSE() const    {return fSE;}
   const char  *GetUrl() const   {return fUrl;}

protected:
   void         SetGUID(const char *guid) {fGUID = guid;}
   void         SetElapsed(Double_t real) {fElapsed = real;}
   void         SetImage(Int_t image)   {fImage = image;}
   void         SetNreplicas(Int_t nrep) {fNreplicas = nrep;}
   void         SetPfn(const char *pfn) {fPfn = pfn;}
   void         SetSE(const char *se)   {fSE = se;}
   void         SetUrl(const char *url) {fUrl = url;}

public:
   static TAlienFile *Open(const char *lfn, const Option_t *option = "",
                           const char *title = "", Int_t compress = 1,
                           Bool_t parallelopen = kFALSE);
   static TString     SUrl(const char *lfn);

   ClassDef(TAlienFile, 4)  //A ROOT file that reads/writes via AliEn services and TXNetFile protocol
};

#endif
