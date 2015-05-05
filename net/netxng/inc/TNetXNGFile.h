// @(#)root/netxng:$Id$
/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNetXNGFile
#define ROOT_TNetXNGFile

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TNetXNGFile                                                                //
//                                                                            //
// Authors: Justin Salmon, Lukasz Janyst                                      //
//          CERN, 2013                                                        //
//                                                                            //
// Enables access to XRootD files using the new client.                       //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "TFile.h"
#include "TSemaphore.h"
#ifndef __CINT__
#include <XrdSys/XrdSysPthread.hh>
#include <XrdCl/XrdClFileSystem.hh>
#include <XrdCl/XrdClXRootDResponses.hh>
#endif

namespace XrdCl {
   class File;
   class ResponseHandler;
}

class TNetXNGFile: public TFile {
private:
#ifndef __CINT__
   XrdCl::File            *fFile;        // Underlying XRootD file
   XrdCl::URL             *fUrl;         // URL of the current file
   XrdCl::OpenFlags::Flags fMode;        // Open mode of the current file
   XrdSysCondVar          *fInitCondVar; // Used to block an async open request
                                         // if requested
   Int_t                   fReadvIorMax; // Max size of a single readv chunk
   Int_t                   fReadvIovMax; // Max number of readv chunks
#endif
   Int_t                   fQueryReadVParams;
   TString                 fNewUrl;

public:
   TNetXNGFile() : TFile(),
         fFile(0), fUrl(0), fMode(XrdCl::OpenFlags::None), fInitCondVar(0),
         fReadvIorMax(0), fReadvIovMax(0){}
   TNetXNGFile(const char *url, Option_t *mode = "", const char *title = "",
         Int_t compress = 1, Int_t netopt = 0, Bool_t parallelopen = kFALSE);
   virtual ~TNetXNGFile();

   virtual void     Init(Bool_t create);
   virtual void     Close(const Option_t *option = "");
   virtual void     Seek(Long64_t offset, ERelativeTo position = kBeg);
   virtual void     SetAsyncOpenStatus(EAsyncOpenStatus status);
   virtual Long64_t GetSize() const;
   virtual Int_t    ReOpen(Option_t *modestr);
   virtual Bool_t   IsOpen() const;
   virtual Bool_t   WriteBuffer(const char *buffer, Int_t length);
   virtual void     Flush();
   virtual Bool_t   ReadBuffer(char *buffer, Int_t length);
   virtual Bool_t   ReadBuffer(char *buffer, Long64_t position, Int_t length);
   virtual Bool_t   ReadBuffers(char *buffer, Long64_t *position, Int_t *length,
                                Int_t nbuffs);
   virtual TString  GetNewUrl() { return fNewUrl; }

private:
   virtual Bool_t IsUseable() const;
   virtual Bool_t GetVectorReadLimits();
   virtual void   SetEnv();
#ifndef __CINT__
   Int_t ParseOpenMode(Option_t *in, TString &modestr,
                       XrdCl::OpenFlags::Flags &mode, Bool_t assumeRead);
#endif

   TNetXNGFile(const TNetXNGFile &other);             // Not implemented
   TNetXNGFile &operator =(const TNetXNGFile &other); // Not implemented

   ClassDef( TNetXNGFile, 0 ) // ROOT class definition
};

#endif // ROOT_TNetXNGFile
