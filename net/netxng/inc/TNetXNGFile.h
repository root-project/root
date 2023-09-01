
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
#ifndef __CLING__
#include <XrdCl/XrdClFileSystem.hh>
#endif

namespace XrdCl {
   class File;
   class URL;
}
class XrdSysCondVar;

#ifdef __CLING__
namespace XrdCl {
   struct OpenFlags {
      enum    Flags {None = 0};
   };
}
#endif

class TNetXNGFile: public TFile {
private:
   XrdCl::File            *fFile;        // Underlying XRootD file
   XrdCl::URL             *fUrl;         // URL of the current file
   XrdCl::OpenFlags::Flags fMode;        // Open mode of the current file
   XrdSysCondVar          *fInitCondVar; // Used to block an async open request
   // if requested
   Int_t                   fReadvIorMax; // Max size of a single readv chunk
   Int_t                   fReadvIovMax; // Max number of readv chunks
   Int_t                   fQueryReadVParams;
   TString                 fNewUrl;

public:
   TNetXNGFile() : TFile(),
      fFile(nullptr), fUrl(nullptr), fMode(XrdCl::OpenFlags::None), fInitCondVar(nullptr),
      fReadvIorMax(0), fReadvIovMax(0) {}
   TNetXNGFile(const char *url, const char *lurl, Option_t *mode, const char *title,
               Int_t compress, Int_t netopt, Bool_t parallelopen);
   TNetXNGFile(const char *url, Option_t *mode = "", const char *title = "",
               Int_t compress = 1, Int_t netopt = 0, Bool_t parallelopen = kFALSE);

   virtual ~TNetXNGFile();

   void     Init(Bool_t create) override;
   void     Close(const Option_t *option = "") override;
   void     Seek(Long64_t offset, ERelativeTo position = kBeg) override;
   virtual void     SetAsyncOpenStatus(EAsyncOpenStatus status);
   Long64_t GetSize() const override;
   Int_t    ReOpen(Option_t *modestr) override;
   Bool_t   IsOpen() const override;
   Bool_t   WriteBuffer(const char *buffer, Int_t length) override;
   void     Flush() override;
   Bool_t   ReadBuffer(char *buffer, Int_t length) override;
   Bool_t   ReadBuffer(char *buffer, Long64_t position, Int_t length) override;
   Bool_t   ReadBuffers(char *buffer, Long64_t *position, Int_t *length,
                        Int_t nbuffs) override;
   TString  GetNewUrl() override { return fNewUrl; }

private:
   virtual Bool_t IsUseable() const;
   virtual Bool_t GetVectorReadLimits();
   virtual void   SetEnv();
   Int_t ParseOpenMode(Option_t *in, TString &modestr,
                       XrdCl::OpenFlags::Flags &mode, Bool_t assumeRead);

   TNetXNGFile(const TNetXNGFile &other);             // Not implemented
   TNetXNGFile &operator =(const TNetXNGFile &other); // Not implemented

   ClassDefOverride(TNetXNGFile, 0)   // ROOT class definition
};

#endif // ROOT_TNetXNGFile
