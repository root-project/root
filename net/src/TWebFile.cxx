// @(#)root/net:$Name:  $:$Id: TWebFile.cxx,v 1.11 2006/05/18 10:11:45 brun Exp $
// Author: Fons Rademakers   17/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWebFile                                                             //
//                                                                      //
// A TWebFile is like a normal TFile except that it reads its data      //
// via a (slightly modified) apache web server. A TWebFile is a         //
// read-only file.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TWebFile.h"
#include "TFilePrefetch.h"
#include "TROOT.h"
#include "TSocket.h"
#include "Bytes.h"


ClassImp(TWebFile)

//______________________________________________________________________________
TWebFile::TWebFile(const char *url) : TFile(url, "WEB")
{
   // Create a Web file object. A web file is the same as a read-only
   // TFile except that it is being read via a HTTP server. The url
   // argument must be of the form: http://host.dom.ain/file.root.
   // If the file specified in the URL does not exist or is not accessible
   // the kZombie bit will be set in the TWebFile object. Use IsZombie()
   // to see if the file is accessible. The preferred interface to this
   // constructor is via TFile::Open().

   TWebFile::Init(kFALSE);
}

//______________________________________________________________________________
TWebFile::TWebFile(TUrl url) : TFile(url.GetUrl(), "WEB")
{
   // Create a Web file object. A web file is the same as a read-only
   // TFile except that it is being read via a HTTP server. Make sure url
   // is a valid TUrl object.
   // If the file specified in the URL does not exist or is not accessible
   // the kZombie bit will be set in the TWebFile object. Use IsZombie()
   // to see if the file is accessible.

   TWebFile::Init(kFALSE);
}

//______________________________________________________________________________
void TWebFile::Init(Bool_t)
{
   // Initialize a TWebFile object.

   char buf[4];

   Seek(0);
   if (ReadBuffer(buf, 4)) {
      Error("TWebFile", "cannot connect to remote host %s", fUrl.GetHost());
      MakeZombie();
      gDirectory = gROOT;
      return;
   }

   if (strncmp(buf, "root", 4)) {
      Error("TWebFile", "remote file does not exists or is not a ROOT file");
      MakeZombie();
      gDirectory = gROOT;
      return;
   }

   TFile::Init(kFALSE);
   fD = -2;   // so TFile::IsOpen() will return true when in TFile::~TFile
}

//______________________________________________________________________________
Bool_t TWebFile::IsOpen() const
{
   // A TWebFile that has been correctly constructed is always considered open.

   return IsZombie() ? kFALSE : kTRUE;
}

//______________________________________________________________________________
Int_t TWebFile::ReOpen(Option_t *mode)
{
   // Reopen a file with a different access mode, like from READ to
   // UPDATE or from NEW, CREATE, RECREATE, UPDATE to READ. Thus the
   // mode argument can be either "READ" or "UPDATE". The method returns
   // 0 in case the mode was successfully modified, 1 in case the mode
   // did not change (was already as requested or wrong input arguments)
   // and -1 in case of failure, in which case the file cannot be used
   // anymore. A TWebFile cannot be reopened in update mode.

   TString opt = mode;
   opt.ToUpper();

   if (opt != "READ" && opt != "UPDATE")
      Error("ReOpen", "mode must be either READ or UPDATE, not %s", opt.Data());

   if (opt == "UPDATE")
      Error("ReOpen", "update mode not allowed for a TWebFile");

   return 1;
}

//______________________________________________________________________________
Bool_t TWebFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range from remote file via HTTP daemon. This
   // routine connects to the remote host, sends the request and returns
   // the buffer. Returns kTRUE in case of error.

   if (fFilePrefetch) {
      if (fFilePrefetch->ReadBuffer(buf,fOffset,len)) return kFALSE;
   }

   TSocket s(fUrl.GetHost(), fUrl.GetPort());
   if (!s.IsValid())
      return kTRUE;

   char msg[256];

   // Give full URL so Apache's virtual hosts solution works.
   // Use protocol 0.9 for efficiency, we are not interested in the 1.0 headers.
#ifdef WIN32
   sprintf(msg, "GET %s://%s:%d/%s?%I64d:%d\r\n", fUrl.GetProtocol(),
           fUrl.GetHost(), fUrl.GetPort(), fUrl.GetFile(), fOffset, len);
#else
   sprintf(msg, "GET %s://%s:%d/%s?%lld:%d\r\n", fUrl.GetProtocol(),
           fUrl.GetHost(), fUrl.GetPort(), fUrl.GetFile(), fOffset, len);
#endif
   s.SendRaw(msg, strlen(msg));
   s.RecvRaw(buf, len);

   fOffset += len;

   fBytesRead  += len;
#ifdef WIN32
   SetFileBytesRead(GetFileBytesRead() + len);
#else
   fgBytesRead += len;
#endif

   return kFALSE;
}

//______________________________________________________________________________
void TWebFile::Seek(Long64_t offset, ERelativeTo pos)
{
   // Set position from where to start reading.

   switch (pos) {
   case kBeg:
      fOffset = offset;
      break;
   case kCur:
      fOffset += offset;
      break;
   case kEnd:
      fOffset = fEND - offset;  // is fEND really EOF or logical EOF?
      break;
   }
}

//______________________________________________________________________________
Long64_t TWebFile::GetSize() const
{
   // Return maximum file size to by-pass truncation checking.

   return kMaxInt;
}
