// @(#)root/net:$Name:  $:$Id: TWebFile.cxx,v 1.17 2006/06/09 01:21:43 rdm Exp $
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

   Init(kFALSE);
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

   Init(kFALSE);
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

   if (strncmp(buf, "root", 4) && strncmp(buf, "PK", 2)) {  // PK is zip file
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

   Int_t st;
   if ((st = ReadBufferViaCache(buf, len))) {
      if (st == 2)
         return kTRUE;
      return kFALSE;
   }

   if (fFilePrefetch) {
      if (!fFilePrefetch->ReadBuffer(buf, fOffset, len))
         return kFALSE;
   }

   // Give full URL so Apache's virtual hosts solution works.
   // Use protocol 0.9 for efficiency, we are not interested in the 1.0 headers.
   TString msg = "GET ";
   msg += fUrl.GetProtocol();
   msg += "://";
   msg += fUrl.GetHost();
   msg += ":";
   msg += fUrl.GetPort();
   msg += "/";
   msg += fUrl.GetFile();
   msg += "?";
   msg += fOffset;
   msg += ":";
   msg += len;
   msg += "\r\n";

   if (GetFromWeb(buf, len, msg) == -1)
      return kTRUE;

   fOffset += len;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TWebFile::ReadBuffers(char *buf,  Long64_t *pos, Int_t *len, Int_t nbuf)
{
   // Read the nbuf blocks described in arrays pos and len,
   // where pos[i] is the seek position of block i of length len[i].
   // Note that for nbuf=1, this call is equivalent to TFile::ReafBuffer
   // This function is overloaded by TNetFile, TWebFile, etc.
   // Returns kTRUE in case of failure.

   // Give full URL so Apache's virtual hosts solution works.
   // Use protocol 0.9 for efficiency, we are not interested in the 1.0 headers.
   TString msgh = "GET ";
   msgh += fUrl.GetProtocol();
   msgh += "://";
   msgh += fUrl.GetHost();
   msgh += ":";
   msgh += fUrl.GetPort();
   msgh += "/";
   msgh += fUrl.GetFile();
   msgh += "?";

   TString msg = msgh;

   Int_t k = 0, n = 0;
   for (Int_t i = 0; i < nbuf; i++) {
      if (n) msg += ",";
      msg += pos[i] + fArchiveOffset;
      msg += ":";
      msg += len[i];
      n   += len[i];
      if (msg.Length() > 8000) {
         msg += "\r\n";
         if (GetFromWeb(&buf[k], n, msg) == -1)
            return kTRUE;
         msg = msgh;
         k += n;
         n = 0;
      }
   }

   msg += "\r\n";

   if (GetFromWeb(&buf[k], n, msg) == -1)
      return kTRUE;

   fOffset = pos[nbuf-1] + len[nbuf-1] + fArchiveOffset;

   return kFALSE;
}

//______________________________________________________________________________
Int_t TWebFile::GetFromWeb(char *buf, Int_t len, const TString &msg)
{
   // Read request from web server. Returns -1 in case of error,
   // 0 in case of success.

   if (!len) return 0;

   TSocket s(fUrl.GetHost(), fUrl.GetPort());
   if (!s.IsValid())
      return -1;

   if (s.SendRaw(msg.Data(), msg.Length()) == -1)
      return -1;

   if (s.RecvRaw(buf, len) == -1)
      return -1;

   fBytesRead += len;
   SetFileBytesRead(GetFileBytesRead() + len);

   return 0;
}

//______________________________________________________________________________
void TWebFile::Seek(Long64_t offset, ERelativeTo pos)
{
   // Set position from where to start reading.

   switch (pos) {
   case kBeg:
      fOffset = offset + fArchiveOffset;
      break;
   case kCur:
      fOffset += offset;
      break;
   case kEnd:
      // this option is not used currently in the ROOT code
      if (fArchiveOffset)
         Error("Seek", "seeking from end in archive is not (yet) supported");
      fOffset = fEND - offset;  // is fEND really EOF or logical EOF?
      break;
   }
}

//______________________________________________________________________________
Long64_t TWebFile::GetSize() const
{
   // Return maximum file size.

   Long64_t size;
   char     asize[64];

   TString msg = "GET ";
   msg += fUrl.GetProtocol();
   msg += "://";
   msg += fUrl.GetHost();
   msg += ":";
   msg += fUrl.GetPort();
   msg += "/";
   msg += fUrl.GetFile();
   msg += "?";
   msg += -1;
   msg += "\r\n";

   if (const_cast<TWebFile*>(this)->GetFromWeb(asize, 64, msg) == -1)
      return kMaxInt;

#ifndef R__WIN32
   size = atoll(asize);
#else
   size = _atoi64(asize);
#endif

   return size;
}
