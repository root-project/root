// @(#)root/net:$Name:  $:$Id: TWebFile.cxx,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
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
#include "TROOT.h"
#include "TSocket.h"
#include "Bytes.h"


ClassImp(TWebFile)

//______________________________________________________________________________
TWebFile::TWebFile(const char *url) : TFile(url, "WEB"), fUrl(url)
{
   // Create a Web file object. A web file is the same as a read-only
   // TFile except that it is being read via a HTTP server. The url
   // argument must be of the form: http://host.dom.ain/file.root.
   // If the file specified in the URL does not exist or is not accessible
   // the kZombie bit will be set in the TWebFile object. Use IsZombie()
   // to see if the file is accessible. The preferred interface to this
   // constructor is via TFile::Open().

   fOffset = 0;

   Init(kFALSE);
}

//______________________________________________________________________________
TWebFile::TWebFile(TUrl url) : TFile(url.GetUrl(), "WEB"), fUrl(url)
{
   // Create a Web file object. A web file is the same as a read-only
   // TFile except that it is being read via a HTTP server. Make sure url
   // is a valid TUrl object.
   // If the file specified in the URL does not exist or is not accessible
   // the kZombie bit will be set in the TWebFile object. Use IsZombie()
   // to see if the file is accessible.

   fOffset = 0;

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
Bool_t TWebFile::ReadBuffer(char *buf, int len)
{
   // Read specified byte range from remote file via HTTP daemon. This
   // routine connects to the remote host, sends the request and returns
   // the buffer. Returns kTRUE in case of error.

   TSocket s(fUrl.GetHost(), fUrl.GetPort());
   if (!s.IsValid())
      return kTRUE;

   char msg[256];

   // Give full URL so Apache's virtual hosts solution works.
   // Use protocol 0.9 for efficiency, we are not interested in the 1.0 headers.
   sprintf(msg, "GET %s://%s:%d%s?%d:%d\r\n", fUrl.GetProtocol(),
           fUrl.GetHost(), fUrl.GetPort(), fUrl.GetFile(), fOffset, len);
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
void TWebFile::Seek(Seek_t offset, ERelativeTo pos)
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
