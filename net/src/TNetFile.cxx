// @(#)root/net:$Name$:$Id$
// Author: Fons Rademakers   14/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNetFile                                                             //
//                                                                      //
// A TNetFile is like a normal TFile except that it reads and writes    //
// its data via a rootd server (for more on the rootd daemon see the    //
// source files ROOTD_*.cxx). TNetFile file names are in standard URL   //
// format with protocol "root". The following are valid TNetFile URL's: //
//                                                                      //
//    root://hpsalo/files/aap.root                                      //
//    root://hpbrun.cern.ch/root/hsimple.root                           //
//    root://pcna49a:5151/~na49/data/run821.root                        //
//    root://pcna49d.cern.ch:5050//v1/data/run810.root                  //
//                                                                      //
// The only difference with the well known httpd URL's is that the root //
// of the remote file tree is the user's home directory. Therefore an   //
// absolute pathname requires a // after the host or port specifier     //
// (see last example). Further the expansion of the standard shell      //
// characters, like ~, $, .., are handled as expected.                  //
// TNetFile (actually TUrl) uses 432 as default port for rootd.         //
//                                                                      //
// Connecting to a rootd requires the remote user id and password.      //
// TNetFile allows three ways for you to provide your login:            //
//   1) Setting it globally via the static functions:                   //
//          TNetFile::SetUser() and TNetFile::SetPasswd()               //
//   2) Getting it from the ~/.netrc file (same file as used by ftp)    //
//   3) Command line prompt                                             //
// The different methods will be tried in the order given above.        //
// On machines with AFS rootd will authenticate using AFS.              //
//                                                                      //
// The rootd daemon lives in the directory $ROOTSYS/bin. It can be      //
// started either via inetd or by hand from the command line (no need   //
// to be super user). For more info about rootd see the web page:       //
// http://root.cern.ch/root/NetFile.html.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef R__LYNXOS
#include <sys/stat.h>
#endif
#include <errno.h>

#include "TNetFile.h"
#include "TROOT.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TSysEvtHandler.h"
#include "Getline.h"
#include "Bytes.h"

// Must match order of ERootdErrors enum define in rootd.h
const char *kRootdErrStr[] = {
   "undefined error",
   "file not found",
   "error in file name",
   "file already exists",
   "no access to file",
   "error opening file",
   "file already opened in read or write mode",
   "file already opened in write mode",
   "no more space on device",
   "bad op code",
   "bad message",
   "error writing to file",
   "error reading from file",
   "no such user",
   "remote not setup for anonymous access",
   "illegal user name",
   "can't cd to home directory",
   "can't get passwd info",
   "wrong passwd",
   "no SRP support in remote daemon",
   "fatal error"
};

char *TNetFile::fgUser;
char *TNetFile::fgPasswd;
SecureAuth_t TNetFile::fgSecAuthHook;


ClassImp(TNetFile)

//______________________________________________________________________________
TNetFile::TNetFile(const char *url, Option_t *option, const char *ftitle, Int_t compress)
         : TFile(url, "NET", ftitle, compress), fUrl(url)
{
   // Create a NetFile object. A net file is the same as a TFile
   // except that it is being accessed via a rootd server. The url
   // argument must be of the form: root://host.dom.ain/file.root.
   // If the file specified in the URL does not exist, is not accessable
   // or can not be created the kZombie bit will be set in the TNetFile
   // object. Use IsZombie() to see if the file is accessable.
   // If the remote daemon thinks the file is still connected, while you are
   // sure this is not the case you can force open the file by preceding the
   // option argument with an "f" or "F" , e.g.: "frecreate". Do this only
   // in cases when you are very sure nobody else is using the file.

   fOffset = 0;

   Bool_t forceOpen = kFALSE;
   if (option[0] == 'F' || option[0] == 'f') {
      fOption   = &option[1];
      forceOpen = kTRUE;
   } else
      fOption = option;

   Bool_t create = kFALSE;
   if (!fOption.CompareTo("NEW", TString::kIgnoreCase) ||
       !fOption.CompareTo("CREATE", TString::kIgnoreCase) ||
       !fOption.CompareTo("RECREATE", TString::kIgnoreCase))
       create = kTRUE;
   Bool_t update = fOption.CompareTo("UPDATE", TString::kIgnoreCase)
                   ? kFALSE : kTRUE;
   Bool_t read   = fOption.CompareTo("READ", TString::kIgnoreCase)
                   ? kFALSE : kTRUE;
   if (!create && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   if (!fUrl.IsValid()) {
      Error("TNetFile", "invalid URL specified: %s", fUrl.GetUrl());
      goto zombie;
   }

   // Open connection to remote rootd server
   fSocket = new TSocket(fUrl.GetHost(), fUrl.GetPort());
   if (!fSocket->IsValid()) {
      Error("TNetFile", "can't open connection to rootd on host %s at port %d",
            fUrl.GetHost(), fUrl.GetPort());
      goto zombie;
   }

   // Set some socket options
   fSocket->SetOption(kNoDelay, 1);
   fSocket->SetOption(kSendBuffer, 65536);
   fSocket->SetOption(kRecvBuffer, 65536);

   // Authenticate to remote rootd server
   if (!Authenticate()) {
      Error("TNetFile", "autentication failed for host %s", fUrl.GetHost());
      goto zombie;
   }

   if (forceOpen)
      fSocket->Send(Form("%s %s", fUrl.GetFile(), ToLower("f"+fOption).Data()), kROOTD_OPEN);
   else
      fSocket->Send(Form("%s %s", fUrl.GetFile(), ToLower(fOption).Data()), kROOTD_OPEN);

   int           stat;
   EMessageTypes kind;

   Recv(stat, kind);

   if (kind == kROOTD_ERR) {
      PrintError("TNetFile", stat);
      goto zombie;
   }

   if (stat == 1)
      fWritable = kTRUE;
   else
      fWritable = kFALSE;

   Init(create);

   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   SafeDelete(fSocket);
   gDirectory = gROOT;
}

//______________________________________________________________________________
TNetFile::~TNetFile()
{
   // TNetFile dtor. Send close message and close socket.

   Close();
   SafeDelete(fSocket);
}

//______________________________________________________________________________
Bool_t TNetFile::Authenticate()
{
   // Authenticate to remote rootd server. Return kTRUE if authentication
   // succeeded.

   Bool_t result = kFALSE;

   char *user   = 0;
   char *passwd = 0;

   // Get user and passwd set via static functions SetUser and SetPasswd.
   if (fgUser)
      user = StrDup(fgUser);
   if (fgPasswd)
      passwd = StrDup(fgPasswd);

   // Check ~/.netrc file if user was not set via the static SetUser() method.
   if (!user)
      CheckNetrc(user, passwd);

   // If user also not set via ~/.netrc ask user.
   if (!user) {
      user = GetUser();
      if (!user)
         Error("Authenticate", "user name not set");
   }

   fUser = user;

   // if not anonymous login try to use secure authentication
   if (strcmp(fUser, "anonymous") && strcmp(fUser, "rootd")) {
      if (!fgSecAuthHook) {
         char *p;
         char *lib = Form("%s/lib/libSRPAuth", gRootDir);
         if ((p = gSystem->DynamicPathName(lib, kTRUE))) {
            delete [] p;
            gSystem->Load(lib);
         }
      }
      if (fgSecAuthHook) {
         Int_t st = (*fgSecAuthHook)(this);
         if (st == 0)
            return kFALSE;
         if (st == 1)
            return kTRUE;
      }
   }

   fSocket->Send(user, kROOTD_USER);

   Int_t         stat;
   EMessageTypes kind;

   Recv(stat, kind);

   if (kind == kROOTD_ERR) {
      PrintError("Authenticate", stat);
      goto out;
   }
   if (kind == kROOTD_AUTH && stat == 1) {
      result = kTRUE;
      goto out;
   }

badpass:
   if (!passwd) {
      passwd = GetPasswd();
      if (!passwd)
         Error("Authenticate", "password not set");
   }

   if (!strcmp(fUser, "anonymous") || !strcmp(fUser, "rootd")) {
      if (!strchr(passwd, '@')) {
         Warning("Authenticate", "please use passwd of form: user@host.do.main");
         delete [] passwd;
         passwd = 0;
         goto badpass;
      }
   }

   if (passwd) {
      int n = strlen(passwd);
      for (int i = 0; i < n; i++)
         passwd[i] = ~passwd[i];
   }

   fSocket->Send(passwd, kROOTD_PASS);

   Recv(stat, kind);
   if (kind == kROOTD_ERR)
      PrintError("Authenticate", stat);
   if (kind == kROOTD_AUTH && stat == 1)
      result = kTRUE;

out:
   delete [] user;
   delete [] passwd;

   return result;
}

//______________________________________________________________________________
Bool_t TNetFile::CheckNetrc(char *&user, char *&passwd)
{
   // Try to get user name and passwd from the ~/.netrc file.
   // This file will only be used when its access mask is 0600.
   // Returns kTRUE if user and passwd were found for the machine
   // specified in the URL. User and passwd must be deleted by
   // the caller. If kFALSE, user and passwd are 0.

#ifdef WIN32
    return kFALSE;
#else
   Bool_t result = kFALSE;
   user = passwd = 0;

   char *net = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".netrc");

   // Only use file when its access rights are 0600
   struct stat buf;
   if (stat(net, &buf) == 0) {
      if (S_ISREG(buf.st_mode) && !S_ISDIR(buf.st_mode) &&
          (buf.st_mode & 0777) == (S_IRUSR | S_IWUSR)) {
         FILE *fd = fopen(net, "r");
         char line[256];
         while (fgets(line, sizeof(line), fd) != 0) {
            if (line[0] == '#') continue;
            char word[6][64];
            int nword = sscanf(line, "%s %s %s %s %s %s", word[0], word[1],
                               word[2], word[3], word[4], word[5]);
            if (nword != 6) continue;
            if (strcmp(word[0], "machine"))  continue;
            if (strcmp(word[2], "login"))    continue;
            if (strcmp(word[4], "password")) continue;

            if (!strcmp(word[1], fUrl.GetHost())) {
               user   = StrDup(word[3]);
               passwd = StrDup(word[5]);
               result = kTRUE;
               break;
            }
         }
         fclose(fd);
      }
   }
   delete [] net;

   return result;
#endif
}

//______________________________________________________________________________
void TNetFile::Close(Option_t *opt)
{
   // Close remote file.

   if (!fSocket) return;

   TFile::Close(opt);
   fSocket->Send(kROOTD_CLOSE);
}

//______________________________________________________________________________
void TNetFile::Flush()
{
   // Flush file to disk.

   if (fSocket && fWritable)
      fSocket->Send(kROOTD_FLUSH);
}

//______________________________________________________________________________
char *TNetFile::GetUser()
{
   // Get user name to be used for authentication to rootd.
   // User is asked to type user name.
   // Returns user name (which must be deleted by caller) or 0.

   char *usr = Getline(Form("Name (%s:%s): ", fUrl.GetHost(),
                                              gSystem->Getenv("USER")));
   if (usr[0]) {
      usr[strlen(usr)-1] = 0;   // get rid of \n
      if (strlen(usr))
         return StrDup(usr);
      else
         return StrDup(gSystem->Getenv("USER"));
   }
   return 0;
}

//______________________________________________________________________________
char *TNetFile::GetPasswd(const char *prompt)
{
   // Get passwd to be used for authentication to rootd.
   // Uses non-echoing command line to get passwd.
   // Returns passwd (which must de deleted by caller) or 0.

   Gl_config("noecho", 1);
   char *pw = Getline((char*)prompt);
   Gl_config("noecho", 0);
   if (pw[0]) {
      pw[strlen(pw)-1] = 0;   // get rid of \n
      return StrDup(pw);
   }
   return 0;
}

//______________________________________________________________________________
void TNetFile::Init(Bool_t create)
{
   // Initialize a TNetFile object.

   Seek(0);

   TFile::Init(create);
   fD = -2;   // so TFile::IsOpen() will return true when in TFile::~TFile
}

//______________________________________________________________________________
Bool_t TNetFile::IsOpen() const
{
   // Retruns kTRUE if file is open, kFALSE otherwise.

   return fSocket == 0 ? kFALSE : kTRUE;
}

//______________________________________________________________________________
void TNetFile::Print(Option_t *)
{
   // Print some info about the net file.

   const char *fname = fUrl.GetFile();
   Printf("URL:           %s", fUrl.GetUrl());
   Printf("Remote file:   %s", &fname[1]);
   Printf("Remote user:   %s", fUser.Data());
   Printf("Title:         %s", fTitle.Data());
   Printf("Option:        %s", fOption.Data());
   Printf("Bytes written: %g", fBytesWrite);
   Printf("Bytes read:    %g", fBytesRead);
}

//______________________________________________________________________________
void TNetFile::PrintError(const char *where, Int_t err)
{
   // Print error string depending on error code.

   Error(where, kRootdErrStr[err]);
}

//______________________________________________________________________________
Bool_t TNetFile::ReadBuffer(char *buf, int len)
{
   // Read specified byte range from remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (!fSocket) return kTRUE;

   Bool_t result = kFALSE;

   if (gApplication && gApplication->GetSignalHandler())
      gApplication->GetSignalHandler()->Delay();

   if (fSocket->Send(Form("%d %d", fOffset, len), kROOTD_GET) < 0) {
      Error("ReadBuffer", "error sending kROOTD_GET command");
      result = kTRUE;
      goto end;
   }

   Int_t         stat, n;
   EMessageTypes kind;

   n = Recv(stat, kind);

   if (kind == kROOTD_ERR || n < 0) {
      PrintError("ReadBuffer", stat);
      result = kTRUE;
      goto end;
   }

   while ((n = fSocket->RecvRaw(buf, len)) < 0 && TSystem::GetErrno() == EINTR)
      TSystem::ResetErrno();

   if (n != len) {
      Error("ReadBuffer", "error receiving buffer of length %d, got %d", len, n);
      result = kTRUE;
      goto end;
   }

   fOffset += len;

   fBytesRead  += len;
#ifdef WIN32
   SetFileBytesRead(GetFileBytesRead() + len);
#else
   fgBytesRead += len;
#endif

end:
   if (gApplication && gApplication->GetSignalHandler())
      gApplication->GetSignalHandler()->HandleDelayedSignal();

   return result;
}

//______________________________________________________________________________
Bool_t TNetFile::WriteBuffer(const char *buf, int len)
{
   // Write specified byte range to remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (!fSocket || !fWritable) return kTRUE;

   Bool_t result = kFALSE;

   gSystem->IgnoreInterrupt();

   if (fSocket->Send(Form("%d %d", fOffset, len), kROOTD_PUT) < 0) {
      Error("WriteBuffer", "error sending kROOTD_PUT command");
      result = kTRUE;
      goto end;
   }
   if (fSocket->SendRaw(buf, len) < 0) {
      Error("WriteBuffer", "error sending buffer");
      result = kTRUE;
      goto end;
   }

   Int_t         stat, n;
   EMessageTypes kind;

   n = Recv(stat, kind);

   if (kind == kROOTD_ERR || n < 0) {
      PrintError("WriteBuffer", stat);
      result = kTRUE;
      goto end;
   }

   fOffset += len;

   fBytesWrite  += len;
#ifdef WIN32
   SetFileBytesWritten(GetFileBytesWritten() + len);
#else
   fgBytesWrite += len;
#endif

end:
   gSystem->IgnoreInterrupt(kFALSE);

   return result;
}

//______________________________________________________________________________
Int_t TNetFile::Recv(Int_t &status, EMessageTypes &kind)
{
   // Return status from rootd server and message kind. Returns -1 in
   // case of error otherwise 8 (sizeof 2 words, status and kind).

   kind   = kROOTD_ERR;
   status = 0;

   if (!fSocket) return -1;

   Int_t hdr[3], n;
   while ((n = fSocket->RecvRaw(hdr, sizeof(hdr))) < 0 && TSystem::GetErrno() == EINTR)
      TSystem::ResetErrno();
   if (n <= 0)
      return -1;

   Int_t len = net2host(hdr[0]);
   if (len != n - (Int_t)sizeof(Int_t))
      return -1;
   kind   = (EMessageTypes) net2host(hdr[1]);
   status = net2host(hdr[2]);

   return n - sizeof(Int_t);
}

//______________________________________________________________________________
void TNetFile::Seek(Seek_t offset, ERelativeTo pos)
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
void TNetFile::SetUser(const char *user)
{
   // Set user name to be used for authentication to rootd.

   if (fgUser)
      delete [] fgUser;

   if (!user || !user[0])
      fgUser = 0;
   else
      fgUser = StrDup(user);
}

//______________________________________________________________________________
void TNetFile::SetPasswd(const char *passwd)
{
   // Set passwd to be used for authentication to rootd.

   if (fgPasswd)
      delete [] fgPasswd;

   if (!passwd || !passwd[0])
      fgPasswd = 0;
   else
      fgPasswd = StrDup(passwd);
}

//______________________________________________________________________________
void TNetFile::SetSecureAuthHook(SecureAuth_t func)
{
   // Set secure authorization function. Automatically called when libSRPAuth
   // is loaded.

   fgSecAuthHook = func;
}
