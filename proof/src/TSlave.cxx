// @(#)root/proof:$Name:  $:$Id: TSlave.cxx,v 1.13 2003/08/29 10:41:28 rdm Exp $
// Author: Fons Rademakers   14/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSlave                                                               //
//                                                                      //
// This class describes a PROOF slave server.                           //
// It contains information like the slaves host name, ordinal number,   //
// performance index, socket, etc. Objects of this class can only be    //
// created via TProof member functions.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSlave.h"
#include "TProof.h"
#include "TSocket.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TAuthenticate.h"
#include "TMessage.h"
#include "THostAuth.h"

// For fast name-to-number translation for authentication methods
const char  kMethods[]= "usrpwd srp    krb5   globus ssh    uidgid";
const char *gAuthMeth[kMAXSEC]= {"UsrPwd","SRP","Krb5","Globus","SSH","UidGid"};


ClassImp(TSlave)

//______________________________________________________________________________
TSlave::TSlave(const char *host, Int_t port, Int_t ord, Int_t perf,
               const char *image, Int_t security, TProof *proof)
{
   // Create a PROOF slave object. Called via the TProof ctor.

   fName     = host;
   fPort     = port;
   fImage    = image;
   fWorkDir  = kPROOF_WorkDir;
   fOrdinal  = ord;
   fPerfIdx  = perf;
   fSecurity = -1;
   fProof    = proof;
   fSocket   = 0;
   fInput    = 0;

   // Open connection to remote PROOF slave server.
   fSocket = new TSocket(host, port, 65536);  // make tcpwindosize configurable
   if (fSocket->IsValid()) {

      // Remove socket from global TROOT socket list. Only the TProof object,
      // representing all slave sockets, will be added to this list. This will
      // ensure the correct termination of all proof servers in case the
      // root session terminates.
      gROOT->GetListOfSockets()->Remove(fSocket);

      // Tell remote server to act as master or slave server
      if (proof->IsMaster())
         fSocket->Send("slave");
      else
         fSocket->Send("master");

      // Inquire remote protocol
      fSocket->Send(kROOTD_PROTOCOL);
      Int_t    ProofdProto, what;
      fSocket->Recv(ProofdProto, what);
      if (ProofdProto <= 0 || ProofdProto > 7) ProofdProto = 7;

      TAuthenticate *auth;
      if (security == (Int_t) TAuthenticate::kSRP) {
         auth = new TAuthenticate(fSocket, host, Form("proofs:%d", ProofdProto), "");
      } else if (security == (Int_t) TAuthenticate::kKrb5) {
         auth = new TAuthenticate(fSocket, host, Form("proofk:%d", ProofdProto), "");
      } else {
         auth = new TAuthenticate(fSocket, host, Form("%s:%d", proof->GetUrlProt(),ProofdProto), "");
      }

      // Authenticate to proofd...
      if (!proof->IsMaster()) {

         // we are remote client... need full authentication procedure, either:
         // - via TAuthenticate::SetGlobalUser()/SetGlobalPasswd())
         // - ~/.rootnetrc or ~/.netrc
         // - interactive
         if (!auth->Authenticate()) {
            int sec = auth->GetSecurity();
            if (sec >= 0 && sec <= kMAXSEC) {
               Error("TSlave", "%s authentication failed for host %s", gAuthMeth[sec], host);
            } else {
               Error("TSlave", "authentication failed for host %s (method: %d - unknown)", host, sec);
            }
            delete auth;
            SafeDelete(fSocket);
            return;
         }
         proof->fUser     = auth->GetRemoteLogin(auth->GetHostAuth(),auth->GetSecurity(),auth->GetDetails());
         proof->fPasswd   = auth->GetPasswd();
         proof->fSecurity = auth->GetSecurity();
         fUser            = proof->fUser;
         fSecurity        = proof->fSecurity;

         PDB(kGlobal,3) {
            auth->GetHostAuth()->PrintEstablished();
            Info("TSlave", "Remote Client: fUser is .... %s", proof->fUser.Data());
         }

         if (ProofdProto > 6) {
            // Now we send authentication details to access, eg, data servers
            // not in the proof cluster and to be propagated to slaves.
            // This is triggered by the 'proofdserv <dserv1> <dserv2> ...'
            // card in .rootauthrc
            SendHostAuth(this, 0);
         }

      } else {

         // we are a master server... authenticate either:
         // - stored user/passwd (coming from client)
         // - ~/.rootnetrc or ~/.netrc
         // - but NOT interactive (obviously)

         // check for .rootnetrc
         // if not in .rootnetrc and SRP -> fail
         // if not in .rootnetrc and normal -> set global user/passwd
         // Checks for (user,passwd) are done inside TAuthenticate now (4/2003)

         if (!auth->Authenticate()) {
            int sec = auth->GetSecurity();
            if (sec >= 0 && sec <= kMAXSEC) {
               Error("TSlave", "%s authentication failed for host %s", gAuthMeth[sec], host);
            } else {
               Error("TSlave", "authentication failed for host %s (method: %d - unknown)", host, sec);
            }
            delete auth;
            SafeDelete(fSocket);
            return;
         }
         fUser = auth->GetRemoteLogin(auth->GetHostAuth(),auth->GetSecurity(),auth->GetDetails());

         PDB(kGlobal,3) {
            auth->GetHostAuth()->PrintEstablished();
            Info("TSlave", "Master Server: fUser is .... %s", fUser.Data());
         }

         if (ProofdProto > 6) {
            SendHostAuth(this, 1);
         }
      }

      // fSecurity is the method successfully tried ...
      fSecurity = auth->GetSecurity();

      delete auth;

      char buf[512];

      fSocket->Recv(buf, sizeof(buf));

      if (strcmp(buf, "Okay")) {
         Printf("%s", buf);
         SafeDelete(fSocket);
      } else {
         // get back startup message of proofserv (we are now talking with
         // the real proofserver and not anymore with the proofd front-end)

         Int_t what;
         fSocket->Recv(buf, sizeof(buf), what);

         if (what == kMESS_NOTOK) {
            SafeDelete(fSocket);
            return;
         }

         // exchange protocol level between client and master and between
         // master and slave
         fSocket->Send(kPROOF_Protocol, kROOTD_PROTOCOL);
         fSocket->Recv(fProtocol, what);
         fProof->fProtocol = fProtocol;   // on master this is the protocol
                                          // of the last slave
         TMessage mess;

         // if unsecure, send user name and passwd to remote host (use trivial
         // inverted byte encoding)
         TString passwd = "";
         if (fSecurity == TAuthenticate::kClear || fSecurity == TAuthenticate::kRfio) {
            passwd = fProof->fPasswd;
            for (int i = 0; i < passwd.Length(); i++) {
               char inv = ~passwd(i);
               passwd.Replace(i, 1, inv);
            }
         }

         if (!fProof->IsMaster())
            mess << fProof->fUser << passwd << fProof->fConfFile;
         else
            mess << fProof->fUser << passwd << fOrdinal;

         fSocket->Send(mess);

         // set some socket options
         fSocket->SetOption(kNoDelay, 1);
      }
   } else
      SafeDelete(fSocket);
}

//______________________________________________________________________________
TSlave::~TSlave()
{
   // Destroy slave.

   Close();
}

//______________________________________________________________________________
void TSlave::Close(Option_t *)
{
   // Close slave socket.

   SafeDelete(fInput);
   SafeDelete(fSocket);
}

//______________________________________________________________________________
Int_t TSlave::Compare(const TObject *obj) const
{
   // Used to sort slaves by performance index.

   TSlave *sl = (TSlave *) obj;

   if (fPerfIdx > sl->GetPerfIdx()) return 1;
   if (fPerfIdx < sl->GetPerfIdx()) return -1;
   return 0;
}

//______________________________________________________________________________
void TSlave::Print(Option_t *) const
{
   // Printf info about slave.

   Printf("*** Slave %d  (%s)", fOrdinal, fSocket ? "valid" : "invalid");
   Printf("    Host name:               %s", GetName());
   Printf("    Port number:             %d", GetPort());
   Printf("    User:                    %s", GetUser());
   Printf("    Authentication method:   %d (%s)", GetSecurity(), gAuthMeth[GetSecurity()] );
   Printf("    Slave protocol version:  %d", GetProtocol());
   Printf("    Image name:              %s", GetImage());
   Printf("    Working directory:       %s", GetWorkDir());
   Printf("    Performance index:       %d", GetPerfIdx());
   Printf("    MB's processed:          %.2f", float(GetBytesRead())/(1024*1024));
   Printf("    MB's sent:               %.2f", fSocket ? float(fSocket->GetBytesRecv())/(1024*1024) : 0.0);
   Printf("    MB's received:           %.2f", fSocket ? float(fSocket->GetBytesSent())/(1024*1024) : 0.0);
   Printf("    Real time used (s):      %.3f", GetRealTime());
   Printf("    CPU time used (s):       %.3f", GetCpuTime());
}

//______________________________________________________________________________
void TSlave::SetInputHandler(TFileHandler *ih)
{
   // Adopt and register input handler for this slave. Handler will be deleted
   // by the slave.

   fInput = ih;
   fInput->Add();
}

//______________________________________________________________________________
Int_t TSlave::SendHostAuth(TSlave *sl, Int_t opt)
{
   // Sends the list of the relevant THostAuth objects to the master or
   // to the active slaves.

   int       retval = 0;
   int       bsiz = 0;
   char     *buf  = 0;
   TList    *authInfo = 0;

   // Get pointer to list with authentication info
   authInfo = TAuthenticate::GetAuthInfo();

   if (opt == 0) {
      // We are a client notifying the Master for additional nodes to be
      // considered (typically data servers not in the proof cluster ...).
      // The list is specified in .rootauthrc (or ROOTAUTHRC) under the key
      // 'proofserv'.
      char *net;
      if (gSystem->Getenv("ROOTAUTHRC")) {
         net = (char *)gSystem->Getenv("ROOTAUTHRC");
      } else {
         net = gSystem->ConcatFileName(gSystem->HomeDirectory(), ".rootauthrc");
      }
      PDB(kGlobal,3) Info("SendHostAuth","file: %s",net);

      // Check if file can be read ...
      if (gSystem->AccessPathName(net, kReadPermission)) { return 0; }

      // Open file
      FILE *fd = fopen(net, "r");

      // Scan it ...
      char line[kMAXPATHLEN], host[kMAXPATHLEN], key[kMAXPATHLEN], rest[kMAXPATHLEN];
      char *pnx = 0;
      int  cont = 0;
      while (fgets(line, sizeof(line), fd) != 0) {
         // Skip comment lines
         if (line[0] == '#') continue;
         // Get rid of end of line '\n', if there ...
         if (line[strlen(line)-1] == '\n') line[strlen(line)-1] = '\0';
         if (cont == 0) {
            // scan line
            int nw= sscanf(line, "%s %s", key, rest);
            // no useful info provided for this line
            if (nw < 2) continue;
         }
         // This is the list we are looking for ...
         if (!strcmp(key, "proofserv") || cont == 1) {
            PDB(kGlobal,3) Info("SendHostAuth","found proofserv: %s", rest);

            if (cont == 0) {
               pnx = strstr(line, rest);
            } else if (cont == 1) {
               pnx  = line;
               cont = 0;
            }

            while (pnx != 0 && cont == 0) {
               rest[0] = '\0';
               sscanf(pnx, "%s %s", host, rest);
               PDB(kGlobal,3) Info("SendHostAuth", "found host: %s %s (cont=%d)", host, rest, cont);

               // Check if a protocol is requested
               char *pd1 = 0, *pd2 = 0;
               char  meth[10] = { "" }, usr[256] = { "" };
               int   met = -1;
               pd1 = strchr(host,':');
               if (pd1 != 0) pd2 = strchr(pd1+1, ':');
               if (pd2 != 0) {
                  strcpy(meth, pd2+1);

                  if (strlen(meth) > 1) {
                     // Method passed as string: translate it to number
                     const char *pmet = strstr(kMethods, meth);
                     if (pmet) {
                        met = ((int)(pmet-kMethods))/7;
                     } else {
                        PDB(kGlobal,2) Info("SendHostAuth", "unrecognized method (%s): ", meth);
                        met = -1;
                     }
                  } else {
                     met = atoi(meth);
                  }
                  int plen= (int)(pd2-host);
                  host[plen]='\0';
               }
               if (pd1 != 0) {
                  strcpy(usr, pd1+1);
                  int plen = (int)(pd1-host);
                  host[plen]='\0';
               }
               PDB(kGlobal,3) Info("SendHostAuth", "host user method: %s %s %d", host, usr, met);

               // Get methods from file .rootauthrc
               char **user; Int_t *nmeth,  *security[kMAXSEC]; char **details[kMAXSEC];
               user = new char*[1]; user[0] = StrDup(usr);
               Int_t Nuser = TAuthenticate::GetAuthMeth(host, "root", &user, &nmeth, security, details);
               // Now copy the info to send into buffer
               int ju = 0;
               for (ju = 0; ju < Nuser; ju++) {
                  int i = 0;
                  bsiz = strlen(host)+strlen(user[ju])+2+(nmeth[ju]+1)*3;
                  int jm = -1;
                  for (i = 0; i < nmeth[ju]; i++) {
                     bsiz += strlen(details[i][ju])+1;
                     if (security[i][ju] == met) jm = i;
                  }
                  bsiz += 20;
                  if (jm == -1) {
                     // Details for the method chosen were not found in the file
                     // Get defaults ...
                     char *newdet = TAuthenticate::GetDefaultDetails(met, 0, user[ju]);
                     bsiz += strlen(newdet)+1;
                     buf = new char[bsiz];
                     sprintf(buf,"h:%s u:%s n:%d", host, user[ju], nmeth[ju]+1);
                     sprintf(buf,"%s '%d %s' ", buf, met, newdet);
                     for (i = 0; i < nmeth[ju]; i++) {
                        sprintf(buf,"%s '%d %s' ", buf, security[i][ju], details[i][ju]);
                     }
                     delete [] newdet;
                  } else {
                     // Details for the method chosen were found in the file
                     // Put them first ...
                     buf = new char[bsiz];
                     sprintf(buf,"h:%s u:%s n:%d", host, user[ju], nmeth[ju]);
                     // First the one specified, if any
                     for (i = 0; i < nmeth[ju]; i++) {
                        if (security[i][ju] == met)
                           sprintf(buf,"%s '%d %s' ", buf, security[i][ju], details[i][ju]);
                     }
                     for (i = 0; i < nmeth[ju]; i++) {
                        if (security[i][ju] != met)
                           sprintf(buf,"%s '%d %s' ", buf, security[i][ju], details[i][ju]);
                     }
                  }
                  sl->GetSocket()->Send(buf, kPROOF_SENDHOSTAUTH);
                  delete [] buf;
               }

               // Got to next, if any
               if (strlen(rest) > 0) {
                  // Check if there is a continuation line
                  if ((int)rest[0] == 92) cont = 1;
                  pnx = strstr(pnx, rest);
               } else {
                  pnx = 0;
               }
            }
         }
      }

      // End of transmission ...
      sl->GetSocket()->Send("END", kPROOF_SENDHOSTAUTH);

      fclose(fd);

   } else if (opt == 1) {
      // We are a Master notifying the Slaves for nodes to be considered
      // This includes the other slaves and additional info received from the Client

      PDB(kGlobal,3)
         Info("SendHostAuth","Number of HostAuth instantiations in memory: %d",authInfo->GetSize());

      // Loop over list of auth info
      if (authInfo->GetSize() > 0) {
         TIter next(authInfo);
         THostAuth *fHA;
         while ((fHA = (THostAuth*) next())) {
            PDB(kGlobal,3) fHA->Print();
            // Now copy the info to send into buffer
            int i = 0;
            int nmeth = fHA->NumMethods();
            bsiz = strlen(fHA->GetHost())+strlen(fHA->GetUser())+2+(nmeth+1)*3;
            for (i = 0; i < nmeth; i++) {
               bsiz += strlen(fHA->GetDetails(fHA->GetMethods(i)))+1;
            }
            bsiz += 20;
            buf = new char[bsiz];
            sprintf(buf,"h:%s u:%s n:%d",fHA->GetHost(),fHA->GetUser(),nmeth);
            for (i = 0; i < nmeth; i++) {
               sprintf(buf,"%s '%d %s' ", buf, fHA->GetMethods(i), fHA->GetDetails(fHA->GetMethods(i)));
            }
            sl->GetSocket()->Send(buf, kPROOF_SENDHOSTAUTH);
            delete [] buf;
         }
      }

      // End of transmission ...
      sl->GetSocket()->Send("END", kPROOF_SENDHOSTAUTH);
   }
   return retval;
}
