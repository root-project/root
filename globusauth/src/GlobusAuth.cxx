// @(#)root/globus:$Name:  $:$Id: GlobusAuth.cxx,v 1.21 2005/11/17 01:20:45 rdm Exp $
// Author: Gerardo Ganis  15/01/2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* Parts of this file are adapted from the Globus Tool Kit version 2.2.3
 * are subject to related licenses.
 * Please refer to www.globus.org for details
 */

#include "config.h"

#include <errno.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>

#include "TSocket.h"
#include "TAuthenticate.h"
#include "THostAuth.h"
#include "TDatime.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TEnv.h"
#include "Getline.h"
#include "NetErrors.h"

#define HAVE_MEMMOVE 1
extern "C" {
#include <globus_common.h>
#include <globus_gss_assist.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include <sys/ipc.h>
#include <sys/shm.h>
}

static gss_cred_id_t GlbCredHandle = GSS_C_NO_CREDENTIAL;
static gss_cred_id_t GlbDelCredHandle = GSS_C_NO_CREDENTIAL;
static int gShmIdCred = -1;

Int_t GlobusAuthenticate(TAuthenticate *, TString &, TString &);
Int_t GlobusCertFile(Int_t, TString &);
Int_t GlobusCheckSecContext(const char *, gss_ctx_id_t);
Int_t GlobusCheckSecContext(const char *, gss_ctx_id_t);
Int_t GlobusCheckSecCtx(const char *, TRootSecContext *);
Int_t GlobusCleanupContext(gss_ctx_id_t);
void  GlobusCleanupShm();
void  GlobusError(char *, OM_uint32, OM_uint32, Int_t);
Int_t GlobusGetCredHandle(Int_t, Int_t, gss_cred_id_t *);
Int_t GlobusGetDelCred();
Int_t GlobusGetLocalEnv(Int_t *, TString);
Int_t GlobusGetSecContLifeTime(gss_ctx_id_t);
Int_t GlobusNameFromCred(gss_cred_id_t, TString &);
Int_t GlobusNamesFromCert(TString, TString &, TString &);
Int_t GlobusNeedProxy(TString);
void  GlobusSetCertificates(Int_t,Int_t,TString, TString &);


class GlobusAuthInit {
 public:
   GlobusAuthInit() {
      TAuthenticate::SetGlobusAuthHook(&GlobusAuthenticate);
}};
static GlobusAuthInit globusauth_init;

//______________________________________________________________________________
Int_t GlobusAuthenticate(TAuthenticate * Auth, TString & user,
                         TString & details)
{
   // Globus authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //         2 in case of the remote node doesn not seem to support
   //           Globus Authentication
   //         3 in case of the remote node doesn not seem to have
   //           certificates for our CA or is unable to init credentials

   int auth = 0, rc;
   int retval = 0, kind = 0, type = 0, server_auth = 0, brcv = 0, bsnd = 0;
   gss_ctx_id_t GlbContextHandle = GSS_C_NO_CONTEXT;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GssReqFlags = 0;
   int GlbTokenStatus = 0;
   char *host_subj = 0;
   TDatime ExpDate = TDatime();

   // Check if called for cleanup
   if (user == "-1") {
      if (gDebug > 2)
         Info("GlobusAuthenticate", " cleanup call (%s)",details.Data());

      if (details == "context") {
         // Security context cleaning
         GlobusCleanupContext((gss_ctx_id_t)Auth);
      } else if (details == "shm") {
         // Shared memory cleaning (TProofServ)
         GlobusCleanupShm();
      }
      return 1;
   }

   // From the calling TAuthenticate
   TSocket *sock = Auth->GetSocket();
   TString protocol = Auth->GetProtocol();

   if (gDebug > 2)
      Info("GlobusAuthenticate", " enter: protocol:'%s' user:'%s'", protocol.Data(),
           user.Data());

   Int_t ReUse = TAuthenticate::GetAuthReUse();
   Int_t Prompt = TAuthenticate::GetPromptUser();
   TString PromptReUse = TString(Form("pt:%d ru:%d", Prompt, ReUse));
   if (gDebug > 2)
      Info("GlobusAuthenticate", "Prompt: %d, ReUse: %d", Prompt, ReUse);

   // The host FQDN ... for debugging
   const char *hostFQDN = sock->GetInetAddress().GetHostName();

   // Determine local calling environment ...
   Int_t LocalCallEnv = -1;
   if ((rc = GlobusGetLocalEnv(&LocalCallEnv, protocol))) {
      if (gDebug > 0)
          Error("GlobusAuthenticate",
            "unable to set relevant environment variables (rc=%d)",
            rc);
      return -1;
   }
   if (gDebug > 3)
      Info("GlobusAuthenticate", " LocalCallEnv is %d", LocalCallEnv);

   // Set local certificates according to user requests ...
   GlobusSetCertificates(LocalCallEnv,Prompt,PromptReUse,details);

   // Now we send to the rootd/proofd daemons the issuer name
   // of our globus certificates ..
   // We get it the x509 relevant certificate ...
   // The location depends on the calling environment
   TString certfile;
   if ((rc = GlobusCertFile(LocalCallEnv, certfile))) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
               "PROOF Master: unable to determine cert file path (rc=%d)", rc);
      return -1;
   }

   TString isuj, ssuj;
   if ((rc = GlobusNamesFromCert(certfile, isuj, ssuj))) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
               "PROOF Master: unable to determine relevant names(rc=%d)", rc);
      return -1;
   }

   // Find out if we need to init proxies
   Int_t NeedProxy = GlobusNeedProxy(ssuj);

   // Get credential handle ... either genuine or delegated
   if (GlobusGetCredHandle(LocalCallEnv, NeedProxy, &GlbCredHandle)) {
      if (gDebug > 0)
         Error("GlobusAuthenticate", "unable to acquire valid credentials");
      return -1;
   }
   if (gDebug > 3)
      Info("GlobusAuthenticate", " Credential Handle is 0x%x",
           GlbCredHandle);

   // Inquire credentials for Subject name and convert it in human readable form ...
   if ((rc = GlobusNameFromCred(GlbCredHandle, ssuj))) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
               "PROOF Master: unable to determine name from cred (rc=%d)", rc);
      return -1;
   }

   // Create Options string
   Int_t Opt = ReUse * kAUTH_REUSE_MSK +
               Auth->GetRSAKeyType() * kAUTH_RSATY_MSK;
   TString Options(Form("%d %d %s", Opt, ssuj.Length(), ssuj.Data()));

   // Check established authentications
   kind = kROOTD_GLOBUS;
   retval = ReUse;
   if ((rc = Auth->AuthExists(ssuj, TAuthenticate::kGlobus, Options,
             &kind, &retval, &GlobusCheckSecCtx)) == 1) {
      // A valid authentication exists: we are done ...
      return 1;
   }

   if (rc == -2) {
      return rc;
   }
   if (kind == kROOTD_ERR) {
      return 0;
   }
   // If server does not support Globus authentication we can't continue ...
   if (retval == 0 || kind != kROOTD_GLOBUS) {
      if (gDebug > 2)
         Info("GlobusAuthenticate", " got retval: %d kind: %d from server",
              retval, kind);
      return 2;
   }

   // Now we send the issuer to the server daemon
   char buf[20];
   sprintf(buf, "%d", (int) (isuj.Length() + 1));
   if ((bsnd = sock->Send(buf, kMESS_STRING)) != (int) (strlen(buf)+1)) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
            "Length of Issuer name not send correctly: bytes sent: %d (tot len: %d)",
            bsnd - 1, strlen(buf));
      return 0;
   }
   // Now we send it to the server daemon
   if ((bsnd = sock->Send(isuj.Data(), kMESS_STRING)) < (Int_t)(isuj.Length()+1)) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
            "Issuer name not send correctly: bytes sent: %d (tot len: %d)",
            bsnd - 1, isuj.Length());
      return 0;
   }
   // Now we wait for the replay from the server ...
   sock->Recv(retval, kind);
   if (kind == kROOTD_ERR) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
               "recv host subj: host unable init credentials");
      return 3;
   }
   if (kind != kROOTD_GLOBUS) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
            "recv host subj: unexpected message from daemon:"
            " kind: %d (expecting: %d)",kind, kROOTD_GLOBUS);
   } else {
      if (retval == 0) {
         if (gDebug > 0)
            Error("GlobusAuthenticate",
               "recv host subj: host not able to authenticate this CA");
         return 3;
      } else {
         if (gDebug > 3)
            Info("GlobusAuthenticate",
                 "recv host subj: buffer length is: %d", retval);
         host_subj = new char[retval + 1];
         brcv = sock->Recv(host_subj, retval, kind);
         if (gDebug > 3)
            Info("GlobusAuthenticate",
                 "received host_subj: %s: (%d)", host_subj, brcv);
         if (strlen(host_subj) < (UInt_t)(retval - 1) ||
             retval <= 1) {
            if (gDebug > 0) {
               Error("GlobusAuthenticate",
                  "recv host subj: did not receive all the bytes"
                  " (recv: %d, due >%d)", brcv, retval);
               Error("GlobusAuthenticate", "recv host subj: (%d) %s",
                  strlen(host_subj), host_subj);
            }
            if (host_subj) delete[] host_subj;
            return 0;
         }
      }
   }
   // Now we have a valid subject name for the host ...
   if (gDebug > 2)
      Info("GlobusAuthenticate", "Host subject: %s", host_subj);

   // We need to associate a FILE* stream with the socket
   // It will automatically closed when the socket will be closed ...
   int SockFd = sock->GetDescriptor();
   FILE *FILE_SockFd = fdopen(SockFd, "w+");

   // Type of request for credentials depend on calling environment
   GssReqFlags =
       LocalCallEnv >
       0 ? (GSS_C_DELEG_FLAG | GSS_C_MUTUAL_FLAG) : GSS_C_MUTUAL_FLAG;
   if (gDebug > 3)
      Info("GlobusAuthenticate",
           " GssReqFlags: 0x%x, GlbCredentials: 0x%x", GssReqFlags,
           (int) GlbCredHandle);

   // Now we are ready to start negotiating with the Server
   if ((MajStat =
        globus_gss_assist_init_sec_context(&MinStat, GlbCredHandle,
                                           &GlbContextHandle, host_subj,
                                           GssReqFlags, &GssRetFlags,
                                           &GlbTokenStatus,
                                           globus_gss_assist_token_get_fd,
                                           (void *) FILE_SockFd,
                                           globus_gss_assist_token_send_fd,
                                           (void *) FILE_SockFd)) !=
       GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusAuthenticate: gss_assist_init_sec_context",
                  MajStat, MinStat, GlbTokenStatus);
      if (host_subj) delete[] host_subj;
      return 0;
   } else {
      // Set expiration date
      ExpDate.Set(ExpDate.Convert() + GlobusGetSecContLifeTime(GlbContextHandle));
      if (gDebug > 2) {
         Info("GlobusAuthenticate", "authenticated to host %s", hostFQDN);
         Info("GlobusAuthenticate", "expiring on '%s'", ExpDate.AsString());
      }
      if (fflush(FILE_SockFd) != 0) {
         Warning("GlobusAuthenticate",
                 "unable to fflush socket: may cause Auth problems on server side\n");
      }
      auth = 1;
   }

   // Now we have the subject and we can release some resources ...
   if (host_subj) delete[] host_subj;

   // Receive username used for login or key request info and type of key
   int nrec = sock->Recv(retval, type);  // returns user

   Int_t RSAKey = 0;
   if (ReUse == 1) {

      if (type != kROOTD_RSAKEY || retval < 1 || retval > 2)
         Warning("GlobusAuthenticate",
                 "problems recvn RSA key flag: got message %d, flag: %d",
                 type, RSAKey);
      RSAKey = retval - 1;

      // Send the key securely
      TAuthenticate::SendRSAPublicKey(sock,RSAKey);

      // Receive username used for login
      nrec = sock->Recv(retval, type);  // returns user
   }

   if (type != kROOTD_GLOBUS || retval < 1)
      Warning("GlobusAuthenticate",
              "problems recvn (user,offset) length (%d:%d bytes:%d)", type,
              retval, nrec);
   char *rfrm = new char[retval + 1];
   nrec = sock->Recv(rfrm, retval + 1, type);  // returns user
   if (type != kMESS_STRING)
      Warning("GlobusAuthenticate",
              "username and offset not received (%d:%d)", type, nrec);
   else if (gDebug > 2)
      Info("GlobusAuthenticate", "logging remotely as %s ", rfrm);

   // Parse answer
   char *lUser = new char[retval];
   Int_t OffSet = -1;
   sscanf(rfrm, "%s %d", lUser, &OffSet);

   // Return username
   user = lUser;

   // Receive Token
   char *Token = 0;
   if (ReUse == 1 && OffSet > -1) {
      if (TAuthenticate::SecureRecv(sock, 1, RSAKey, &Token) == -1) {
         Warning("SRPAuthenticate",
                 "Problems secure-receiving Token -"
                 " may result in corrupted Token");
      }
      if (gDebug > 3)
         Info("GlobusAuthenticate", "received from server: token: '%s' ",
              Token);
   } else {
      Token = StrDup("");
   }

   // Create SecContext object
   TRootSecContext *ctx =
      Auth->GetHostAuth()->CreateSecContext((const char *)lUser,
          hostFQDN, (Int_t)TAuthenticate::kGlobus, OffSet,
          details, (const char *)Token, ExpDate,
          (void *)GlbContextHandle, RSAKey);
   // Transmit it to TAuthenticate
   Auth->SetSecContext(ctx);

   // receive status from server
   sock->Recv(server_auth, kind);
   if (gDebug > 2)
      Info("GlobusAuthenticate", "received auth status from server: %d ",
           server_auth);

   if (auth && !server_auth)
      Warning("GlobusAuthenticate",
              " it looks like server did not authenticate ");

   // free allocated memory ...
   if (rfrm) delete[] rfrm;
   if (lUser) delete[] lUser;
   if (Token) delete[] Token;

   // return result
   return auth;
}

//______________________________________________________________________________
int GlobusGetDelCred()
{
   // This function fetchs from the shared memory segment created by 'proofd'.
   // the delegated credentials needed to autheticate the slaves ...
   // The shared memory segment is destroyed.

   struct shmid_ds shm_ds;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;

   if (gDebug > 2)
      Info("GlobusGetDelCred:", "Enter ...");

   // Attach segment to address
   gss_buffer_t databuf = (gss_buffer_t) shmat(gShmIdCred, 0, 0);

   // Import credentials
   //    credential= (gss_buffer_t)malloc(sizeof(gss_buffer_desc)+databuf->length);
   gss_buffer_t credential =
       (gss_buffer_t) new char[sizeof(gss_buffer_desc) + databuf->length];
   credential->length = databuf->length;
   credential->value =
       (void *) ((char *) credential + sizeof(size_t) + sizeof(void *));
   void *dbufval =
       (void *) ((char *) databuf + sizeof(size_t) + sizeof(void *));
   memmove(credential->value, dbufval, credential->length);
   if ((MajStat =
        gss_import_cred(&MinStat, &GlbDelCredHandle, 0, 0, credential, 0,
                        0)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusGetDelCred: gss_import_cred", MajStat, MinStat,
                  0);
      return 1;
   } else if (gDebug > 3)
      Info("GlobusGetDelCred:",
           "Globus Credentials successfully imported (0x%x)",
           GlbDelCredHandle);

   if (credential) delete[] credential;

   // Detach from shared memory segment
   int rc = shmdt((const void *) databuf);
   if (rc != 0) {
      if (gDebug > 0)
         Info("GlobusGetDelCred:",
              "unable to detach from shared memory segment (rc=%d)", rc);
   }
   if (gDebug > 3) {
      rc = shmctl(gShmIdCred, IPC_STAT, &shm_ds);
      Info("GlobusGetDelCred:",
           "Process: uid: %d, euid: %d - Buffer: uid: %d, cuid: %d",
           getuid(), geteuid(), shm_ds.shm_perm.uid, shm_ds.shm_perm.cuid);
   }

   rc = shmctl(gShmIdCred, IPC_RMID, &shm_ds);
   if (rc == 0) {
      if (gDebug > 2)
         Info("GlobusGetDelCred:",
              "shared memory segment successfully marked as destroyed");
   } else {
      Warning("GlobusGetDelCred:",
              "unable to mark segment %d as destroyed", gShmIdCred);
   }

   return 0;
}

//______________________________________________________________________________
void GlobusError(char *mess, OM_uint32 majs, OM_uint32 mins, int toks)
{
   // Handle error ...

   char *GlbErr = 0;

   if (!globus_gss_assist_display_status_str
       (&GlbErr, mess, majs, mins, toks)) {
        Error("GlobusError:","%s (majst=%d,minst=%d,tokst:%d)",
            GlbErr, majs, mins, toks);
   } else {
      Error("GlobusError:","%s (not resolved) (majst=%d,minst=%d,tokst:%d)",
            mess, majs, mins, toks);
   }

   if (GlbErr) delete[] GlbErr;
}

//______________________________________________________________________________
int GlobusGetLocalEnv(int *LocalEnv, TString protocol)
{
   // Determines calling environment.
   // Returns 0 if successful; 1 otherwise.

   int retval = 0;

   // Calling application
   TApplication *lApp = gROOT->GetApplication();
   if (gDebug > 2) {
      int i = 0;
      for (; i < lApp->Argc(); i++) {
         Info("GlobusGetLocalEnv", "Application arguments: %d: %s", i,
              lApp->Argv(i));
      }
   }

   *LocalEnv = 0;
   if (lApp != 0) {
      if (gROOT->IsProofServ()) {
         // This is PROOF ... either Master or Slave ...
         if (gDebug > 3) {
            Info("GlobusGetLocalEnv",
                 "PROOF environment, called by the MASTER/SLAVE");
            Info("GlobusGetLocalEnv",
                 "String with pointer to del cred is 0x%x",
                 GlbDelCredHandle);
         }
         *LocalEnv = 2;
         gShmIdCred = -1;
         const char *p = gSystem->Getenv("ROOTSHMIDCRED");
         if (p)
            gShmIdCred = strtol(p, (char **)0, 10);
         if (gShmIdCred <= 0) {
            Info("GlobusGetLocalEnv",
                    " Delegate credentials undefined");
            retval = 1;
         }
      } else {
         if (strstr(protocol.Data(), "proof") != 0) {
            if (gDebug > 3)
               Info("GlobusGetLocalEnv",
                    " PROOF environment, called by the CLIENT");
            *LocalEnv = 1;
         } else if (strstr(protocol.Data(), "root") != 0 ||
                    strstr(protocol.Data(), "sock") != 0) {
            if (gDebug > 3)
               Info("GlobusGetLocalEnv",
                    "ROOT environment (%s)", protocol.Data());
         } else {
            if (gDebug > 0)
               Info("GlobusGetLocalEnv",
                    "unable to recognize the environment"
                    " (protocol: %s)-> assume ROOT",protocol.Data());
         }
      }
   } else {
      if (gDebug > 0)
         Info("GlobusGetLocalEnv",
              "unable to get pointer to current application"
              " -> assume ROOT environment");
   }

   return retval;
}

//______________________________________________________________________________
Int_t GlobusNamesFromCert(TString CertFile, TString &IssuerName, TString &SubjectName)
{
   // Get Issuer and Client Names from CertFile.
   // Returns 0 is successfull, 1 otherwise.

   if (gDebug > 2)
      Info("GlobusNamesFromCert", "Enter: CertFile: %s", CertFile.Data());

   // Test the existence of the certificate file //
   if (gSystem->AccessPathName(CertFile, kReadPermission)) {
      if (gDebug > 0)
         Error("GlobusNamesFromCert", "cannot read requested file %s",
               CertFile.Data());
      return 1;
   }

   // Load the certificate ...
   X509 *xcert = 0;
   FILE *fcert = fopen(CertFile.Data(), "r");
   if (fcert == 0 || !PEM_read_X509(fcert, &xcert, 0, 0)) {
      if (gDebug > 0)
         Error("GlobusNamesFromCert", "Unable to load user certificate ");
      return 2;
   }
   fclose(fcert);

   // Get the issuer name
   IssuerName =
       TString(X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0));
   // Get the subject name
   SubjectName =
       TString(X509_NAME_oneline(X509_get_subject_name(xcert), 0, 0));

   // Notify
   if (gDebug > 2) {
      Info("GlobusNamesFromCert", "Issuer Name: %s", IssuerName.Data());
      Info("GlobusNamesFromCert", "Subject Name: %s", SubjectName.Data());
   }

   // Successful
   return 0;
}

//______________________________________________________________________________
Int_t GlobusNameFromCred(gss_cred_id_t Cred, TString &SubjName)
{
   // Get Subject Name from Credential handle Cred.
   // Returns 0 is successfull, 1 otherwise.

   if (gDebug > 2)
      Info("GlobusNamesFromCred", "Enter: Handle: 0x%p", Cred);

   // Inquire credentials for Subject name and convert it in human readable form ...
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   gss_name_t Name;
   OM_uint32 LifeTime;
   gss_cred_usage_t CredUsage;
   gss_OID_set Mech;
   gss_OID NameType;
   if ((MajStat = gss_inquire_cred(&MinStat, Cred, &Name,
                  &LifeTime, &CredUsage, &Mech)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusNameFromCred: gss_inquire_cred",
                     MajStat, MinStat,0);
      return 1;
   }
   gss_buffer_desc OutBuf;
   if ((MajStat = gss_display_name(&MinStat, Name, &OutBuf,
                  &NameType)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusNameFromCred: gss_display_name",
                     MajStat, MinStat, 0);
      return 2;
   } else
      SubjName = TString((const char *)OutBuf.value);

   // Notify
   if (gDebug > 2)
      Info("GlobusNameFromCred", "Subject Name: %s", SubjName.Data());

   // Successful
   return 0;
}

//______________________________________________________________________________
Int_t GlobusCertFile(int LocalEnv, TString &CertFile)
{
   // Get path of local active certificate.
   // Returns 0 is successfull, 1 otherwise.

   if (gDebug > 2)
      Info("GlobusCertFile", "Enter: LocalEnv: %d", LocalEnv);

   TString usercert_default("/.globus/usercert.pem");

   // Check Standard location first
   if (gSystem->Getenv("X509_USER_CERT") != 0) {
      CertFile = TString(gSystem->Getenv("X509_USER_CERT"));
   } else {
      // If it did not work, action depend on environment
      if (LocalEnv == 2) {
         // We are a Proof master: exit;
         if (gDebug > 0)
            Error("GlobusCertFile",
                  "PROOF Master: host certificate not defined");
         return 1;
      } else {
         // build default location
         CertFile = TString(gSystem->HomeDirectory() + usercert_default;
      }
   }

   if (gDebug > 2)
      Info("GlobusCertFile", "Return: %s", CertFile.Data());

   return 0;
}

//______________________________________________________________________________
Int_t GlobusNeedProxy(TString SubjectName)
{
   // Returns kTRUE if user needs to initialize a proxy

   // check if there is a proxy file associated with this user
   int nProxy = 1;
   char proxy_file[256];
   sprintf(proxy_file, "/tmp/x509up_u%d", getuid());

 again:

   if (gDebug > 3)
      Info("GlobusNeedProxy", "Testing Proxy file: %s", proxy_file);

   if (!gSystem->AccessPathName(proxy_file, kReadPermission)) {
      // Second: load the proxy certificate ...
      X509 *xcert = 0;
      FILE *fcert = fopen(proxy_file, "r");
      if (fcert == 0 || !PEM_read_X509(fcert, &xcert, 0, 0)) {
         if (gDebug > 0)
            Error("GlobusNeedProxy",
                  "Unable to load user proxy certificate ");
         if (fcert) fclose(fcert);
         return 1;
      }
      fclose(fcert);
      // Get proxy names
      TString ProxyIssuerName(X509_NAME_oneline(X509_get_issuer_name(xcert),0,0));
      if (gDebug > 3) {
         Info("GlobusNeedProxy",
              "Proxy Issuer Name: %s", ProxyIssuerName.Data());
      }

      if (ProxyIssuerName.Index(SubjectName) == 0) {
         gSystem->Setenv("X509_USER_PROXY", proxy_file);
         return 0;
      } else {
         sprintf(proxy_file, "/tmp/x509up_u%d.%d", getuid(), nProxy);
         nProxy++;
         goto again;
      }
   } else {
      gSystem->Setenv("X509_USER_PROXY", proxy_file);
      return 1;
   }

}

//______________________________________________________________________________
int GlobusGetCredHandle(Int_t LocalEnv, Int_t NeedProxy, gss_cred_id_t * CredHandle)
{
   // Get Credential Handle, either from scratch, or from delegated info ...
   // Returns 0 is successfull, 1 otherwise.

   int retval = 0;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;

   if (gDebug > 2)
      Info("GlobusGetCredHandle", "Enter: LocalEnv: %d", LocalEnv);

   if (LocalEnv == 2) {
      // If we are a PROOF Master autheticating vs Slaves
      // we only need to fetch the delegated credentials
      // from the shared memory segment the first time we are called ...
      if (GlbDelCredHandle == GSS_C_NO_CREDENTIAL) {
         if (GlobusGetDelCred()) {
            if (gDebug > 0)
               Error("GlobusGetCredHandle",
                  "unable to fetch valid credentials from the shared memory segment");
            retval = 1;
            goto exit;
         }
      }
      *CredHandle = GlbDelCredHandle;
   } else {

      // Inquire Globus credentials:
      // This is looking to file X509_USER_PROXY for valid a X509 cert
      // (default /tmp/x509up_u<uid> )
      if ((NeedProxy == 1) ||
          (MajStat =
           globus_gss_assist_acquire_cred(&MinStat, GSS_C_INITIATE,
                                          CredHandle)) != GSS_S_COMPLETE) {

         // Check if interactive session
         if (isatty(0) && isatty(1)) {

            if (gDebug > 3) {
               GlobusError("GlobusNameFromCred: gss_display_name",
                     MajStat, MinStat, 0);
               Info("GlobusGetCredHandle",
                    "Failed to acquire credentials: trying to initialize proxies ...");
            }

            // Try to get credentials with usual command line ...
            if (!gSystem->Getenv("GLOBUS_LOCATION")) {
               if (gDebug > 0)
                  Error("GlobusGetCredHandle",
                     "Please define a valid GLOBUS_LOCATION");
               retval = 2;
               goto exit;
            }
            // First check if there are special requests for proxy duration ...
            TString InitDur(gEnv->GetValue("Globus.ProxyDuration", "default"));
            if (!InitDur.Contains("default")) {
               InitDur.Insert(0,"-hours ");
               if (gDebug > 2)
                  Info("GlobusAuthenticate", "InitDur: %s (%s)", InitDur.Data(),
                      gEnv->GetValue("Globus.ProxyDuration", "default"));
            } else
               InitDur = TString("");

            // ... and for number of bits in key ...
            TString InitBit(gEnv->GetValue("Globus.ProxyKeyBits", "default"));
            if (!InitBit.Contains("default")) {
               InitBit.Insert(0,"-bits ");
               if (gDebug > 2)
                  Info("GlobusAuthenticate", "InitBit: %s (%s)", InitBit.Data(),
                      gEnv->GetValue("Globus.ProxyKeyBits", "default"));
            } else
               InitBit = TString("");

            // ... and the proxy ...
            TString InitPxy(Form("-out %s", gSystem->Getenv("X509_USER_PROXY")));
            if (gDebug > 3)
               Info("GlobusAutheticate", "InitPxy: %s", InitPxy.Data());

            // ... and environment variables
            TString InitEnv(Form("export X509_CERT_DIR=%s",
               gSystem->Getenv("X509_CERT_DIR")));
            InitEnv += TString(Form("; export X509_USER_CERT=%s",
               gSystem->Getenv("X509_USER_CERT")));
            InitEnv += TString(Form("; export X509_USER_KEY=%s",
               gSystem->Getenv("X509_USER_KEY")));
            if (gDebug > 3)
               Info("GlobusAutheticate", "InitEnv: %s", InitEnv.Data());

            // to execute command to initiate the proxies one needs
            // to source the globus shell environment
            TString ProxyInit("source $GLOBUS_LOCATION/etc/globus-user-env.sh; ");
            ProxyInit += InitEnv;
            ProxyInit += Form("; grid-proxy-init %s %s %s",
                               InitDur.Data(), InitBit.Data(), InitPxy.Data());
            gSystem->Exec(ProxyInit);

            //  retry now
            if ((MajStat =
                 globus_gss_assist_acquire_cred(&MinStat, GSS_C_INITIATE,
                                                CredHandle)) !=
                GSS_S_COMPLETE) {
               if (gDebug > 0)
                  GlobusError("GlobusGetCredHandle: gss_assist_acquire_cred",
                           MajStat, MinStat, 0);
               retval = 3;
               goto exit;
            }
         } else {
            Warning("GlobusGetCredHandle",
                    "not a tty: cannot prompt for credentials, returning failure");
            retval = 3;
            goto exit;
         }
      }
   }

 exit:
   return retval;
}

//______________________________________________________________________________
Int_t GlobusGetSecContLifeTime(gss_ctx_id_t ctx)
{
   // Returns lifetime of established sec context 'ctx'

   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GlbContLifeTime = 0;
   gss_OID   MechType;
   gss_name_t *TargName = 0, *Name = 0;
   int       Dum1, Dum2;

   if (ctx != 0 && ctx != GSS_C_NO_CONTEXT) {
      if ((MajStat = gss_inquire_context(&MinStat, ctx, Name,
                     TargName, &GlbContLifeTime, &MechType, &GssRetFlags,
                     &Dum1, &Dum2)) != GSS_S_COMPLETE) {
         if (gDebug > 0)
            GlobusError("GlobusGetSecContLifeTime: gss_inquire_context",
                          MajStat, MinStat, 0);
         return 0;
      } else {
         if (gDebug > 3)
            Info("GlobusGetSecContLifeTime"," remaining lifetime: %d sec",
                  GlbContLifeTime);
         return (Int_t)GlbContLifeTime;
      }
   }
   return 0;
}

//______________________________________________________________________________
void GlobusSetCertificates(int LocalEnv, int Prompt, TString PromptReUse, TString &Details)
{
   // Defines certificate and key files to use, inquiring the client if needed.

   if (gDebug > 2)
      Info("GlobusSetCertificates", "Enter: LocalEnv: %d", LocalEnv);

   Details = "";

   if (LocalEnv < 2) {

      TString LocDet;
      // Defaults
      if (strlen(TAuthenticate::GetDefaultUser()) > 0) {
         LocDet = TAuthenticate::GetDefaultUser();
      } else {
         LocDet = TString("cd:~/.globus ");
         LocDet+= TString("cf:usercert.pem ");
         LocDet+= TString("kf:userkey.pem ");
         LocDet+= TString("ad:/etc/grid-security/certificates");
      }
      if (gDebug > 3)
         Info("GlobusSetCertificates", " LocDet : %s", LocDet.Data());

      TString PromptString = LocDet;
      PromptString.Insert(0," Local Globus Certificates (");
      PromptString += TString(")\n Enter <key>:<new value> to change: ");

      TString ddir(""), dcer(""), dkey(""), dadi("");
      char *tmp = StrDup(LocDet.Data());
      char *nxt = strtok(tmp," ");
      while (nxt) {
         if (!strncasecmp(nxt, "cd:", 3))
            ddir = TString(nxt + 3);
         if (!strncasecmp(nxt, "cf:", 3))
            dcer = TString(nxt + 3);
         if (!strncasecmp(nxt, "kf:", 3))
            dkey = TString(nxt + 3);
         if (!strncasecmp(nxt, "ad:", 3))
            dadi = TString(nxt + 3);
         // Get next
         nxt = strtok(0," ");
      }
      if (tmp) delete[] tmp;
      // Fill in defaults where needed
      if (ddir == "")
         ddir = TString("~/.globus");
      if (dcer == "")
         dcer = TString("usercert.pem");
      if (dkey == "")
         dkey = TString("userkey.pem");
      if (dadi == "")
         dadi = TString("/etc/grid-security/certificates");

      // Check if needs to prompt the client
      char *det = 0;
      if (Prompt) {
         if (!gROOT->IsProofServ()) {
            det = Getline(PromptString.Data());
         } else {
            Warning("GlobusSetCertificate",
                    "proofserv: cannot prompt for info");
         }
         if (det && det[0])
            det[strlen(det) - 1] = '\0';  // get rid of \n
      }
      if (gDebug > 3)
         if (det)
            Info("GlobusSetCertificates", "got det: %s (%d)", det,
                 strlen(det));

      if (det && strlen(det) > 0) {

         char *tmp = StrDup(det);
         char *nxt = strtok(tmp," ");
         while (nxt) {
            if (!strncasecmp(nxt, "cd:", 3))
               ddir = TString(nxt + 3);
            if (!strncasecmp(nxt, "cf:", 3))
               dcer = TString(nxt + 3);
            if (!strncasecmp(nxt, "kf:", 3))
               dkey = TString(nxt + 3);
            if (!strncasecmp(nxt, "ad:", 3))
               dadi = TString(nxt + 3);
            // Get next
            nxt = strtok(0," ");
         }
         if (tmp) delete[] tmp;
      }
      // Build Details
      Details = PromptReUse + TString(" ") + ddir + TString(" ") +
         dcer + TString(" ") + dkey + TString(" ") + dadi;

      // Perform "~" expansion ...
      // or allow for paths relative to .globus
      const char *globusdef = "/.globus/";
      gSystem->ExpandPathName(ddir);
      if (strncmp(ddir.Data(), "/", 1)) {
         ddir.Insert(0,globusdef);
         ddir.Insert(0,gSystem->HomeDirectory());
      }
      gSystem->ExpandPathName(dcer);
      if (strncmp(dcer.Data(), "/", 1)) {
         dcer.Insert(0,"/");
         dcer.Insert(0,ddir);
      }
      gSystem->ExpandPathName(dkey);
      if (strncmp(dkey.Data(), "/", 1)) {
         dkey.Insert(0,"/");
         dkey.Insert(0,ddir);
      }
      gSystem->ExpandPathName(dadi);
      if (strncmp(dadi.Data(), "/", 1)) {
         dadi.Insert(0,globusdef);
         dadi.Insert(0,gSystem->HomeDirectory());
      }
      if (gDebug > 3)
         Info("GlobusSetCertificates", "after expansion: %s %s %s",
              dcer.Data(), dkey.Data(), dadi.Data());

      // Save them
      gSystem->Setenv("X509_CERT_DIR", dadi);
      gSystem->Setenv("X509_USER_CERT", dcer);
      gSystem->Setenv("X509_USER_KEY", dkey);
   }

   return;
}

//______________________________________________________________________________
Int_t GlobusCleanupContext(gss_ctx_id_t ctx)
{
   // This function cleans up security context ctx

   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;

   // Delete context
   if ((MajStat = gss_delete_sec_context(&MinStat, &ctx,
                  GSS_C_NO_BUFFER)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusCleanupContext: gss_delete_sec_context",
                     MajStat,MinStat, 0);
      return 0;
   }

   return 1;
}

//______________________________________________________________________________
Int_t GlobusCheckSecCtx(const char *subj, TRootSecContext *ctx)
{
   // Globus version of CheckSecCtx to be passed to TAuthenticate::AuthExists
   // Check if Subj matches the one in Ctx
   // Returns: 1 if ok, 0 if not
   // Deactivates Ctx is not valid

   Int_t rc = 0;

   if (ctx->IsActive())
      rc = GlobusCheckSecContext(subj,(gss_ctx_id_t)(ctx->GetContext()));

   return rc;
}

//______________________________________________________________________________
void GlobusCleanupShm()
{
   // This function cleans up shared memories associated with Globus

   if (gROOT->IsProofServ()) {
      struct shmid_ds shm_ds;
      int rc;
      // Delegated Credentials
      gShmIdCred = -1;
      const char *p = gSystem->Getenv("ROOTSHMIDCRED");
      if (p)
         gShmIdCred = strtol(p, (char **)0, 10);
      if (gShmIdCred != -1) {
         if ((rc = shmctl(gShmIdCred, IPC_RMID, &shm_ds)) != 0) {
            if ((rc == EINVAL) || (rc == EIDRM)) {
               if (gDebug > 3)
                  Info("GlobusCleanupShm:",
                       "credentials shared memory segment %s"
                       "already marked as destroyed");
            } else {
               Warning("GlobusCleanupShm:",
                       "unable to mark segment as destroyed (error: 0x%x)",
                       rc);
            }
         } else if (gDebug > 3)
            Info("GlobusCleanupShm:",
                 "shared memory segment %d marked for destruction",
                 gShmIdCred);
      } else if (gDebug > 3) {
         Info("GlobusCleanupShm:",
              "gShmIdCred not defined in this session");
      }
   }
}

//______________________________________________________________________________
Int_t GlobusCheckSecContext(const char *SubjName, gss_ctx_id_t Ctx)
{
   // Checks if SubjName match the one assigned to sec context Ctx
   // Check also validity of Ctx.
   // Returns 1 if everything is ok, 0 if non-matching
   // -1 if Ctx is no more valid and should be discarded

   if (!Ctx)
      return 0;

   int rc = 0;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GlbContLifeTime = 0;

   if (gDebug > 2)
      Info("GlobusCheckSecContext",
           "checking subj:%s", SubjName);

   // Check validity of the retrieved context ...
   Int_t Dum1, Dum2;
   gss_OID MechType;
   gss_name_t *TargName = 0;
   gss_name_t Name;
   if (Ctx != 0 && Ctx != GSS_C_NO_CONTEXT) {

      if ((MajStat = gss_inquire_context(&MinStat, Ctx, &Name,
                     TargName, &GlbContLifeTime, &MechType,
                     &GssRetFlags, &Dum1, &Dum2)) != GSS_S_COMPLETE) {
         if (gDebug > 0)
            GlobusError("GlobusCheckSecContext: gss_inquire_context",
                                                MajStat, MinStat, 0);
         rc = -1;
      } else {
         gss_buffer_desc Name_buffer;
         // Get the subject name now
         if ((MajStat = gss_display_name(&MinStat, Name, &Name_buffer,
                                        GLOBUS_NULL)) != GSS_S_COMPLETE) {
            if (gDebug > 0)
               GlobusError("GlobusCheckSecContext: gss_display_name",
                                                 MajStat, MinStat, 0);
            Name_buffer.length = 0;
            Name_buffer.value = GLOBUS_NULL;
         } else {
            char *theName = new char[Name_buffer.length+1];
            strncpy(theName,(char *)Name_buffer.value,(Int_t)Name_buffer.length);
            theName[Name_buffer.length]= '\0';
            if (gDebug > 2)
               Info("GlobusCheckSecContext","with Subject Name: %s (%d)",
                                             theName,Name_buffer.length);
            if (!strcmp(theName, SubjName)) {
               if (gDebug > 2)
                  Info("GlobusCheckSecContext",
                       "client already authenticated (remaining lifetime: %d sec)",
                        GlbContLifeTime);
               rc = 1;
            }
            // Release allocated space
            if (theName) delete[] theName;
            if ((MajStat = gss_release_name(&MinStat, &Name))
                                          != GSS_S_COMPLETE) {
               if (gDebug > 0)
                   GlobusError("GlobusCheckSecContext: gss_release_name",
                                                      MajStat, MinStat, 0);
            }
         }
      }

   } else {
      rc = -1;
   }

   return rc;
}
