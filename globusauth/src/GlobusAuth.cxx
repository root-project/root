// @(#)root/globus:$Name:  $:$Id: GlobusAuth.cxx,v 1.3 2003/10/07 14:03:02 rdm Exp $
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

#include <errno.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>

extern "C" {
#include <globus_gss_assist.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include <sys/ipc.h>
#include <sys/shm.h>
}
#include "TSocket.h"
#include "TAuthenticate.h"
#include "THostAuth.h"
#include "TError.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TEnv.h"
#include "Getline.h"
#include "rpderr.h"
static gss_cred_id_t GlbDelCredHandle = GSS_C_NO_CREDENTIAL;
int gShmIdCred = -1;
char gConfDir[kMAXPATHLEN] = { 0 };
static int gLocalCallEnv = -1;
char gUser[256] = { 0 };

int GlobusGetDelCred();
void GlobusCleanup();
void GlobusError(char *, OM_uint32, OM_uint32, int);
int GlobusStoreSecContext(char *, gss_ctx_id_t, char *);
int GlobusGetLocalEnv(int *, TString);
int GlobusGetNames(int, char **, char **);
int GlobusGetCredHandle(int, gss_cred_id_t *);
int GlobusCheckSecContext(char *, char *);
int GlobusUpdateSecContInfo(int);
void GlobusSetCertificates(int);

Int_t GlobusAuthenticate(TAuthenticate *, TString &, TString &);

class GlobusAuthInit {
 public:
   GlobusAuthInit() {
      TAuthenticate::SetGlobusAuthHook(&GlobusAuthenticate);
}};
static GlobusAuthInit globusauth_init;

// For established Security Contexts
static char **hostGlbSecCont = 0;
static char **subjGlbSecCont = 0;
static gss_ctx_id_t *sptrGlbSecCont = 0;
static int NumGlbSecCont = 0;

// OffSet in Auth Tab remote file
TString gDetails;
static int gNeedProxy = 1;
char gPromptReUse[20];
Int_t gPrompt = 0;
Int_t gRSAKey = 0;

TSocket *sock = 0;
THostAuth *HostAuth = 0;
TString protocol;

//______________________________________________________________________________
Int_t GlobusAuthenticate(TAuthenticate * Auth, TString & user,
                         TString & details)
{
   // Globus authentication code.
   // Returns 0 in case authentication failed
   //         1 in case of success
   //         2 in case of the remote node doesn not seem to support Globus Authentication
   //         3 in case of the remote node doesn not seem to have certificates for our CA

   int auth = 0, rc;
   int retval = 0, kind = 0, type = 0, server_auth = 0, brcv = 0, bsnd = 0;
   gss_cred_id_t GlbCredHandle = GSS_C_NO_CREDENTIAL;
   gss_ctx_id_t GlbContextHandle = GSS_C_NO_CONTEXT;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GssReqFlags = 0;
   int GlbTokenStatus = 0;
   char *isuj = 0;
   char *ssuj = 0;
   char *host_subj = 0;
   gss_buffer_desc OutBuf;

   // From the calling TAuthenticate
   sock = Auth->GetSocket();
   HostAuth = Auth->GetHostAuth();
   protocol = Auth->GetProtocol();

   if (gDebug > 2)
      Info("GlobusAuthenticate", " enter: %s %s", protocol.Data(),
           user.Data());

   // If we are called for local cleanup, do it and return ...
   if (protocol == "cleanup") {
      GlobusCleanup();
      return 1;
   }
   Int_t ReUse = TAuthenticate::GetAuthReUse();
   gPrompt     = TAuthenticate::GetPromptUser();
   sprintf(gPromptReUse, "pt:%d ru:%d", gPrompt, ReUse);

   // The host FQDN ... for debugging
   const char *hostFQDN = sock->GetInetAddress().GetHostName();

   // Determine local calling environment ...
   if ((rc = GlobusGetLocalEnv(&gLocalCallEnv, protocol))) {
      Error("GlobusAuthenticate",
            "PROOF Master: unable to set relevant environment variables (rc=%d)",
            rc);
      return -1;
   }
   if (gDebug > 3)
      Info("GlobusAuthenticate", " gLocalCallEnv is %d", gLocalCallEnv);

   // Set local certificates according to user requests ...
   GlobusSetCertificates(gLocalCallEnv);

   // Now we send to the rootd/proofd daemons the issuer name of our globus certificates ..
   // We get it the x509 relevant certificate ... the location depends on the calling environment
   char *stmp;
   if ((rc = GlobusGetNames(gLocalCallEnv, &isuj, &stmp))) {
      Error("GlobusAuthenticate",
            "PROOF Master: unable to determine relevant names(rc=%d)", rc);
      return -1;
   }
   if (gDebug > 2)
      Info("GlobusAuthenticate", " Issuer name is %s (%d)", isuj,
           strlen(isuj));
   if (stmp) delete[] stmp;

   // Get credential handle ... either genuine or delegated
   if (GlobusGetCredHandle(gLocalCallEnv, &GlbCredHandle)) {
      Error("GlobusAuthenticate", "unable to acquire valid credentials");
      return -1;
   }
   if (gDebug > 3)
      Info("GlobusAuthenticate", " Credential Handle is 0x%x",
           GlbCredHandle);

   // Inquire credentials for Subject name and convert it in human readable form ...
   gss_name_t Name;
   OM_uint32 LifeTime;
   gss_cred_usage_t CredUsage;
   gss_OID_set Mech;
   gss_OID NameType;
   if ((MajStat =
        gss_inquire_cred(&MinStat, GlbCredHandle, &Name, &LifeTime,
                         &CredUsage, &Mech)) != GSS_S_COMPLETE) {
      GlobusError("GlobusAuthenticate: gss_inquire_cred", MajStat, MinStat,
                  0);
      return -1;
   }
   if ((MajStat =
        gss_display_name(&MinStat, Name, &OutBuf,
                         &NameType)) != GSS_S_COMPLETE) {
      GlobusError("GlobusAuthenticate: gss_inquire_cred", MajStat, MinStat,
                  0);
      return -1;
   } else {
      ssuj = StrDup((char *) OutBuf.value);
   }
   if (gDebug > 2)
      Info("GlobusAuthenticate", " Subject name is %s (%d)", ssuj,
           strlen(ssuj));

   // Create Options string
   char *Options = new char[strlen(ssuj) + 20];
   int Opt = ReUse * kAUTH_REUSE_MSK;
   if (GlobusCheckSecContext((char *) hostFQDN, ssuj) > -1) {
      sprintf(Options, "%d %d %s", Opt, strlen(ssuj), ssuj);
   } else {
      sprintf(Options, "%d 4 None", Opt);
   }

   // Check established authentications
   kind = kROOTD_GLOBUS;
   retval = ReUse;
   rc = 0;
   if ((rc =
        TAuthenticate::AuthExists(Auth, (Int_t) TAuthenticate::kGlobus,
                                  gDetails, Options, &kind,
                                  &retval)) == 1) {
      // A valid authentication exists: we are done ...
      if (Options) delete[] Options;
      return 1;
   }
   if (rc == -2) {
      if (Options) delete[] Options;
      return rc;
   }
   // If server does not support Globus authentication we can't continue ...
   if (retval == 0 || kind != kROOTD_GLOBUS) {
      if (gDebug > 2)
         Info("GlobusAuthenticate", " got retval: %d kind: %d from server",
              retval, kind);
      return 2;
   }
   // Now we send the issuer to the server daemon
   char *buf = new char[20];
   sprintf(buf, "%d", (int) (strlen(isuj) + 1));
   if ((bsnd = sock->Send(buf, kMESS_STRING)) != (int) (strlen(buf) + 1)) {
      Error("GlobusAuthenticate",
            "Length of Issuer name not send correctly: bytes sent: %d (tot len: %d)",
            bsnd - 1, strlen(buf));
      return 0;
   }
   if (buf) delete[] buf;
   // Now we send it to the server daemon
   if ((bsnd = sock->Send(isuj, kMESS_STRING)) != (int) (strlen(isuj) + 1)) {
      Error("GlobusAuthenticate",
            "Issuer name not send correctly: bytes sent: %d (tot len: %d)",
            bsnd - 1, strlen(isuj));
      return 0;
   }
   // Now we wait for the replay from the server ...
   sock->Recv(retval, kind);
   if (kind != kROOTD_GLOBUS) {
      Error("GlobusAuthenticate",
            "recv host subj: unexpected message from daemon: kind: %d (expecting: %d)",
            kind, kROOTD_GLOBUS);
   } else {
      if (retval == 0) {
         Error("GlobusAuthenticate",
               "recv host subj: host not able to authenticate this CA");
         return 3;
      } else {
         if (gDebug > 3)
            Info("GlobusAuthenticate",
                 "recv host subj: buffer length is: %d", retval);
         host_subj = new char[retval + 1];
         brcv = sock->Recv(host_subj, retval, kind);
         if (brcv < (retval - 1)) {
            Error("GlobusAuthenticate",
                  "recv host subj: did not receive all the bytes (recv: %d, due >%d)",
                  brcv, retval);
            Error("GlobusAuthenticate", "recv host subj: (%d) %s",
                  strlen(host_subj), host_subj);
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
       gLocalCallEnv >
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
      GlobusError("GlobusAuthenticate: gss_assist_init_sec_context",
                  MajStat, MinStat, GlbTokenStatus);
      if (host_subj) delete[] host_subj;
      return 0;
   } else {
      GlobusStoreSecContext((char *) hostFQDN, GlbContextHandle, ssuj);
      if (gDebug > 2)
         Info("GlobusAuthenticate", "authenticated to host %s", hostFQDN);
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

   if (ReUse == 1) {

      if (type != kROOTD_RSAKEY)
         Warning("GlobusAuthenticate",
                 "problems recvn RSA key flag: got message %d, flag: %d",
                 type, gRSAKey);
      gRSAKey = 1;

      // Send the key securely
      TAuthenticate::SendRSAPublicKey(sock);

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
   // Keep track of remote login username ...
   strcpy(gUser, lUser);

   // Receive Token
   char *Token = 0;
   if (ReUse == 1 && OffSet > -1) {
      if (TAuthenticate::SecureRecv(sock, gRSAKey, &Token) == -1) {
         Warning("SRPAuthenticate",
                 "Problems secure-receiving Token - may result in corrupted Token");
      }
      if (gDebug > 3)
         Info("GlobusAuthenticate", "received from server: token: '%s' ",
              Token);
   } else {
      Token = StrDup("");
   }

   // Create and save AuthDetails object
   TAuthenticate::SaveAuthDetails(Auth, (Int_t) TAuthenticate::kGlobus,
                                  OffSet, ReUse, gDetails, lUser, gRSAKey,
                                  Token);
   details = gDetails;

   // receive status from server
   sock->Recv(server_auth, kind);
   if (gDebug > 2)
      Info("GlobusAuthenticate", "received auth status from server: %d ",
           server_auth);

   if (auth && !server_auth)
      Warning("GlobusAuthenticate",
              " it looks like server did not authenticate ");

   // free allocated memory ...
   if (isuj) delete[] isuj;
   if (ssuj) delete[] ssuj;
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

   char *GlbErr;

   if (!globus_gss_assist_display_status_str
       (&GlbErr, mess, majs, mins, toks)) {
   } else {
      GlbErr = new char[kMAXPATHLEN];
      sprintf(GlbErr, "%s: error messaged not resolved ", mess);
   }
   Error(":Error: %s (majst=%d,minst=%d,tokst:%d)", GlbErr, majs, mins,
         toks);

   if (GlbErr) delete[] GlbErr;
}

//______________________________________________________________________________
int GlobusStoreSecContext(char *host, gss_ctx_id_t context_handle,
                          char *client_name)
{
   // Store relevant info about an established security context for later use.
   // On success returns number of stored security contexts; 0 otherwise.

   if (gDebug > 2)
      Info("GlobusStoreSecContext", "Enter: %s", host);

   // Now we can count it
   NumGlbSecCont++;

   if (NumGlbSecCont > 1) {
      char **tmph = hostGlbSecCont;
      char **tmpc = subjGlbSecCont;
      gss_ctx_id_t *tmps = sptrGlbSecCont;

      hostGlbSecCont = new char *[NumGlbSecCont];
      subjGlbSecCont = new char *[NumGlbSecCont];
      sptrGlbSecCont = new gss_ctx_id_t[NumGlbSecCont];

      int i;
      for (i = 0; i < NumGlbSecCont - 1; i++) {
         hostGlbSecCont[i] = strdup(tmph[i]);
         subjGlbSecCont[i] = strdup(tmpc[i]);
         sptrGlbSecCont[i] = tmps[i];

      }

      hostGlbSecCont[NumGlbSecCont - 1] = strdup(host);
      subjGlbSecCont[NumGlbSecCont - 1] = strdup(client_name);
      sptrGlbSecCont[NumGlbSecCont - 1] = context_handle;

      if (tmph) delete[] tmph;
      if (tmpc) delete[] tmpc;
      if (tmps) delete[] tmps;

   } else {

      hostGlbSecCont = new char *[NumGlbSecCont];
      subjGlbSecCont = new char *[NumGlbSecCont];
      sptrGlbSecCont = new gss_ctx_id_t[NumGlbSecCont];
      hostGlbSecCont[NumGlbSecCont - 1] = strdup(host);
      subjGlbSecCont[NumGlbSecCont - 1] = strdup(client_name);
      sptrGlbSecCont[NumGlbSecCont - 1] = context_handle;
   }

   if (gDebug > 2) {
      int isave = NumGlbSecCont - 1;
      Info("GlobusStoreSecContext",
           "stored new sec context (session: %d, no:%d, ctx_id_t:0x%x) for client %s for host %s",
           getpid(), NumGlbSecCont, sptrGlbSecCont[isave],
           subjGlbSecCont[isave], hostGlbSecCont[isave]);
   }

   return NumGlbSecCont;
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
         Info("GlobusGetLocalEnv", " Application arguments: %d: %s", i,
              lApp->Argv()[i]);
      }
   }

   *LocalEnv = 0;
   if (lApp != 0) {
      if (lApp->Argc() > 10 && gROOT->IsProofServ()) {
         // This is PROOF ... either Master or Slave ...
         if (gDebug > 3) {
            Info("GlobusGetLocalEnv",
                 " PROOF environment, called by the MASTER/SLAVE");
            Info("GlobusGetLocalEnv",
                 " String with Pointer to del cred is 0x%x",
                 GlbDelCredHandle);
         }
         *LocalEnv = 2;
         strncpy(gConfDir, lApp->Argv()[2], strlen(lApp->Argv()[2]) + 1);
         gShmIdCred = atoi(lApp->Argv()[7]);
         if (setenv("X509_CERT_DIR", lApp->Argv()[8], 1)) {
            Error("GlobusGetLocalEnv",
                  "PROOF Master: unable to set X509_CERT_DIR ");
            retval = 1;
         }
         if (setenv("X509_USER_CERT", lApp->Argv()[9], 1)) {
            Error("GlobusGetLocalEnv",
                  "PROOF Master: unable to set X509_USER_CERT ");
            retval = 2;
         }
         if (setenv("X509_USER_KEY", lApp->Argv()[10], 1)) {
            Error("GlobusGetLocalEnv",
                  "PROOF Master: unable to set X509_USER_KEY ");
            retval = 3;
         }
      } else {
         if (strstr(protocol.Data(), "proof") != 0) {
            if (gDebug > 3)
               Info("GlobusGetLocalEnv",
                    " PROOF environment, called by the CLIENT");
            *LocalEnv = 1;
         } else if (strstr(protocol.Data(), "root") != 0) {
            if (gDebug > 3)
               Info("GlobusGetLocalEnv", " ROOT environment");
         } else {
            Warning("GlobusGetLocalEnv",
                    " Unable to recognize the environment (protocol: %s)-> assume ROOT",
                    protocol.Data());
         }
      }
   } else {
      Warning("GlobusGetLocalEnv",
              " Unable to get pointer to current application -> assume ROOT environment");
   }

   return retval;
}

//______________________________________________________________________________
int GlobusGetNames(int LocalEnv, char **IssuerName, char **SubjectName)
{
   // Get Issuer and Client Names from local certificates.
   // Returns 0 is successfull, 1 otherwise.

   char *usercert_default = "/.globus/usercert.pem";
   char *cert_file = 0;
   X509 *xcert = 0;

   if (gDebug > 2)
      Info("GlobusGetNames", "Enter: LocalEnv: %d", LocalEnv);

   int retval = 0;

   if (LocalEnv == 2) {
      // We are a Proof master: the location is given by X509_USER_CERT ... if not exit;
      if (getenv("X509_USER_CERT") != 0) {
         int lcf = strlen(getenv("X509_USER_CERT"));
         cert_file = new char[lcf];
         strncpy(cert_file, getenv("X509_USER_CERT"), lcf + 1);
      } else {
         Error("GlobusGetNames",
               "PROOF Master: host certificate not defined");
         retval = 1;
      }
   } else {
      // We are a client: determine the location for user certificate ...
      if (getenv("X509_USER_CERT") != 0) {
         int lcf = strlen(getenv("X509_USER_CERT"));
         cert_file = new char[lcf];
         strncpy(cert_file, getenv("X509_USER_CERT"), lcf + 1);
      } else {
         char *userhome = getenv("HOME");
         cert_file =
             new char[strlen(userhome) + strlen(usercert_default) + 1];
         strncpy(cert_file, userhome, strlen(userhome) + 1);
         strncpy(cert_file + strlen(cert_file), usercert_default,
                 strlen(usercert_default) + 1);
      }
   }

   // Test the existence of the certificate file //
   if (access(cert_file, F_OK)) {
      Error("GlobusGetNames", "requested file %s does not exist",
            cert_file);
      //     retval= 2;
      if (cert_file) delete[] cert_file;
      return 2;
   } else if (access(cert_file, R_OK)) {
      Error("GlobusGetNames", "no permission to read requested file %s",
            cert_file);
      //     retval= 4;
      if (cert_file) delete[] cert_file;
      return 4;
   } else if (gDebug > 3) {
      Info("GlobusGetNames", "File with certificate: %s", cert_file);
   }
   // Second: load the certificate ...
   FILE *fcert = fopen(cert_file, "r");
   if (fcert == 0 || !PEM_read_X509(fcert, &xcert, 0, 0)) {
      Error("GlobusGetNames", "Unable to load user certificate ");
      if (cert_file) delete[] cert_file;
      return 5;
   }
   fclose(fcert);

   // Get the issuer name
   *IssuerName =
       StrDup(X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0));
   // Get the subject name
   *SubjectName =
       StrDup(X509_NAME_oneline(X509_get_subject_name(xcert), 0, 0));

   if (gDebug > 2) {
      Info("GlobusGetNames", "Issuer Name: %s", *IssuerName);
      Info("GlobusGetNames", "Subject Name: %s", *SubjectName);
   }

   if (cert_file) delete[] cert_file;

   // Now check if there is a proxy file associated with this user
   int nProxy = 1;
   gNeedProxy = 1;
   char proxy_file[256];
   sprintf(proxy_file, "/tmp/x509up_u%d", getuid());
 again:

   if (gDebug > 3)
      Info("GlobusGetNames", "Testing Proxy file: %s", proxy_file);

   if (!access(proxy_file, F_OK) && !access(proxy_file, R_OK)) {
      // Second: load the proxy certificate ...
      fcert = fopen(proxy_file, "r");
      if (fcert == 0 || !PEM_read_X509(fcert, &xcert, 0, 0)) {
         Error("GlobusGetNames", "Unable to load user proxy certificate ");
         return 5;
      }
      fclose(fcert);
      // Get proxy names
      char *ProxyIssuerName =
          StrDup(X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0));
      char *ProxySubjectName =
          StrDup(X509_NAME_oneline(X509_get_subject_name(xcert), 0, 0));
      if (gDebug > 3) {
         Info("GlobusGetNames", "Proxy Issuer Name: %s", ProxyIssuerName);
         Info("GlobusGetNames", "Proxy Subject Name: %s",
              ProxySubjectName);
      }

      if (strstr(ProxyIssuerName, *SubjectName) == ProxyIssuerName) {
         gNeedProxy = 0;
         setenv("X509_USER_PROXY", proxy_file, 1);
         if (ProxyIssuerName) delete[] ProxyIssuerName;
         if (ProxySubjectName) delete[] ProxySubjectName;
         if (gDebug > 3)
            Info("GlobusGetNames", "Using Proxy file:%s (gNeedProxy:%d)",
                 getenv("X509_USER_PROXY"), gNeedProxy);

         return retval;
      } else {
         sprintf(proxy_file, "/tmp/x509up_u%d.%d", getuid(), nProxy);
         nProxy++;
         if (ProxyIssuerName) delete[] ProxyIssuerName;
         if (ProxySubjectName) delete[] ProxySubjectName;
         goto again;
      }
   } else {
      setenv("X509_USER_PROXY", proxy_file, 1);
      return retval;
   }

   return retval;
}

//______________________________________________________________________________
int GlobusGetCredHandle(int LocalEnv, gss_cred_id_t * CredHandle)
{
   // Get Credential Handle, either from scratch, or from delegated info ...
   // Returns 0 is successfull, 1 otherwise.

   int retval = 0;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;

   if (gDebug > 2)
      Info("GlobusGetCredHandle", "Enter: LocalEnv: %d", LocalEnv);

   if (LocalEnv == 2) {
      // If we are a PROOF Master autheticating vs Slaves we only need to fetch the delegated
      // credentials from the shared memory segment the first time we are called ...
      if (GlbDelCredHandle == GSS_C_NO_CREDENTIAL) {
         if (GlobusGetDelCred()) {
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
      if ((gNeedProxy == 1) ||
          (MajStat =
           globus_gss_assist_acquire_cred(&MinStat, GSS_C_INITIATE,
                                          CredHandle)) != GSS_S_COMPLETE) {

         // Check if interactive session
         if (isatty(0) && isatty(1)) {

            if (gDebug > 3)
               Info("GlobusGetCredHandle",
                    "Failed to acquire credentials: trying to initialize proxies ...");

            // Try to get credentials with usual command line ...
            char *GlobusLocation = getenv("GLOBUS_LOCATION");
            if (GlobusLocation == 0) {
               Error("GlobusGetCredHandle",
                     "Please define a valid GLOBUS_LOCATION");
               retval = 2;
               goto exit;
            }
            // First check if there are special requests for proxy duration ...
            const char *duration =
                gEnv->GetValue("Globus.ProxyDuration", "default");
            char initdur[256] = { 0 };
            if (strstr(duration, "default") == 0) {
               sprintf(initdur, "-hours %s", duration);
            }
            if (gDebug > 3)
               Info("GlobusAutheticate", "initdur: %s (%s)", initdur,
                    duration);

            // ... and for number of bits in key ...
            const char *keybits =
                gEnv->GetValue("Globus.ProxyKeyBits", "default");
            char initbit[256] = { 0 };
            if (strstr(keybits, "default") == 0) {
               sprintf(initbit, "-bits %s", keybits);
            }
            if (gDebug > 3)
               Info("GlobusAutheticate", "initbit: %s (%s)", initbit,
                    keybits);

            // ... and for number of bits in key ...
            char *usrpxy = getenv("X509_USER_PROXY");
            char initpxy[256] = { 0 };
            sprintf(initpxy, "-out %s", usrpxy);
            if (gDebug > 3)
               Info("GlobusAutheticate", "initpxy: %s (%s)", initpxy,
                    usrpxy);

            // ... and environment variables
            char *cerdir = getenv("X509_CERT_DIR");
            char *usrcer = getenv("X509_USER_CERT");
            char *usrkey = getenv("X509_USER_KEY");
            char initenv[kMAXPATHLEN] = { 0 };
            sprintf(initenv,
                    "export X509_CERT_DIR=%s; export X509_USER_CERT=%s; export X509_USER_KEY=%s",
                    cerdir, usrcer, usrkey);

            // to execute command to initiate the proxies one needs
            // to source the globus shell environment
            char proxyinit[kMAXPATHLEN] = { 0 };
            sprintf(proxyinit,
                    "source $GLOBUS_LOCATION/etc/globus-user-env.sh; %s; grid-proxy-init %s %s %s",
                    initenv, initdur, initbit, initpxy);
            if (gDebug > 3)
               Info("GlobusAutheticate", "Executing: %s", proxyinit);
            gSystem->Exec(proxyinit);

            //  retry now
            if ((MajStat =
                 globus_gss_assist_acquire_cred(&MinStat, GSS_C_INITIATE,
                                                CredHandle)) !=
                GSS_S_COMPLETE) {
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
int GlobusCheckSecContext(char *Host, char *SubjName)
{
   // Checks if there is already a valid security context established with
   // remote host for subject ...
   // On success returns entry number, -1 otherwise.

   int retval = -1;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GlbContLifeTime = 0;

   if (gDebug > 2) {
      Info("GlobusCheckSecContext", "contacting host: %s", Host);
      Info("GlobusCheckSecContext",
           "we have got %d sec context handles in memory", NumGlbSecCont);
   }

   if (NumGlbSecCont > 0) {
      int i;
      for (i = 0; i < NumGlbSecCont; i++) {
         if (!strcmp(hostGlbSecCont[i], Host)
             && !strcmp(subjGlbSecCont[i], SubjName)) {
            if (gDebug > 3)
               Info("GlobusCheckSecContext",
                    "we already have a sec context with host: %s for subj: %s",
                    Host, SubjName);

            // Check validity of the retrieved context ...
            int Dum1, Dum2;
            gss_OID MechType;
            gss_name_t *TargName = 0, *Name = 0;
            if (sptrGlbSecCont[i] != 0
                && sptrGlbSecCont[i] != GSS_C_NO_CONTEXT) {
               if ((MajStat =
                    gss_inquire_context(&MinStat, sptrGlbSecCont[i], Name,
                                        TargName, &GlbContLifeTime,
                                        &MechType, &GssRetFlags, &Dum1,
                                        &Dum2)) != GSS_S_COMPLETE) {
                  GlobusError("GlobusCheckSecContext: gss_inquire_context",
                              MajStat, MinStat, 0);
                  GlobusUpdateSecContInfo(i);  // delete it from tables ...
               } else {
                  if (gDebug > 3)
                     Info("GlobusCheckSecContext",
                          "client (%s) already authenticated from host %s (remaining lifetime: %d sec)",
                          SubjName, Host, GlbContLifeTime);
                  retval = i;
               }
            }
         }
      }
   }
   return retval;
}

//______________________________________________________________________________
int GlobusUpdateSecContInfo(int entry)
{
   // Removes entries corresponding to expired, unvalid or deleted security
   // context. If entry=0 check the validity; if entry>0 remove 'entry'
   // without checking. Returns number of valid sec contexts established.

   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GlbContLifeTime = 0;
   int nGoodCont = 0;
   char **Hosts = new char *[NumGlbSecCont];
   char **Clien = new char *[NumGlbSecCont];
   gss_ctx_id_t *SCPtr = new gss_ctx_id_t[NumGlbSecCont];

   if (gDebug > 2)
      Info("GlobusUpdateSecContInfo", "Enter: entry: %d", entry);

   int i;
   for (i = 0; i < NumGlbSecCont; i++) {
      if (entry == 0) {
         int Dum1, Dum2;
         gss_OID MechType;
         gss_name_t *TargName = 0, *client_name = 0;

         // Check validity of the retrieved context ...
         if (sptrGlbSecCont[i] != 0
             && sptrGlbSecCont[i] != GSS_C_NO_CONTEXT) {
            if ((MajStat =
                 gss_inquire_context(&MinStat, sptrGlbSecCont[i],
                                     client_name, TargName,
                                     &GlbContLifeTime, &MechType,
                                     &GssRetFlags, &Dum1,
                                     &Dum2)) != GSS_S_COMPLETE) {
               GlobusError("GlobusUpdateSecContInfo: gss_inquire_context",
                           MajStat, MinStat, 0);
            } else {
               // This is a valid one ...
               Hosts[nGoodCont] = strdup(hostGlbSecCont[i]);
               Clien[nGoodCont] = strdup(subjGlbSecCont[i]);
               SCPtr[nGoodCont] = sptrGlbSecCont[i];
               nGoodCont++;
            }
         }
      } else if (entry != i) {
         Hosts[nGoodCont] = strdup(hostGlbSecCont[i]);
         Clien[nGoodCont] = strdup(subjGlbSecCont[i]);
         SCPtr[nGoodCont] = sptrGlbSecCont[i];
         nGoodCont++;
      }
   }

   // Update reference table
   if (hostGlbSecCont) delete[] hostGlbSecCont;
   if (subjGlbSecCont) delete[] subjGlbSecCont;
   if (sptrGlbSecCont) delete[] sptrGlbSecCont;
   NumGlbSecCont = nGoodCont;

   if (NumGlbSecCont > 0) {

      if (gDebug > 3)
         Info("GlobusUpdateSecContInfo",
              " %d valid established security contexts found");

      hostGlbSecCont = new char *[NumGlbSecCont];
      subjGlbSecCont = new char *[NumGlbSecCont];
      sptrGlbSecCont = new gss_ctx_id_t[NumGlbSecCont];

      for (i = 0; i < NumGlbSecCont; i++) {
         hostGlbSecCont[i] = strdup(Hosts[i]);
         subjGlbSecCont[i] = strdup(Clien[i]);
         sptrGlbSecCont[i] = SCPtr[i];
         if (gDebug > 3)
            Info("GlobusUpdateSecContInfo",
                 "Sec cont %d for subject %s with host %s", i, Clien[i],
                 Hosts[i]);
      }
   } else if (gDebug > 3) {
      Info("GlobusUpdateSecContInfo",
           "No valid established security contexts remains");
   }

   return nGoodCont;
}

//______________________________________________________________________________
void GlobusSetCertificates(int LocalEnv)
{
   // Defines certificate and key files to use, inquiring the client if needed.

   char *userhome = getenv("HOME");
   char *globusdef = ".globus";
   char *details = 0;
   Int_t nr, i;

   if (gDebug > 2)
      Info("GlobusSetCertificates", "Enter: LocalEnv: %d", LocalEnv);

   gDetails = "";

   if (LocalEnv < 2) {

      char temp[kMAXPATHLEN] = { 0 };

      // Defaults
      char tmpvar[4][kMAXPATHLEN];
      char *ddir = 0, *dcer = 0, *dkey = 0, *dadi = 0;
      if (strlen(TAuthenticate::GetDefaultUser()) > 0) {

	 details = StrDup(TAuthenticate::GetDefaultUser());

      } else {
         details =
         StrDup("cd:~/.globus cf:usercert.pem kf:userkey.pem ad:/etc/grid-security/certificates");
      }

      if (gDebug > 3)
         Info("GlobusSetCertificates", " details : %s", details);

      nr = sscanf(details, "%s %s %s %s", tmpvar[0], tmpvar[1], tmpvar[2],
                  tmpvar[3]);
      for (i = 0; i < nr; i++) {
         if (!strncmp(tmpvar[i], "cd:", 3) || !strncmp(tmpvar[i], "Cd:", 3)
             || !strncmp(tmpvar[i], "cD:", 3)
             || !strncmp(tmpvar[i], "CD:", 3))
            ddir = StrDup(tmpvar[i] + 3);
         if (!strncmp(tmpvar[i], "cf:", 3) || !strncmp(tmpvar[i], "Cf:", 3)
             || !strncmp(tmpvar[i], "cF:", 3)
             || !strncmp(tmpvar[i], "CF:", 3))
            dcer = StrDup(tmpvar[i] + 3);
         if (!strncmp(tmpvar[i], "kf:", 3) || !strncmp(tmpvar[i], "Kf:", 3)
             || !strncmp(tmpvar[i], "kF:", 3)
             || !strncmp(tmpvar[i], "KF:", 3))
            dkey = StrDup(tmpvar[i] + 3);
         if (!strncmp(tmpvar[i], "ad:", 3) || !strncmp(tmpvar[i], "Ad:", 3)
             || !strncmp(tmpvar[i], "aD:", 3)
             || !strncmp(tmpvar[i], "AD:", 3))
            dadi = StrDup(tmpvar[i] + 3);
      }
      if (ddir == 0)
         ddir = StrDup("~/.globus");
      if (dcer == 0)
         dcer = StrDup("usercert.pem");
      if (dkey == 0)
         dkey = StrDup("userkey.pem");
      if (dadi == 0)
         dadi = StrDup("/etc/grid-security/certificates");

      // Check if needs to prompt the client
      char *det = 0;
      if (gPrompt) {
         char *dets = 0;
         if (!gROOT->IsProofServ()) {
            dets =
                Getline(Form
                        (" Local Globus Certificates (%s)\n Enter <key>:<new value> to change: ",
                         details));
         } else {
            Warning("GlobusSetCertificate",
                    "proofserv: cannot prompt for info");
         }
         if (dets && dets[0]) {
            dets[strlen(dets) - 1] = '\0';  // get rid of \n
            det = new char[strlen(dets)];
            strcpy(det, dets);
         } else
            det = "";

         if (gDebug > 3)
            Info("GlobusSetCertificates", "got det: %s (%d)", det,
                 strlen(det));

      } else {
         det = "";
      }

      if (details) delete[] details;

      if (strlen(det) > 0) {

         nr = sscanf(det, "%s %s %s %s", tmpvar[0], tmpvar[1], tmpvar[2],
                     tmpvar[3]);
         for (i = 0; i < nr; i++) {
            if (!strncmp(tmpvar[i], "cd:", 3)
                || !strncmp(tmpvar[i], "Cd:", 3)
                || !strncmp(tmpvar[i], "cD:", 3)
                || !strncmp(tmpvar[i], "CD:", 3)) {
               if (ddir) delete[] ddir;
               ddir = StrDup(tmpvar[i] + 3);
            }
            if (!strncmp(tmpvar[i], "cf:", 3)
                || !strncmp(tmpvar[i], "Cf:", 3)
                || !strncmp(tmpvar[i], "cF:", 3)
                || !strncmp(tmpvar[i], "CF:", 3)) {
               if (dcer) delete[] dcer;
               dcer = StrDup(tmpvar[i] + 3);
            }
            if (!strncmp(tmpvar[i], "kf:", 3)
                || !strncmp(tmpvar[i], "Kf:", 3)
                || !strncmp(tmpvar[i], "kF:", 3)
                || !strncmp(tmpvar[i], "KF:", 3)) {
               if (dkey) delete[] dkey;
               dkey = StrDup(tmpvar[i] + 3);
            }
            if (!strncmp(tmpvar[i], "ad:", 3)
                || !strncmp(tmpvar[i], "Ad:", 3)
                || !strncmp(tmpvar[i], "aD:", 3)
                || !strncmp(tmpvar[i], "AD:", 3)) {
               if (dadi) delete[] dadi;
               dadi = StrDup(tmpvar[i] + 3);
            }
         }
      }
      // Build gDetails
      temp[0] = '\0';
      sprintf(temp, "%s cd:%s cf:%s kf:%s ad:%s", gPromptReUse, ddir, dcer,
              dkey, dadi);
      gDetails = temp;

      // Perform "~" expansion ... or allow for paths relative to .globus
      if (!strncmp(ddir, "~/", 2)) {
         temp[0] = '\0';
         sprintf(temp, "%s%s", userhome, ddir + 1);
         if (ddir) delete[] ddir;
         ddir = StrDup(temp);
      } else if (strncmp(ddir, "/", 1)) {
         temp[0] = '\0';
         sprintf(temp, "%s/%s/%s", userhome, globusdef, ddir);
         if (ddir) delete[] ddir;
         ddir = StrDup(temp);
      }
      if (!strncmp(dcer, "~/", 2)) {
         temp[0] = '\0';
         sprintf(temp, "%s%s", userhome, dcer + 1);
         if (dcer) delete[] dcer;
         dcer = StrDup(temp);
      } else if (strncmp(dcer, "/", 1)) {
         temp[0] = '\0';
         sprintf(temp, "%s/%s", ddir, dcer);
         if (dcer) delete[] dcer;
         dcer = StrDup(temp);
      }
      if (!strncmp(dkey, "~/", 2)) {
         temp[0] = '\0';
         sprintf(temp, "%s%s", userhome, dkey + 1);
         if (dkey) delete[] dkey;
         dkey = StrDup(temp);
      } else if (strncmp(dkey, "/", 1)) {
         temp[0] = '\0';
         sprintf(temp, "%s/%s", ddir, dkey);
         if (dkey) delete[] dkey;
         dkey = StrDup(temp);
      }
      if (!strncmp(dadi, "~/", 2)) {
         temp[0] = '\0';
         sprintf(temp, "%s%s", userhome, dadi + 1);
         if (dadi) delete[] dadi;
         dadi = StrDup(temp);
      } else if (strncmp(dadi, "/", 1)) {
         temp[0] = '\0';
         sprintf(temp, "%s/%s/%s", userhome, globusdef, dadi);
         if (dadi) delete[] dadi;
         dadi = StrDup(temp);
      }
      if (gDebug > 3)
         Info("GlobusSetCertificates", "after expansion: %s %s %s", dcer,
              dkey, dadi);

      // Save them
      setenv("X509_CERT_DIR", dadi, 1);
      setenv("X509_USER_CERT", dcer, 1);
      setenv("X509_USER_KEY", dkey, 1);

      // Release allocated memory
      if (ddir) delete[] ddir;
      if (dcer) delete[] dcer;
      if (dkey) delete[] dkey;
      if (dadi) delete[] dadi;

   }

   return;
}

//______________________________________________________________________________
void GlobusCleanup()
{
   // This function cleans up any stuff related to Globus, releasing
   // credentials, security contexts and allocated memory ...

   if (gDebug > 2)
      Info("GlobusCleanup:", "cleaning up local Globus stuff ...");

   if (NumGlbSecCont == 0) {
      if (gDebug > 3)
         Info("GlobusCleanup:", "Globus never used: nothing to clean");
      return;
   }

   int status = 0;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;


   // Now we can delete the sec context */
   int i;
   for (i = 0; i < NumGlbSecCont; i++) {
      if ((MajStat =
           gss_delete_sec_context(&MinStat, sptrGlbSecCont + i,
                                  GSS_C_NO_BUFFER)) != GSS_S_COMPLETE) {
         GlobusError("GlobusCleanup: gss_delete_sec_context", MajStat,
                     MinStat, 0);
         status = 1;
      }
   }

   // Release memory allocated for security contexts ...
   if (hostGlbSecCont) delete[] hostGlbSecCont;
   if (subjGlbSecCont) delete[] subjGlbSecCont;
   if (sptrGlbSecCont) delete[] sptrGlbSecCont;
   NumGlbSecCont = 0;

   // Finally check if the shm for imported stuff has not yet been destroyed ...
   // Recall the shm_id first ...
   TApplication *lApp = gROOT->GetApplication();

   if (lApp != 0) {
      if (lApp->Argc() > 7 && gROOT->IsProofServ()) {
         struct shmid_ds shm_ds;
         int rc;
         // Delegated Credentials
         gShmIdCred = atoi(lApp->Argv()[7]);
         if (gShmIdCred != -1) {
            if ((rc = shmctl(gShmIdCred, IPC_RMID, &shm_ds)) != 0) {
               if ((rc == EINVAL) || (rc == EIDRM)) {
                  if (gDebug > 3)
                     Info("GlobusCleanup:",
                          "credentials shared memory segment already marked as destroyed");
               } else {
                  Warning("GlobusCleanup:",
                          "unable to mark segment as destroyed (error: 0x%x)",
                          rc);
               }
            } else if (gDebug > 3)
               Info("GlobusCleanup:",
                    "shared memory segment %d marked for destruction",
                    gShmIdCred);
         } else if (gDebug > 3) {
            Info("GlobusCleanup:",
                 "gShmIdCred not defined in this session");
         }
      }
   }
}
