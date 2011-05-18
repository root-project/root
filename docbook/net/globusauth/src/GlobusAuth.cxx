// @(#)root/globus:$Id$
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

#include "RConfigure.h"

#include <errno.h>
#include <signal.h>
#include <string.h>
// #include <stdlib.h>

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
#ifdef IOV_MAX
#undef IOV_MAX
#endif
#include <globus_common.h>
#include <globus_gss_assist.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include <sys/ipc.h>
#include <sys/shm.h>
}

static gss_cred_id_t gGlbCredHandle = GSS_C_NO_CREDENTIAL;
static gss_cred_id_t gGlbDelCredHandle = GSS_C_NO_CREDENTIAL;
static int gShmIdCred = -1;

Int_t GlobusAuthenticate(TAuthenticate *, TString &, TString &);
Int_t GlobusCheckSecContext(const char *, gss_ctx_id_t);
Int_t GlobusCheckSecCtx(const char *, TRootSecContext *);
Int_t GlobusCleanupContext(gss_ctx_id_t);
void  GlobusCleanupShm();
Int_t GlobusIssuerName(TString &);
void  GlobusError(const char *, OM_uint32, OM_uint32, Int_t);
Int_t GlobusGetCredHandle(Int_t, gss_cred_id_t *);
Int_t GlobusGetDelCred();
void  GlobusGetDetails(Int_t, Int_t, TString &);
Int_t GlobusGetLocalEnv(Int_t *, TString);
Int_t GlobusGetSecContLifeTime(gss_ctx_id_t);
Int_t GlobusNameFromCred(gss_cred_id_t, TString &);

class GlobusAuthInit {
 public:
   GlobusAuthInit() {
      TAuthenticate::SetGlobusAuthHook(&GlobusAuthenticate);
}};
static GlobusAuthInit globusauth_init;

//______________________________________________________________________________
Int_t GlobusAuthenticate(TAuthenticate * tAuth, TString & user,
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
   gss_ctx_id_t glbContextHandle = GSS_C_NO_CONTEXT;
   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;
   OM_uint32 gssRetFlags = 0;
   OM_uint32 gssReqFlags = 0;
   int glbTokenStatus = 0;
   char *host_subj = 0;
   TDatime expDate = TDatime();

   // Check if called for cleanup
   if (user == "-1") {
      if (gDebug > 2)
         Info("GlobusAuthenticate", " cleanup call (%s)",details.Data());

      if (details == "context") {
         // Security context cleaning
         GlobusCleanupContext((gss_ctx_id_t)tAuth);
      } else if (details == "shm") {
         // Shared memory cleaning (TProofServ)
         GlobusCleanupShm();
      }
      return 1;
   }

   // From the calling TAuthenticate
   TSocket *sock = tAuth->GetSocket();
   TString protocol = tAuth->GetProtocol();

   if (gDebug > 2)
      Info("GlobusAuthenticate", " enter: protocol:'%s' user:'%s'", protocol.Data(),
           user.Data());

   // The host FQDN ... for debugging
   const char *hostFQDN = sock->GetInetAddress().GetHostName();

   // Determine local calling environment ...
   Int_t localCallEnv = -1;
   if ((rc = GlobusGetLocalEnv(&localCallEnv, protocol))) {
      if (gDebug > 0)
          Error("GlobusAuthenticate",
            "unable to set relevant environment variables (rc=%d)",
            rc);
      return -1;
   }
   if (gDebug > 3)
      Info("GlobusAuthenticate", " localCallEnv is %d", localCallEnv);

   // Get credential handle ... either genuine or delegated
   if (GlobusGetCredHandle(localCallEnv, &gGlbCredHandle)) {
      if (gDebug > 0)
         Error("GlobusAuthenticate", "unable to acquire valid credentials");
      return -1;
   }
   if (gDebug > 3)
      Info("GlobusAuthenticate", " Credential Handle is 0x%x",
           gGlbCredHandle);

   // Inquire credentials for subject name and convert it in human readable form ...
   TString ssuj;
   if ((rc = GlobusNameFromCred(gGlbCredHandle, ssuj))) {
      if (gDebug > 0)
         Error("GlobusAuthenticate",
               "PROOF Master: unable to determine name from cred (rc=%d)", rc);
      return -1;
   }

   // Create Options string
   Int_t opt = TAuthenticate::GetAuthReUse() * kAUTH_REUSE_MSK +
               tAuth->GetRSAKeyType() * kAUTH_RSATY_MSK;
   TString options(Form("%d %d %s", opt, ssuj.Length(), ssuj.Data()));

   // Check established authentications
   kind = kROOTD_GLOBUS;
   retval = TAuthenticate::GetAuthReUse();
   if ((rc = tAuth->AuthExists(ssuj, TAuthenticate::kGlobus, options,
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
         Info("GlobusAuthenticate", "server does not support Globus authentication");
      return -1;
   }

   if (sock->GetRemoteProtocol() < 18) {
      TString isuj;
      if (GlobusIssuerName(isuj)) {
         if (gDebug > 0)
            Error("GlobusAuthenticate",
                  "unable to determine issuer name from certificate");
         return 0;
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
         return 0;
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
   int sockFd = sock->GetDescriptor();
   FILE *sockStream = fdopen(sockFd, "w+");

   // Type of request for credentials depend on calling environment
   gssReqFlags =
       localCallEnv >
       0 ? (GSS_C_DELEG_FLAG | GSS_C_MUTUAL_FLAG) : GSS_C_MUTUAL_FLAG;
   if (gDebug > 3)
      Info("GlobusAuthenticate",
           " gssReqFlags: %p, GlbCredentials: %p", gssReqFlags, gGlbCredHandle);

   // Now we are ready to start negotiating with the Server
   if ((majStat =
        globus_gss_assist_init_sec_context(&minStat, gGlbCredHandle,
                                           &glbContextHandle, host_subj,
                                           gssReqFlags, &gssRetFlags,
                                           &glbTokenStatus,
                                           globus_gss_assist_token_get_fd,
                                           (void *) sockStream,
                                           globus_gss_assist_token_send_fd,
                                           (void *) sockStream)) !=
       GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusAuthenticate: gss_assist_init_sec_context",
                  majStat, minStat, glbTokenStatus);
      if (host_subj) delete[] host_subj;
      sock->Send(0,kROOTD_ERR);
      return 0;
   } else {
      // Set expiration date
      expDate.Set(expDate.Convert() + GlobusGetSecContLifeTime(glbContextHandle));
      if (gDebug > 2) {
         Info("GlobusAuthenticate", "authenticated to host %s", hostFQDN);
         Info("GlobusAuthenticate", "expiring on '%s'", expDate.AsString());
      }
      if (fflush(sockStream) != 0) {
         Warning("GlobusAuthenticate", "unable to fflush socket:"
                 " may cause authentication problems on server side");
      }
      auth = 1;
   }

   // Now we have the subject and we can release some resources ...
   if (host_subj) delete[] host_subj;

   // Receive username used for login or key request info and type of key
   int nrec = sock->Recv(retval, type);  // returns user

   Int_t rsaKey = 0;
   if (type == kROOTD_RSAKEY) {
      if (retval <= 0 || retval > 2)
         Warning("GlobusAuthenticate",
                 "problems recvn RSA key flag: got message %d, retval: %d",
                 type, retval);
      rsaKey = retval - 1;

      // Send the key securely
      TAuthenticate::SendRSAPublicKey(sock,rsaKey);

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
   Int_t offSet = -1;
   sscanf(rfrm, "%s %d", lUser, &offSet);

   // Return username
   user = lUser;

   // Receive token
   char *token = 0;
   if (TAuthenticate::GetAuthReUse() == 1 && offSet > -1) {
      if (TAuthenticate::SecureRecv(sock, 1, rsaKey, &token) == -1) {
         Warning("GlobusAuthenticate",
                 "Problems secure-receiving token -"
                 " may result in corrupted token");
      }
      if (gDebug > 3)
         Info("GlobusAuthenticate", "received from server: token: '%s' ",
              token);
   } else {
      token = StrDup("");
   }

   // Create SecContext object
   GlobusGetDetails(localCallEnv, 0, details);
   TRootSecContext *ctx =
      tAuth->GetHostAuth()->CreateSecContext((const char *)lUser,
          hostFQDN, (Int_t)TAuthenticate::kGlobus, offSet,
          details, (const char *)token, expDate,
          (void *)glbContextHandle, rsaKey);
   // Transmit it to TAuthenticate
   tAuth->SetSecContext(ctx);

   // receive status from server
   sock->Recv(server_auth, kind);
   if (gDebug > 2)
      Info("GlobusAuthenticate", "received auth status from server: %d (%d)",
           server_auth, kind);

   if (auth && !server_auth) {
      Warning("GlobusAuthenticate",
              " it looks like server did not authenticate: probably a problem with mapping");
      auth = 0;
   }

   // free allocated memory ...
   if (rfrm) delete[] rfrm;
   if (lUser) delete[] lUser;
   if (token) delete[] token;

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
   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;

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
   if ((majStat =
        gss_import_cred(&minStat, &gGlbDelCredHandle, 0, 0, credential, 0,
                        0)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusGetDelCred: gss_import_cred", majStat, minStat, 0);
      return 1;
   } else if (gDebug > 3)
      Info("GlobusGetDelCred:",
           "Globus Credentials successfully imported (0x%x)",
           gGlbDelCredHandle);

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
void GlobusError(const char *mess, OM_uint32 majs, OM_uint32 mins, int toks)
{
   // Handle error ...

   char *glbErr = 0;

   if (!globus_gss_assist_display_status_str
       (&glbErr, (char *)mess, majs, mins, toks)) {
        Error("GlobusError:","%s (majst=%d,minst=%d,tokst:%d)",
                             glbErr, majs, mins, toks);
   } else {
      Error("GlobusError:","%s (not resolved) (majst=%d,minst=%d,tokst:%d)",
                           mess, majs, mins, toks);
   }

   if (glbErr) delete[] glbErr;
}

//______________________________________________________________________________
Int_t GlobusGetLocalEnv(Int_t *localEnv, TString protocol)
{
   // Determines calling environment.
   // Returns 0 if successful; 1 otherwise.

   int retval = 0;

   // Calling application
   TApplication *lApp = gROOT->GetApplication();
   if (gDebug > 2) {
      int i = 0;
      for (; i < lApp->Argc(); i++) {
         Info("GlobusGetLocalEnv", "application arguments: %d: %s", i,
              lApp->Argv(i));
      }
   }

   *localEnv = 0;
   if (lApp != 0) {
      if (gROOT->IsProofServ()) {
         // This is PROOF ... either Master or Slave ...
         if (gDebug > 3) {
            Info("GlobusGetLocalEnv",
                 "PROOF environment, called by the MASTER/SLAVE");
            Info("GlobusGetLocalEnv",
                 "string with pointer to del cred is 0x%x",
                 gGlbDelCredHandle);
         }
         *localEnv = 2;
         gShmIdCred = -1;
         const char *p = gSystem->Getenv("ROOTSHMIDCRED");
         if (p)
            gShmIdCred = strtol(p, (char **)0, 10);
         if (gShmIdCred <= 0) {
            Info("GlobusGetLocalEnv",
                    "delegate credentials undefined");
            retval = 1;
         }
      } else {
         if (strstr(protocol.Data(), "proof") != 0) {
            if (gDebug > 3)
               Info("GlobusGetLocalEnv",
                    "PROOF environment, called by the CLIENT");
            *localEnv = 1;
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
Int_t GlobusNameFromCred(gss_cred_id_t cred, TString &subjName)
{
   // Get subject name from credential handle cred.
   // Returns 0 is successfull, 1 otherwise.

   if (gDebug > 2)
      Info("GlobusNamesFromCred", "Enter: Handle: 0x%p", cred);

   // Inquire credentials for Subject name and convert it in human readable form ...
   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;
   gss_name_t name;
   OM_uint32 lifeTime;
   gss_cred_usage_t credUsage;
   gss_OID_set mech;
   if ((majStat = gss_inquire_cred(&minStat, cred, &name,
                  &lifeTime, &credUsage, &mech)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusNameFromCred: gss_inquire_cred",
                     majStat, minStat,0);
      return 1;
   }
   gss_buffer_desc outBuf;
   gss_OID nameType;
   if ((majStat = gss_display_name(&minStat, name, &outBuf,
                  &nameType)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusNameFromCred: gss_display_name",
                     majStat, minStat, 0);
      return 2;
   } else
      subjName = TString((const char *)outBuf.value);

   // Notify
   if (gDebug > 2)
      Info("GlobusNameFromCred", "subject name: %s", subjName.Data());

   // Successful
   return 0;
}

//______________________________________________________________________________
Int_t GlobusGetSecContLifeTime(gss_ctx_id_t ctx)
{
   // Returns lifetime of established sec context 'ctx'

   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;
   OM_uint32 gssRetFlags = 0;
   OM_uint32 glbContLifeTime = 0;
   gss_OID   mechType;
   gss_name_t *targName = 0, *name = 0;
   int       dum1, dum2;

   if (ctx != 0 && ctx != GSS_C_NO_CONTEXT) {
      if ((majStat = gss_inquire_context(&minStat, ctx, name,
                     targName, &glbContLifeTime, &mechType, &gssRetFlags,
                     &dum1, &dum2)) != GSS_S_COMPLETE) {
         if (gDebug > 0)
            GlobusError("GlobusGetSecContLifeTime: gss_inquire_context",
                          majStat, minStat, 0);
         return 0;
      } else {
         if (gDebug > 3)
            Info("GlobusGetSecContLifeTime"," remaining lifetime: %d sec",
                  glbContLifeTime);
         return (Int_t)glbContLifeTime;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t GlobusCleanupContext(gss_ctx_id_t ctx)
{
   // This function cleans up security context ctx

   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;

   // Delete context
   if ((majStat = gss_delete_sec_context(&minStat, &ctx,
                  GSS_C_NO_BUFFER)) != GSS_S_COMPLETE) {
      if (gDebug > 0)
         GlobusError("GlobusCleanupContext: gss_delete_sec_context",
                     majStat,minStat, 0);
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
Int_t GlobusCheckSecContext(const char *subjName, gss_ctx_id_t ctx)
{
   // Checks if SubjName match the one assigned to sec context Ctx
   // Check also validity of Ctx.
   // Returns 1 if everything is ok, 0 if non-matching
   // -1 if Ctx is no more valid and should be discarded

   if (!ctx)
      return 0;

   int rc = 0;
   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;
   OM_uint32 gssRetFlags = 0;
   OM_uint32 glbContLifeTime = 0;

   if (gDebug > 2)
      Info("GlobusCheckSecContext", "checking subj:%s", subjName);

   // Check validity of the retrieved context ...
   Int_t dum1, dum2;
   gss_OID mechType;
   gss_name_t *targName = 0;
   gss_name_t name;
   if (ctx != 0 && ctx != GSS_C_NO_CONTEXT) {

      if ((majStat = gss_inquire_context(&minStat, ctx, &name,
                     targName, &glbContLifeTime, &mechType,
                     &gssRetFlags, &dum1, &dum2)) != GSS_S_COMPLETE) {
         if (gDebug > 0)
            GlobusError("GlobusCheckSecContext: gss_inquire_context",
                                                majStat, minStat, 0);
         rc = -1;
      } else {
         gss_buffer_desc nameBuffer;
         // Get the subject name now
         if ((majStat = gss_display_name(&minStat, name, &nameBuffer,
                                        GLOBUS_NULL)) != GSS_S_COMPLETE) {
            if (gDebug > 0)
               GlobusError("GlobusCheckSecContext: gss_display_name",
                                                 majStat, minStat, 0);
            nameBuffer.length = 0;
            nameBuffer.value = GLOBUS_NULL;
         } else {
            char *theName = new char[nameBuffer.length+1];
            strncpy(theName,(char *)(nameBuffer.value),(Int_t)(nameBuffer.length));
            theName[nameBuffer.length]= '\0';
            if (gDebug > 2)
               Info("GlobusCheckSecContext","with subject name: %s (%d)",
                                             theName, nameBuffer.length);
            if (!strcmp(theName, subjName)) {
               if (gDebug > 2)
                  Info("GlobusCheckSecContext",
                       "client already authenticated (remaining lifetime: %d sec)",
                        glbContLifeTime);
               rc = 1;
            }
            // Release allocated space
            if (theName)
               delete[] theName;
            if ((majStat = gss_release_name(&minStat, &name))
                                          != GSS_S_COMPLETE) {
               if (gDebug > 0)
                   GlobusError("GlobusCheckSecContext: gss_release_name",
                                                      majStat, minStat, 0);
            }
         }
      }

   } else {
      rc = -1;
   }

   return rc;
}

//______________________________________________________________________________
int GlobusGetCredHandle(Int_t localEnv, gss_cred_id_t * credHandle)
{
   // Get Credential Handle, either from scratch, or from delegated info ...
   // Returns 0 is successfull, 1 otherwise.

   int retval = 0;
   OM_uint32 majStat = 0;
   OM_uint32 minStat = 0;

   if (gDebug > 2)
      Info("GlobusGetCredHandle", "Enter: LocalEnv: %d", localEnv);

   if (localEnv == 2) {
      // If we are a PROOF Master autheticating vs Slaves
      // we only need to fetch the delegated credentials
      // from the shared memory segment the first time we are called ...
      if (gGlbDelCredHandle == GSS_C_NO_CREDENTIAL) {
         if (GlobusGetDelCred()) {
            if (gDebug > 0)
               Error("GlobusGetCredHandle",
                  "unable to fetch valid credentials from the shared memory segment");
            retval = 1;
            goto exit;
         }
      }
      *credHandle = gGlbDelCredHandle;
   } else {

      // Inquire Globus credentials:
      // This is looking to file X509_USER_PROXY for valid a X509 cert
      // (default /tmp/x509up_u<uid> )
      if ((majStat =
           globus_gss_assist_acquire_cred(&minStat, GSS_C_INITIATE,
                                          credHandle)) != GSS_S_COMPLETE) {

         // Check if interactive session
         if (isatty(0) && isatty(1)) {

           // Check special settings for the certificates
           TString det;
           GlobusGetDetails(localEnv, 1, det);

           if (gDebug > 3) {
               GlobusError("GlobusNameFromCred: gss_display_name",
                     majStat, minStat, 0);
               Info("GlobusGetCredHandle",
                    "Failed to acquire credentials: trying to initialize proxies ...");
            }

            // Try to get credentials with usual command line ...
            // First check if there are special requests for proxy duration ...
            TString initDur(gEnv->GetValue("Globus.ProxyDuration", "default"));
            if (!initDur.Contains("default")) {
               initDur.Insert(0,"-hours ");
               if (gDebug > 2)
                  Info("GlobusGetCredHandle", "initDur: %s (%s)", initDur.Data(),
                      gEnv->GetValue("Globus.ProxyDuration", "default"));
            } else
               initDur = TString("");

            // ... and for number of bits in key ...
            TString initBit(gEnv->GetValue("Globus.ProxyKeyBits", "default"));
            if (!initBit.Contains("default")) {
               initBit.Insert(0,"-bits ");
               if (gDebug > 2)
                  Info("GlobusGetCredHandle", "initBit: %s (%s)", initBit.Data(),
                      gEnv->GetValue("Globus.ProxyKeyBits", "default"));
            } else
               initBit = TString("");

            // ... and the proxy ...
            TString initPxy;
            if (gSystem->Getenv("X509_USER_PROXY")) {
               initPxy = Form("-out %s", gSystem->Getenv("X509_USER_PROXY"));
               if (gDebug > 3)
                  Info("GlobusGetCredHandle", "initPxy: %s", initPxy.Data());
            }

            // ... and environment variables
            TString initEnv(Form("export X509_CERT_DIR=%s",
               gSystem->Getenv("X509_CERT_DIR")));
            initEnv += TString(Form("; export X509_USER_CERT=%s",
               gSystem->Getenv("X509_USER_CERT")));
            initEnv += TString(Form("; export X509_USER_KEY=%s",
               gSystem->Getenv("X509_USER_KEY")));
            if (gDebug > 3)
               Info("GlobusGetCredHandle", "initEnv: %s", initEnv.Data());

            // to execute command to initiate the proxies one needs
            // to source the globus shell environment
            TString proxyInit;
            if (gSystem->Getenv("GLOBUS_LOCATION"))
               proxyInit = TString("source $GLOBUS_LOCATION/etc/globus-user-env.sh; ");
            proxyInit += initEnv;
            proxyInit += Form("; grid-proxy-init %s %s %s",
                               initDur.Data(), initBit.Data(), initPxy.Data());
            gSystem->Exec(proxyInit);

            //  retry now
            if ((majStat =
                 globus_gss_assist_acquire_cred(&minStat, GSS_C_INITIATE,
                                                credHandle)) !=
                GSS_S_COMPLETE) {
               if (gDebug > 0)
                  GlobusError("GlobusGetCredHandle: gss_assist_acquire_cred",
                           majStat, minStat, 0);
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
void GlobusGetDetails(Int_t localEnv, Int_t opt, TString &details)
{
   // Resolve the details string from localEnv. If opt == 0 just fill the string,
   // otherwise initialize the envs, prompting the user, if needed
   // Returns 0 is successfull, 1 otherwise.

   if (localEnv < 2) {

      // User settings
      Int_t reUse = TAuthenticate::GetAuthReUse();
      Int_t prompt = TAuthenticate::GetPromptUser();
      if (gDebug > 2)
         Info("GlobusGetDetails", "prompt: %d, reUse: %d", prompt, reUse);

      // System defaults
      TString ddir = "~/.globus";
      TString dcer = "usercert.pem";
      TString dkey = "userkey.pem";
      TString dadi = "/etc/grid-security/certificates";

      // User defaults
      if (strlen(TAuthenticate::GetDefaultUser()) > 0) {
         TString dets = TAuthenticate::GetDefaultUser();
         TString t;
         Int_t from = 0;
         while (dets.Tokenize(t,from," ")) {
            if (t.BeginsWith("cd:"))
               ddir = t.ReplaceAll("cd:", "");
            else if (t.BeginsWith("cf:"))
               dcer = t.ReplaceAll("cf:", "");
            else if (t.BeginsWith("kf:"))
               dkey = t.ReplaceAll("kf:", "");
            else if (t.BeginsWith("ad:"))
               dadi = t.ReplaceAll("ad:", "");
         }
      }

      // Check if needs to prompt the client
      if (TAuthenticate::GetPromptUser()) {
         TString ppt(Form(" Local Globus settings (%s %s %s %s)\n"
                          " Enter <key>:<new value> to change: ",
                          ddir.Data(), dcer.Data(), dkey.Data(), dadi.Data()));

         TString indet;
         if (!gROOT->IsProofServ()) {
            indet = Getline(ppt);
            // get rid of \n
            indet.Remove(TString::kTrailing, '\n');
            if (indet.Length() > 0) {
               TString t;
               Int_t from = 0;
               while (indet.Tokenize(t,from," ")) {
                  if (t.BeginsWith("cd:"))
                     ddir = t.ReplaceAll("cd:", "");
                  else if (t.BeginsWith("cf:"))
                     dcer = t.ReplaceAll("cf:", "");
                  else if (t.BeginsWith("kf:"))
                     dkey = t.ReplaceAll("kf:", "");
                  else if (t.BeginsWith("ad:"))
                     dadi = t.ReplaceAll("ad:", "");
               }
            }
         } else {
            Warning("GlobusGetDetails",
                    "proofserv: cannot prompt for info");
         }
      }

      // Build Details
      details = Form("pt:%d ru:%d %s %s %s %s",
                     TAuthenticate::GetPromptUser(),
                     TAuthenticate::GetAuthReUse(),
                     ddir.Data(), dcer.Data(), dkey.Data(), dadi.Data());

      // Set the envs, if required
      if (opt > 0) {

         // Perform "~" expansion ...
         gSystem->ExpandPathName(ddir);
         gSystem->ExpandPathName(dcer);
         gSystem->ExpandPathName(dkey);
         gSystem->ExpandPathName(dadi);

         // or allow for paths relative to .globus
         if (!ddir.BeginsWith("/"))
            ddir.Insert(0, Form("%s/.globus/", gSystem->HomeDirectory()));
         if (!dcer.BeginsWith("/"))
            dcer.Insert(0, Form("%s/", ddir.Data()));
         if (!dkey.BeginsWith("/"))
            dkey.Insert(0, Form("%s/", ddir.Data()));
         if (!dadi.BeginsWith("/"))
            dadi.Insert(0, Form("%s/.globus/", gSystem->HomeDirectory()));

         if (gDebug > 3)
            Info("GlobusSetCertificates", "after expansion: %s %s %s",
                 dcer.Data(), dkey.Data(), dadi.Data());
         // Save them
         gSystem->Setenv("X509_CERT_DIR", dadi);
         gSystem->Setenv("X509_USER_CERT", dcer);
         gSystem->Setenv("X509_USER_KEY", dkey);
      }
   }

   // Done
   return;
}

//______________________________________________________________________________
Int_t GlobusIssuerName(TString &issuerName)
{
   // Get issuer name from the certificate read either from the
   // certificate file or from the proxy file.
   // Returns 0 is successfull, 1 otherwise.

   if (gDebug > 2)
      Info("GlobusIssuerName", "enter");

   // Locate the relevant file
   TString fn = gSystem->Getenv("X509_USER_PROXY");
   if (fn.Length() <= 0)
      fn = Form("/tmp/x509up_u%d",gSystem->GetUid());
   if (gSystem->AccessPathName(fn, kReadPermission)) {
      TString emsg = Form("cannot read requested file(s): %s ", fn.Data());
      // Not available: try the certificate file itself
      fn = gSystem->Getenv("X509_USER_CERT");
      if (fn.Length() <= 0)
         fn = Form("%s/.globus/usercert.pem",gSystem->HomeDirectory());
      if (gSystem->AccessPathName(fn, kReadPermission)) {
         emsg += fn;
         Error("GlobusIssuerName", emsg.Data());
         return 1;
      }
   }

   // Load the certificate ...
   X509 *xcert = 0;
   FILE *fcert = fopen(fn.Data(), "r");
   if (!fcert) {
      Error("GlobusIssuerName", "unable to open file %s", fn.Data());
      return 1;
   }

   // Read certificate(s) from the file
   Bool_t notfound = kTRUE;
   while (notfound && PEM_read_X509(fcert, &xcert, 0, 0)) {
      // Retrieve issuer name
      char *in = X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0);
      // Retrieve subject name
      char *cn = X509_NAME_oneline(X509_get_subject_name(xcert), 0, 0);
      if (strncmp(in, cn, strlen(in))) {
         // This is the certificate
         issuerName = in;
         notfound = kFALSE;
      }
      free(in); free(cn);
   }
   fclose(fcert);

   // Notify failure
   if (notfound) {
      Error("GlobusIssuerName", "certificate not found in file %s", fn.Data());
      return 1;
   }

   // Notify
   if (gDebug > 2)
      Info("GlobusIssuerName", "issuer name: %s", issuerName.Data());

   // Successful
   return 0;
}
