// @(#)root/rpdutils:$Id$
// Author: Gerardo Ganis    7/4/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// globus                                                               //
//                                                                      //
// Set of utilities for rootd/proofd daemon authentication via Globus   //
// certificates.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pwd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>

#include "rpdp.h"


extern int gDebug;

namespace ROOT {


//--- Globals ---------------------------------------------------------------

//___________________________________________________________________________
void GlbsToolError(char *msg, int maj, int min, int tok)
{
   // Handle error ...

   char *e = 0;

   if (globus_gss_assist_display_status_str(&e, msg, maj, min, tok) || !e) {
      ErrorInfo("Error: %s: error messaged not resolved"
                " (majst=%d,minst=%d,tokst:%d)", msg, maj, min, tok);
   } else {
      ErrorInfo("Error: %s (majst=%d,minst=%d,tokst:%d)", e, maj, min, tok);
      delete [] e;
   }
   NetSend(kErrFatal, kROOTD_ERR);
}

//_________________________________________________________________________
int GlbsToolCheckCert(char **subjname)
{
   // Load information about available certificate and find our
   // subject name (needed by the client).
   // Returns 0 on success, 1 otherwise

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckCert: enter");

   // Get the path to the hostcert.conf file
   std::string hcconf = "/hostcert.conf";
   if (getenv("ROOTHOSTCERT")) {
      hcconf = getenv("ROOTHOSTCERT");
   } else {
      if (getenv("ROOTETCDIR"))
         hcconf.insert(0,getenv("ROOTETCDIR"));
      else
         hcconf.insert(0,"/etc/root");
   }
   // std::string::insert is buggy on some compilers (eg gcc 2.96):
   // new length correct but data not always null terminated
   hcconf[hcconf.length()] = 0;

   std::string fns[4];
   FILE *fconf = 0;
   if ((fconf = fopen(hcconf.data(), "r"))) {
      char line[kMAXPATHLEN];
      if (gDebug > 2)
         ErrorInfo("GlbsToolCheckCert: reading file %s", hcconf.c_str());

      // Parse the lines: keep the last non empty one
      while (fgets(line, sizeof(line), fconf)) {
         if (line[0] == '#') continue;     // skip comment lines
         if (strlen(line) == 0) continue;  // skip empty lines
         if (line[strlen(line)-1] == '\n')
            line[strlen(line)-1] = '\0';
         int i = 0;
         char *p0 = &line[0];
         char *p1 = p0;
         while ((p1 = strchr(p0+1, ' '))) {
            *p1 = '\0';
            fns[i++] = p0;
            p1++;
            while (p1[0] == ' ')
               p1++;
            p0 = p1;
         }
         // Do not miss the last one
         if (i < 4) fns[i++] = p0;
         // Fill the remaining ones with the default symbol
         while (i < 4)
            fns[i++] = "*";
      }
      fclose(fconf);
      if (gDebug > 2)
          ErrorInfo("GlbsToolCheckCert: from file: {%s,%s,%s,%s}",
                    fns[0].c_str(), fns[1].c_str(), fns[2].c_str(), fns[3].c_str());
   } else {
      // Fill with the default symbol
      int i = 0;
      while (i < 4)
         fns[i++] = "*";
   }

   // Check one by one now, usign the default, if needed
   int rdir = 0, rcer = 0;

   // Certificate directory
   std::string dir = fns[0];
   if (access(dir.c_str(), R_OK)) {
      // Try default
      dir = "/etc/grid-security/certificates";
      if (access(dir.c_str(), R_OK)) {
         // Failure
         if (gDebug > 0)
            ErrorInfo("GlbsToolCheckCert: no readable certificate dir found");
         rdir = 1;
      }
   }
   if (!rdir)
      if ((rdir = setenv("X509_CERT_DIR", dir.c_str(), 1)))
         ErrorInfo("GlbsToolCheckCert: unable to set X509_CERT_DIR ");

   // Gridmap file
   std::string map = fns[3];
   if (access(map.c_str(), R_OK)) {
      // Try default
      map = "/etc/grid-security/grid-mapfile";
      if (access(map.c_str(), R_OK)) {
         // Failure
         if (gDebug > 0)
            ErrorInfo("GlbsToolCheckCert: no readable grid-mapfile found");
         rdir = 1;
      }
   }
   if (!rdir)
      if ((rdir = setenv("GRIDMAP", map.c_str(), 1)))
         ErrorInfo("GlbsToolCheckCert: unable to set GRIDMAP ");

   // Certificate and key
   std::string cert = fns[1];
   std::string key = fns[2];
   if (access(cert.c_str(), R_OK) || access(key.c_str(), R_OK)) {
      // Try first default
      cert = "/etc/grid-security/root/rootcert.pem";
      key = "/etc/grid-security/root/rootkey.pem";
      if (access(cert.c_str(), R_OK) || access(key.c_str(), R_OK)) {
         // Try second default
         cert = "/etc/grid-security/hostcert.pem";
         key = "/etc/grid-security/hostkey.pem";
         if (access(cert.c_str(), R_OK) || access(key.c_str(), R_OK)) {
            // Failure
            if (gDebug > 0)
               ErrorInfo("GlbsToolCheckCert: no readable {cert, key} pair found");
            rcer = 1;
         }
      }
   }
   if (!rcer) {
      // Set envs
      if ((rcer = setenv("X509_USER_CERT", cert.c_str(), 1)))
         ErrorInfo("GlbsToolCheckCert: unable to set X509_HOST_CERT ");
      if ((rcer = setenv("X509_USER_KEY", key.c_str(), 1)))
         ErrorInfo("GlbsToolCheckCert: unable to set X509_HOST_KEY ");

      // Open the certificate to get the subject name
      FILE *fcert = fopen(cert.c_str(), "r");
      X509 *xcert = 0;
      if (!PEM_read_X509(fcert, &xcert, 0, 0)) {
         ErrorInfo("GlbsToolCheckCert: unable to load certificate from %s",
                   cert.c_str());
         rcer = 1;
      } else {
         *subjname = X509_NAME_oneline(X509_get_subject_name(xcert), 0, 0);
         if (gDebug > 2)
            ErrorInfo("GlbsToolCheckCert: subject: %s", *subjname);
      }
   }

   // Notify (on success)
   if (!rdir && !rcer)
      if (gDebug > 2)
         ErrorInfo("GlbsToolCheckCert: using: {%s,%s,%s,%s}",
                    dir.c_str(), cert.c_str(), key.c_str(), map.c_str());

   // Done
   return ((rdir != 0 || rcer != 0) ? 1 : 0);
}

//______________________________________________________________________________
int GlbsToolCheckContext(int shmId)
{
   // Checks validity of security context exported in shared memory
   // segment shmId. Returns 1 if valid, 0 othrwise.

   int retval = 0;
   OM_uint32 majstat = 0;
   OM_uint32 minstat = 0;
   gss_ctx_id_t context_handle = GSS_C_NO_CONTEXT;
   OM_uint32 gssRetFlags = 0;
   OM_uint32 glbContLifeTime = 0;
   int dum1, dum2;
   gss_OID mechType;
   gss_name_t *targname = 0, *name = 0;

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckContext: checking contetx in shm : %d",
                shmId);

   // retrieve the context from shared memory ...
   gss_buffer_t databuf = (gss_buffer_t) shmat(shmId, 0, 0);
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolCheckContext: retrieving info from shared memory: %d",
           shmId);

   // Import the security context ...
   gss_buffer_t secContExp =
       (gss_buffer_t) new char[sizeof(gss_buffer_desc) + databuf->length];
   secContExp->length = databuf->length;
   secContExp->value =
       (void *) ((char *) secContExp + sizeof(size_t) + sizeof(void *));
   void *dbufval =
       (void *) ((char *) databuf + sizeof(size_t) + sizeof(void *));
   memmove(secContExp->value, dbufval, secContExp->length);
   if ((majstat =
        gss_import_sec_context(&minstat, secContExp,
                               &context_handle)) != GSS_S_COMPLETE) {
      GlbsToolError("GlbsToolCheckContext: gss_import_sec_context",
                    majstat, minstat, 0);
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolCheckContext: GlbsTool Sec Context successfully imported (0x%x)",
           context_handle);

   delete[]secContExp;

   // Detach from shared memory segment
   int rc = shmdt((const void *) databuf);
   if (rc != 0) {
      ErrorInfo
          ("GlbsToolCheckContext: unable to detach from shared memory segment %d (rc=%d)",
           shmId, rc);
   }
   // Check validity of the retrieved context ...
   if (context_handle != 0 && context_handle != GSS_C_NO_CONTEXT) {
      if ((majstat =
           gss_inquire_context(&minstat, context_handle, name, targname,
                               &glbContLifeTime, &mechType, &gssRetFlags,
                               &dum1, &dum2)) != GSS_S_COMPLETE) {
         GlbsToolError("GlbsToolCheckContext: gss_inquire_context",
                       majstat, minstat, 0);
         // mark segment for distruction
         struct shmid_ds shm_ds;
         if (!shmctl(shmId, IPC_RMID, &shm_ds))
            ErrorInfo
                ("GlbsToolCheckContext: unable to mark shared memory segment %d for desctruction",
                 shmId);
      } else {
         if (gDebug > 2)
            ErrorInfo
                ("GlbsToolCheckContext: found valid context in shm %d",
                 shmId);
         retval = 1;
      }
   }

   return retval;
}

//______________________________________________________________________________
int GlbsToolStoreContext(gss_ctx_id_t context_handle, char *user)
{
   // Exports a security context for later use and stores in a shared memory
   // segments. On success returns Id of the allocated shared memory segment,
   // 0 otherwise.

   OM_uint32 majstat;
   OM_uint32 minstat;
   key_t shm_key = IPC_PRIVATE;
   int shm_flg = 0777;
   struct shmid_ds shm_ds;

   if (gDebug > 2)
      ErrorInfo("GlbsToolStoreContext: Enter");

   // First we have to prepare the export of the security context
   gss_buffer_t secContExp = new gss_buffer_desc;
   if ((majstat =
        gss_export_sec_context(&minstat, &context_handle,
                               secContExp)) != GSS_S_COMPLETE) {
      GlbsToolError("GlbsToolStoreContext: gss_export_sec_context",
                    majstat, minstat, 0);
      gss_release_buffer(&minstat,secContExp);
      delete secContExp;
      return 0;
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreContext: security context prepared for export");

   // This is the size of the needed shared memory segment
   int shm_size = sizeof(gss_buffer_desc) + secContExp->length;
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreContext: needed shared memory segment sizes: %d",
           shm_size);

   // Here we allocate the shared memory segment
   int shmId = shmget(shm_key, shm_size, shm_flg);
   if (shmId < 0) {
      ErrorInfo
          ("GlbsToolStoreContext: while allocating shared memory segment (rc=%d)",
           shmId);
      gss_release_buffer(&minstat,secContExp);
      delete secContExp;
      return 0;
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreContext: shared memory segment allocated (id=%d)",
           shmId);

   // Attach segment to address
   gss_buffer_t databuf = (gss_buffer_t) shmat(shmId, 0, 0);
   if (databuf == (gss_buffer_t)(-1)) {
      ErrorInfo("GlbsToolStoreContext: while attaching to shared memory"
                " segment (rc=%d)", shmId);
      gss_release_buffer(&minstat,secContExp);
      shmctl(shmId, IPC_RMID, &shm_ds);
      return 0;
   }
   databuf->length = secContExp->length;
   databuf->value =
       (void *) ((char *) databuf + sizeof(size_t) + sizeof(void *));
   memmove(databuf->value, secContExp->value, secContExp->length);

   // Now we can detach from the shared memory segment ...
   // and release memory we don't anylonger
   int rc = 0;
   if ((rc = shmdt((const void *) databuf)) != 0) {
      ErrorInfo
          ("GlbsToolStoreContext: unable to detach from shared memory segment (rc=%d)",
           rc);
   }

   // Release buffer used for export
   if ((majstat = gss_release_buffer(&minstat, secContExp)) != GSS_S_COMPLETE)
      GlbsToolError("GlbsToolStoreContext: gss_release_buffer",
                    majstat, minstat, 0);
   delete secContExp;

   // We need to change the ownership of the shared memory segment used
   // for credential export to allow proofserv to destroy it
   if (shmctl(shmId, IPC_STAT, &shm_ds) == -1) {
      ErrorInfo
          ("GlbsToolStoreContext: can't get info about shared memory segment %d",
           shmId);
      shmctl(shmId, IPC_RMID, &shm_ds);
      return 0;
   }
   // Get info about user logging in
   struct passwd *pw = getpwnam(user);

   if (pw) {
      // Give use ownership of the shared memory segment ...
      shm_ds.shm_perm.uid = pw->pw_uid;
      shm_ds.shm_perm.gid = pw->pw_gid;
      if (shmctl(shmId, IPC_SET, &shm_ds) == -1) {
         ErrorInfo
             ("GlbsToolStoreContext: can't change ownership of shared memory segment %d",
             shmId);
         shmctl(shmId, IPC_RMID, &shm_ds);
         return 0;
      }
   } else {
      ErrorInfo
          ("GlbsToolStoreContext: user %s unknown to the system!", user);
   }
   // return shmId to rootd
   return shmId;
}

//______________________________________________________________________________
int GlbsToolStoreToShm(gss_buffer_t buffer, int *shmId)
{
   // Creates a shm and stores buffer in it.
   // Returns 0 on success (shm id in shmId), >0 otherwise.

   key_t shm_key = IPC_PRIVATE;
   int shm_flg = 0777;
   struct shmid_ds shm_ds;

   if (gDebug > 2)
      ErrorInfo("GlbsToolStoreToShm: Enter: shmId: %d", *shmId);

   // This is the size of the needed shared memory segment
   int shm_size = sizeof(gss_buffer_desc) + buffer->length;
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreToShm: needed shared memory segment sizes: %d",
           shm_size);

   // Here we allocate the shared memory segment
   int lshmId = shmget(shm_key, shm_size, shm_flg);
   if (lshmId < 0) {
      ErrorInfo
          ("GlbsToolStoreToShm: while allocating shared memory segment (rc=%d)",
           lshmId);
      return 1;
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreToShm: shared memory segment allocated (id=%d)",
           lshmId);

   *shmId = lshmId;

   // Attach segment to address
   gss_buffer_t databuf = (gss_buffer_t) shmat(lshmId, 0, 0);
   if (databuf == (gss_buffer_t)(-1)) {
      ErrorInfo("GlbsToolStoreToShm: while attaching to shared memory"
                " segment (rc=%d)", lshmId);
      shmctl(lshmId, IPC_RMID, &shm_ds);
      return 2;
   }
   databuf->length = buffer->length;
   databuf->value =
       (void *) ((char *) databuf + sizeof(size_t) + sizeof(void *));
   memmove(databuf->value, buffer->value, buffer->length);

   // Now we can detach from the shared memory segment ... and release memory we don't anylonger
   int rc = 0;
   if ((rc = shmdt((const void *) databuf)) != 0) {
      ErrorInfo
          ("GlbsToolStoreToShm: unable to detach from shared memory segment (rc=%d)",
           rc);
   }
   return 0;
}

//______________________________________________________________________________
char *GlbsToolExpand(char *file)
{
   // Test is expansion is needed and return full path file name
   // (expanded with $HOME).
   // Returned string must be 'delete[] ed' by the caller.

   char *fret = 0;

   if (file) {

      if (file[0] == '/' || (!getenv("HOME"))) {
         fret = new char[strlen(file) + 1];
         strncpy(fret, file, strlen(file));
      } else {
         fret = new char[strlen(file) + strlen(getenv("HOME")) + 2];
         if (file[0] == '~') {
            SPrintf(fret, strlen(file) + strlen(getenv("HOME")) + 2, "%s/%s", getenv("HOME"), file + 1);
         } else {
            SPrintf(fret, strlen(file) + strlen(getenv("HOME")) + 2, "%s/%s", getenv("HOME"), file);
         }
      }

   }
   return fret;
}

//_________________________________________________________________________
int GlbsToolCheckProxy(char **subjname)
{
   // Check existence and validity of user proxy
   // Return 0 on success, 1 otherwise

   // Check if there is a proxy file associated with this user
   char pxy[256];
   SPrintf(pxy, 256, "/tmp/x509up_u%d", getuid());

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckProxy: testing proxy file: %s", pxy);

   if (gDebug > 3)
      ErrorInfo("GlbsToolCheckProxy: uid:%d euid:%d gid:%d egid:%d",
                getuid(), geteuid(), getgid(), getegid());

   if (!access(pxy, R_OK)) {

      if (setenv("X509_USER_PROXY", pxy, 1))
         ErrorInfo("GlbsToolCheckProxy: unable to set X509_USER_PROXY ");

#ifdef R__GLBS22
      globus_gsi_cred_handle_t pxycreds = 0;

      // Init proxy cred handle
      if (globus_gsi_cred_handle_init(&pxycreds, 0)
          != GLOBUS_SUCCESS) {
          ErrorInfo("GlbsToolCheckProxy: %s",
                    "couldn't initialize proxy credential handle");
          return 1;
      }

      // Read proxies
      if (globus_gsi_cred_read_proxy(pxycreds, pxy)
          != GLOBUS_SUCCESS) {
          ErrorInfo("GlbsToolCheckProxy: %s %s",
                    "couldn't read proxy from:", pxy);
          globus_gsi_cred_handle_destroy(pxycreds);
          return 1;
      }

      // Get time left to expiration (in seconds) */
      time_t lifetime;
      if( globus_gsi_cred_get_lifetime(pxycreds,&lifetime)
          != GLOBUS_SUCCESS) {
          ErrorInfo("GlbsToolCheckProxy: %s %s",
                    "couldn't get proxy remaining lifetime");
          globus_gsi_cred_handle_destroy(pxycreds);
          return 1;
      }
      globus_gsi_cred_handle_destroy(pxycreds);

      if (lifetime > 0) {

         if (lifetime < 3600)
            ErrorInfo("GlbsToolCheckProxy: WARNING: %s",
                      "proxy will soon expire (less than %d s)",
                       lifetime);

        // Get issuer to be sent back to the client
        X509 *xcert = 0;
        FILE *fcert = fopen(pxy, "r");
        if (fcert == 0 || !PEM_read_X509(fcert, &xcert, 0, 0)) {
           ErrorInfo("GlbsToolCheckProxy: unable to load user proxy certificate ");
           return 1;
        }
        fclose(fcert);
        *subjname = X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0);
        if (gDebug > 3)
           ErrorInfo("GlbsToolCheckProxy: %s %s",
                   "Proxy Issuer:", *subjname);

      } else {
         ErrorInfo("GlbsToolCheckProxy: ERROR: %s",
                     "proxy are invalid (expired)");
         return 1;
      }

#else
      // Old version: completely different ...
      char *                proxy_type;
      proxy_cred_desc *     pcd = 0;
      time_t                time_after;
      time_t                time_now;
      time_t                time_diff;
      ASN1_UTCTIME *        asn1_time = 0;

      // Init credential descriptor
      pcd = proxy_cred_desc_new();

      // Load user proxy certificate
      pcd->type=CRED_TYPE_PROXY;
      if (proxy_load_user_cert(pcd, proxy_file, 0, 0)) {
         ErrorInfo("GlbsToolCheckProxy: ERROR: %s (%s)",
                     "cannot load proxy certificate",proxy_file);
         return 1;
      }

      // Load user public key
      if ((pcd->upkey = X509_get_pubkey(pcd->ucert)) == 0) {
         ErrorInfo("GlbsToolCheckProxy: ERROR: %s",
                     "cannot get public key");
         return 1;
      }

      // validity: set time_diff to time to expiration (in seconds)
      asn1_time = ASN1_UTCTIME_new();
      X509_gmtime_adj(asn1_time,0);
      time_now = ASN1_UTCTIME_mktime(asn1_time);
      time_after = ASN1_UTCTIME_mktime(X509_get_notAfter(pcd->ucert));
      time_diff = time_after - time_now ;

      if (time_diff > 0) {

         if (time_diff < 3600)
            ErrorInfo("GlbsToolCheckProxy: WARNING: %s",
                      "proxy will soon expire (less than %d s)",
                       time_diff);

        // Get issuer to be sent back to the client
        *subjname = X509_NAME_oneline(X509_get_issuer_name(pcd->ucert), 0, 0);
        if (gDebug > 3)
           ErrorInfo("GlbsToolCheckProxy: %s %s",
                   "Proxy Issuer:", *subjname);

      } else {
         ErrorInfo("GlbsToolCheckProxy: ERROR: %s",
                     "proxy are invalid (expired)");
         return 1;
      }

#endif

   } else {
      // Proxy file not existing or not readable
      ErrorInfo("GlbsToolCheckProxy: Proxy file not existing or"
                     "not readable");
      return 1;
   }

   return 0;
}

} // namespace ROOT
