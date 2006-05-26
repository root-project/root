// @(#)root/rpdutils:$Name:  $:$Id: globus.cxx,v 1.12 2005/09/21 17:23:36 brun Exp $
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
void GlbsToolError(char *mess, int majs, int mins, int toks)
{
   // Handle error ...

   char *GlbErr;

   if (!globus_gss_assist_display_status_str
      (&GlbErr, mess, majs, mins, toks)) {
   } else {
      char GlbErr[kMAXPATHLEN];
      SPrintf(GlbErr,kMAXPATHLEN,"%s: error messaged not resolved ", mess);
   }
   NetSend(kErrFatal, kROOTD_ERR);
   ErrorInfo("Error: %s (majst=%d,minst=%d,tokst:%d)", GlbErr, majs, mins,
             toks);

   delete [] GlbErr;
}

//_________________________________________________________________________
int GlbsToolCheckCert(char *ClientIssuerName, char **SubjName)
{
   // Load information about available certificates from user specified
   // sources or from defaults.
   // Returns number of potentially valid dir/cert/key triplets found.

   int retval = 1;
   std::string HostCertConf = "/hostcert.conf";
   char *certdir_default  = "/etc/grid-security/certificates";
   char *hostcert_default[2] = {"/etc/grid-security/root/rootcert.pem",
                                "/etc/grid-security/hostcert.pem"};
   char *hostkey_default[2] = {"/etc/grid-security/root/rootkey.pem",
                               "/etc/grid-security/hostkey.pem"};
   char *gridmap_default  = "/etc/grid-security/grid-mapfile";
   char dir_def[kMAXPATHLEN] = { 0 }, cert_def[kMAXPATHLEN] = { 0 },
        key_def[kMAXPATHLEN] = { 0 }, map_def[kMAXPATHLEN]  = { 0 };
   char *dir_tmp = 0, *cert_tmp = 0, *key_tmp = 0, *map_tmp = 0;
   bool CertFound = 0;
   X509 *xcert = 0;
   FILE *fcert = 0;
   char *issuer_name = 0;
   int id = 0;

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckCert: enter: %s", ClientIssuerName);

   // Check if a non-standard file has been requested
   if (getenv("ROOTHOSTCERT")) {
      HostCertConf = getenv("ROOTHOSTCERT");
   } else {
      if (getenv("ROOTETCDIR"))
         HostCertConf.insert(0,getenv("ROOTETCDIR"));
      else
         HostCertConf.insert(0,"/etc/root");
   }

   if (HostCertConf.length()) {
      // string::insert is buggy on some compilers (eg gcc 2.96):
      // new length correct but data not always null terminated
      HostCertConf[HostCertConf.length()] = 0;

      // The user/administrator provided a file ... check if it exists
      // and can be read
      FILE *fconf = 0;
      if (!access(HostCertConf.data(), R_OK) &&
          (fconf = fopen(HostCertConf.data(), "r")) != 0) {
         char line[kMAXPATHLEN];
         if (gDebug > 2)
            ErrorInfo("GlbsToolCheckCert: reading host cert file %s",
                      HostCertConf.data());

         // ... let's see what it's inside
         while (fgets(line, sizeof(line), fconf)) {
            if (line[0] == '#')
               continue;        // skip comment lines
            int nw = sscanf(line, "%s %s %s %s", dir_def, cert_def,
                                                 key_def, map_def);

            if (nw < 1)
               continue;        // skip empty lines

            // allow for incomplete lines ... completion with defaults ...
            // check also if default entries have been given ('*')
            if (nw == 1) {
               if (dir_def[0] == '*')
                  strcpy(dir_def, certdir_default);
               strcpy(cert_def, hostcert_default[0]);
               strcpy(key_def, hostkey_default[0]);
               strcpy(map_def, gridmap_default);
            } else if (nw == 2) {
               if (dir_def[0] == '*')
                  strcpy(dir_def, certdir_default);
               if (cert_def[0] == '*')
                  strcpy(cert_def, hostcert_default[0]);
               strcpy(key_def, hostkey_default[0]);
               strcpy(map_def, gridmap_default);
            } else if (nw == 3) {
               if (dir_def[0] == '*')
                  strcpy(dir_def, certdir_default);
               if (cert_def[0] == '*')
                  strcpy(cert_def, hostcert_default[0]);
               if (key_def[0] == '*')
                  strcpy(key_def, hostkey_default[0]);
               strcpy(map_def, gridmap_default);
            } else if (nw == 4) {
               if (dir_def[0] == '*')
                  strcpy(dir_def, certdir_default);
               if (cert_def[0] == '*')
                  strcpy(cert_def, hostcert_default[0]);
               if (key_def[0] == '*')
                  strcpy(key_def, hostkey_default[0]);
               if (map_def[0] == '*')
                  strcpy(map_def, gridmap_default);
            }

            // Expand for test if needed
            dir_tmp  = GlbsToolExpand(dir_def);
            cert_tmp = GlbsToolExpand(cert_def);
            key_tmp  = GlbsToolExpand(key_def);
            map_tmp  = GlbsToolExpand(map_def);
            if (gDebug > 2)
               ErrorInfo
                   ("GlbsToolCheckCert: testing host cert file map %s %s %s %s",
                    dir_tmp, cert_tmp, key_tmp, map_tmp);

            // check that the files exist and can be read
            if (!access(dir_tmp, R_OK)) {
               if (!access(cert_tmp, R_OK)) {
                  if (!access(key_tmp, R_OK)) {
                     /// Load certificate
                     fcert = fopen(cert_tmp, "r");
                     if (!PEM_read_X509(fcert, &xcert, 0, 0)) {
                        ErrorInfo("GlbsToolCheckCert: unable to load host"
                                  " certificate (%s)", cert_tmp);
                        goto goout;
                     }
                     // Get the issuer name
                     issuer_name =
                         X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0);
                     if (strstr(issuer_name, ClientIssuerName) != 0) {
                        CertFound = 1;
                        if (gDebug > 2)
                           ErrorInfo("GlbsToolCheckCert: Issuer Subject:"
                                     " %s matches",issuer_name);
                        fclose(fconf);
                        goto found;
                     }
                  } else {
                     if (gDebug > 2)
                        ErrorInfo("GlbsToolCheckCert: key file not existing"
                                  " or not readable (%s)", key_tmp);
                  }
               } else {
                  if (gDebug > 2)
                     ErrorInfo("GlbsToolCheckCert: cert file not existing"
                               " or not readable (%s)", cert_tmp);
               }
            } else {
               if (gDebug > 2)
                  ErrorInfo("GlbsToolCheckCert: cert directory not existing"
                            " or not readable (%s)",dir_tmp);
            }
            if (gDebug > 2)
               ErrorInfo("GlbsToolCheckCert: read cert key map files:"
                         " %s %s %s %s",dir_tmp, cert_tmp, key_tmp, map_tmp);
            // Cleanup memory
            if (dir_tmp)
               delete[]dir_tmp;
            if (cert_tmp)
               delete[]cert_tmp;
            if (key_tmp)
               delete[]key_tmp;
            if (map_tmp)
               delete[]map_tmp;
         }
         fclose(fconf);

      } else {
         if (gDebug > 2)
            ErrorInfo("GlbsToolCheckCert: host cert conf not existing or"
                      " not readable (%s)",HostCertConf.data());
      }
   } else if (gDebug > 2)
      ErrorInfo("GlbsToolCheckCert: HOSTCERTCONF undefined");
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolCheckCert: Try to use env definitions or defaults ...");

   // We have not found a good one: try with these envs definitions
   // or the defaults ...
   if (getenv("X509_CERT_DIR") != 0) {
      strcpy(dir_def, getenv("X509_CERT_DIR"));
   } else
      strcpy(dir_def, certdir_default);
   if (getenv("GRIDMAP") != 0) {
      strcpy(map_def, getenv("GRIDMAP"));
   } else
      strcpy(map_def, gridmap_default);
   // Expand for test if needed
   dir_tmp  = GlbsToolExpand(dir_def);
   map_tmp  = GlbsToolExpand(map_def);

   // First the ROOT specific, then the host one
   for ( id = 0; id < 2; id++) {
      // Load certificate / key names
      if (getenv("X509_USER_CERT") != 0) {
         strcpy(cert_def, getenv("X509_USER_CERT"));
      } else
         strcpy(cert_def, hostcert_default[id]);
      if (getenv("X509_USER_KEY") != 0) {
         strcpy(key_def, getenv("X509_USER_KEY"));
      } else
         strcpy(key_def, hostkey_default[id]);

      // Expand for test if needed
      cert_tmp = GlbsToolExpand(cert_def);
      key_tmp  = GlbsToolExpand(key_def);

      if (!access(dir_tmp, R_OK)) {
         if (!access(cert_tmp, R_OK)) {
            if (!access(key_tmp, R_OK)) {
               // Load certificate
               fcert = fopen(cert_tmp, "r");
               if (!PEM_read_X509(fcert, &xcert, 0, 0)) {
                  ErrorInfo("GlbsToolCheckCert: unable to load host"
                            " certificate (%s)",cert_tmp);
                  goto goout;
               }
               // Get the issuer name
               issuer_name =
                   X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0);
               if (strstr(issuer_name, ClientIssuerName) != 0) {
                  CertFound = 1;
                  if (gDebug > 2)
                     ErrorInfo
                         ("GlbsToolCheckCert: Issuer Subject: %s matches",
                          issuer_name);
                  goto found;
               }
            } else {
               ErrorInfo("GlbsToolCheckCert: default hostkey file not"
                         " existing or not readable (%s)", key_tmp);
               goto goout;
            }
         } else {
            ErrorInfo("GlbsToolCheckCert: default hostcert file not"
                      " existing or not readable (%s)",cert_tmp);
            goto goout;
         }
      } else {
         ErrorInfo("GlbsToolCheckCert: default cert directory not"
                   " existing or not readable (%s)",dir_tmp);
         goto goout;
      }
      // Release memory before going to next set
      if (cert_tmp)
         delete[]cert_tmp;
      if (key_tmp)
         delete[]key_tmp;
   }

 goout:
   if (dir_tmp)
      delete[]dir_tmp;
   if (cert_tmp)
      delete[]cert_tmp;
   if (key_tmp)
      delete[]key_tmp;
   if (map_tmp)
      delete[]map_tmp;
   return 1;

 found:
   if (dir_tmp)
      delete[]dir_tmp;
   if (cert_tmp)
      delete[]cert_tmp;
   if (key_tmp)
      delete[]key_tmp;
   if (map_tmp)
      delete[]map_tmp;

   if (CertFound) {
      // Get the subject name
      char *subject_name =
          X509_NAME_oneline(X509_get_subject_name(xcert), 0, 0);
      if (gDebug > 2) {
         ErrorInfo("GlbsToolCheckCert: issuer: %s", issuer_name);
         ErrorInfo("GlbsToolCheckCert: subject: %s", subject_name);
      }
      // Send it to the client ...
      *SubjName = strdup(subject_name);
      // Mission ok ..
      retval = 0;
      // free resources
      free(issuer_name);
      free(subject_name);
      // We have found a valid one ...
      fclose(fcert);


       // We set the relevant environment variables ...
       if (setenv("X509_CERT_DIR", dir_def, 1)) {
          ErrorInfo("GlbsToolCheckCert: unable to set X509_CERT_DIR ");
          return 1;
       }
       if (setenv("X509_USER_CERT", cert_def, 1)) {
          ErrorInfo("GlbsToolCheckCert: unable to set X509_USER_CERT ");
          return 1;
       }
       if (setenv("X509_USER_KEY", key_def, 1)) {
          ErrorInfo("GlbsToolCheckCert: unable to set X509_USER_KEY ");
          return 1;
       }
       if (setenv("GRIDMAP", map_def, 1)) {
          ErrorInfo("GlbsToolCheckCert: unable to set GRIDMAP ");
       }
   }
   return retval;
}

//______________________________________________________________________________
int GlbsToolCheckContext(int ShmId)
{
   // Checks validity of security context exported in shared memory
   // segment SHmId. Returns 1 if valid, 0 othrwise.

   int retval = 0;
   OM_uint32 MajStat = 0;
   OM_uint32 MinStat = 0;
   gss_ctx_id_t context_handle = GSS_C_NO_CONTEXT;
   OM_uint32 GssRetFlags = 0;
   OM_uint32 GlbContLifeTime = 0;
   int Dum1, Dum2;
   gss_OID MechType;
   gss_name_t *TargName = 0, *Name = 0;

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckContext: checking contetx in shm : %d",
                ShmId);

   // retrieve the context from shared memory ...
   gss_buffer_t databuf = (gss_buffer_t) shmat(ShmId, 0, 0);
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolCheckContext: retrieving info from shared memory: %d",
           ShmId);

   // Import the security context ...
   gss_buffer_t SecContExp =
       (gss_buffer_t) new char[sizeof(gss_buffer_desc) + databuf->length];
   SecContExp->length = databuf->length;
   SecContExp->value =
       (void *) ((char *) SecContExp + sizeof(size_t) + sizeof(void *));
   void *dbufval =
       (void *) ((char *) databuf + sizeof(size_t) + sizeof(void *));
   memmove(SecContExp->value, dbufval, SecContExp->length);
   if ((MajStat =
        gss_import_sec_context(&MinStat, SecContExp,
                               &context_handle)) != GSS_S_COMPLETE) {
      GlbsToolError("GlbsToolCheckContext: gss_import_sec_context",
                    MajStat, MinStat, 0);
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolCheckContext: GlbsTool Sec Context successfully imported (0x%x)",
           context_handle);

   delete[]SecContExp;

   // Detach from shared memory segment
   int rc = shmdt((const void *) databuf);
   if (rc != 0) {
      ErrorInfo
          ("GlbsToolCheckContext: unable to detach from shared memory segment %d (rc=%d)",
           ShmId, rc);
   }
   // Check validity of the retrieved context ...
   if (context_handle != 0 && context_handle != GSS_C_NO_CONTEXT) {
      if ((MajStat =
           gss_inquire_context(&MinStat, context_handle, Name, TargName,
                               &GlbContLifeTime, &MechType, &GssRetFlags,
                               &Dum1, &Dum2)) != GSS_S_COMPLETE) {
         GlbsToolError("GlbsToolCheckContext: gss_inquire_context",
                       MajStat, MinStat, 0);
         // mark segment for distruction
         struct shmid_ds shm_ds;
         if (!shmctl(ShmId, IPC_RMID, &shm_ds))
            ErrorInfo
                ("GlbsToolCheckContext: unable to mark shared memory segment %d for desctruction",
                 ShmId);
      } else {
         if (gDebug > 2)
            ErrorInfo
                ("GlbsToolCheckContext: found valid context in shm %d",
                 ShmId);
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

   OM_uint32 MajStat;
   OM_uint32 MinStat;
   key_t shm_key = IPC_PRIVATE;
   int shm_flg = 0777;
   struct shmid_ds shm_ds;

   if (gDebug > 2)
      ErrorInfo("GlbsToolStoreContext: Enter");

   // First we have to prepare the export of the security context
   gss_buffer_t SecContExp = new gss_buffer_desc;
   if ((MajStat =
        gss_export_sec_context(&MinStat, &context_handle,
                               SecContExp)) != GSS_S_COMPLETE) {
      GlbsToolError("GlbsToolStoreContext: gss_export_sec_context",
                    MajStat, MinStat, 0);
      gss_release_buffer(&MinStat,SecContExp);
      delete SecContExp;
      return 0;
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreContext: security context prepared for export");

   // This is the size of the needed shared memory segment
   int shm_size = sizeof(gss_buffer_desc) + SecContExp->length;
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreContext: needed shared memory segment sizes: %d",
           shm_size);

   // Here we allocate the shared memory segment
   int ShmId = shmget(shm_key, shm_size, shm_flg);
   if (ShmId < 0) {
      ErrorInfo
          ("GlbsToolStoreContext: while allocating shared memory segment (rc=%d)",
           ShmId);
      gss_release_buffer(&MinStat,SecContExp);
      delete SecContExp;
      return 0;
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreContext: shared memory segment allocated (id=%d)",
           ShmId);

   // Attach segment to address
   gss_buffer_t databuf = (gss_buffer_t) shmat(ShmId, 0, 0);
   if (databuf == (gss_buffer_t)(-1)) {
      ErrorInfo
          ("GlbsToolStoreContext: while attaching to shared memory segment (rc=%d)",
           (int) databuf);
      gss_release_buffer(&MinStat,SecContExp);
      shmctl(ShmId, IPC_RMID, &shm_ds);
      return 0;
   }
   databuf->length = SecContExp->length;
   databuf->value =
       (void *) ((char *) databuf + sizeof(size_t) + sizeof(void *));
   memmove(databuf->value, SecContExp->value, SecContExp->length);

   // Now we can detach from the shared memory segment ...
   // and release memory we don't anylonger
   int rc = 0;
   if ((rc = shmdt((const void *) databuf)) != 0) {
      ErrorInfo
          ("GlbsToolStoreContext: unable to detach from shared memory segment (rc=%d)",
           rc);
   }

   // Release buffer used for export
   if ((MajStat = gss_release_buffer(&MinStat, SecContExp)) != GSS_S_COMPLETE)
      GlbsToolError("GlbsToolStoreContext: gss_release_buffer",
                    MajStat, MinStat, 0);
   delete SecContExp;

   // We need to change the ownership of the shared memory segment used
   // for credential export to allow proofserv to destroy it
   if (shmctl(ShmId, IPC_STAT, &shm_ds) == -1) {
      ErrorInfo
          ("GlbsToolStoreContext: can't get info about shared memory segment %d",
           ShmId);
      shmctl(ShmId, IPC_RMID, &shm_ds);
      return 0;
   }
   // Get info about user logging in
   struct passwd *pw = getpwnam(user);

   // Give use ownership of the shared memory segment ...
   shm_ds.shm_perm.uid = pw->pw_uid;
   shm_ds.shm_perm.gid = pw->pw_gid;
   if (shmctl(ShmId, IPC_SET, &shm_ds) == -1) {
      ErrorInfo
          ("GlbsToolStoreContext: can't change ownership of shared memory segment %d",
           ShmId);
      shmctl(ShmId, IPC_RMID, &shm_ds);
      return 0;
   }
   // return shmid to rootd
   return ShmId;
}

//______________________________________________________________________________
int GlbsToolStoreToShm(gss_buffer_t buffer, int *ShmId)
{
   // Creates a shm and stores buffer in it.
   // Returns 0 on success (shm id in ShmId), >0 otherwise.

   key_t shm_key = IPC_PRIVATE;
   int shm_flg = 0777;
   struct shmid_ds shm_ds;

   if (gDebug > 2)
      ErrorInfo("GlbsToolStoreToShm: Enter: ShmId: %d", *ShmId);

   // This is the size of the needed shared memory segment
   int shm_size = sizeof(gss_buffer_desc) + buffer->length;
   if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreToShm: needed shared memory segment sizes: %d",
           shm_size);

   // Here we allocate the shared memory segment
   int lShmId = shmget(shm_key, shm_size, shm_flg);
   if (lShmId < 0) {
      ErrorInfo
          ("GlbsToolStoreToShm: while allocating shared memory segment (rc=%d)",
           lShmId);
      return 1;
   } else if (gDebug > 2)
      ErrorInfo
          ("GlbsToolStoreToShm: shared memory segment allocated (id=%d)",
           lShmId);

   *ShmId = lShmId;

   // Attach segment to address
   gss_buffer_t databuf = (gss_buffer_t) shmat(lShmId, 0, 0);
   if (databuf == (gss_buffer_t)(-1)) {
      ErrorInfo
          ("GlbsToolStoreToShm: while attaching to shared memory segment (rc=%d)",
           (int) databuf);
      shmctl(lShmId, IPC_RMID, &shm_ds);
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
         strcpy(fret, file);
      } else {
         fret = new char[strlen(file) + strlen(getenv("HOME")) + 2];
         if (file[0] == '~') {
            sprintf(fret, "%s/%s", getenv("HOME"), file + 1);
         } else {
            sprintf(fret, "%s/%s", getenv("HOME"), file);
         }
      }

   }
   return fret;
}

//_________________________________________________________________________
int GlbsToolCheckProxy(char *ClientIssuerName, char **SubjName)
{
   // Check validity of user proxy
   // Set environments X509_CERT_DIR and GRIDMAP following
   // user specifications
   // Called when running as non-root

   std::string HostCertConf = "/hostcert.conf";
   char *certdir_default  = "/etc/grid-security/certificates";
   char *gridmap_default  = "/etc/grid-security/grid-mapfile";
   char dir_def[kMAXPATHLEN] = {0}, map_def[kMAXPATHLEN] = {0},
        cer_def[kMAXPATHLEN] = {0}, key_def[kMAXPATHLEN] = {0};

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckProxy: enter: %s", ClientIssuerName);

   // Check if a non-standard file has been requested
   if (getenv("ROOTHOSTCERT")) {
      HostCertConf = getenv("ROOTHOSTCERT");
   } else {
      if (getenv("ROOTETCDIR"))
         HostCertConf.insert(0,getenv("ROOTETCDIR"));
      else
         HostCertConf.insert(0,"/etc/root");
   }

   if (HostCertConf.length()) {
      // string::insert is buggy on some compilers (eg gcc 2.96):
      // new length correct but data not always null terminated
      HostCertConf[HostCertConf.length()] = 0;

      // The user/administrator provided a file ... check if it
      // exists and can be read
      FILE *fconf = 0;
      if (!access(HostCertConf.data(), R_OK) &&
          (fconf = fopen(HostCertConf.data(), "r")) != 0) {
         char line[kMAXPATHLEN];
         if (gDebug > 2)
            ErrorInfo("GlbsToolCheckProxy: reading host cert file %s",
                       HostCertConf.data());

         // ... let's see what it's inside
         while (fgets(line, sizeof(line), fconf)) {
            if (line[0] == '#')
               continue;        // skip comment lines
            int nw =
                sscanf(line, "%s %s %s %s", dir_def, cer_def, key_def,
                       map_def);
            if (nw < 1)
               continue;

            // allow for incomplete lines ... completion with defaults ...
            // and wild card
            if (nw < 4) {
               if (dir_def[0] == '*')
                  strcpy(dir_def, certdir_default);
               strcpy(map_def, gridmap_default);
            } else if (nw == 4) {
               if (dir_def[0] == '*')
                  strcpy(dir_def, certdir_default);
               if (map_def[0] == '*')
                  strcpy(map_def, gridmap_default);
            }
         }
         fclose(fconf);

      } else {
         if (gDebug > 2)
            ErrorInfo
                ("GlbsToolCheckProxy: host cert conf not existing"
                 " or not readable (%s)",HostCertConf.data());
      }
   } else if (gDebug > 2)
      ErrorInfo("GlbsToolCheckProxy: HOSTCERTCONF undefined");

   // Use system defaults if user did not enter a choice
   if (strlen(dir_def) == 0) {
      if (getenv("X509_CERT_DIR") != 0) {
         strcpy(dir_def, getenv("X509_CERT_DIR"));
      } else
         strcpy(dir_def, certdir_default);
   }

   if (strlen(map_def) == 0) {
      if (getenv("GRIDMAP") != 0) {
         strcpy(map_def, getenv("GRIDMAP"));
      } else
         strcpy(map_def, gridmap_default);
   }

   // Now expand to full paths
   char *dir_tmp  = GlbsToolExpand(dir_def);
   char *map_tmp  = GlbsToolExpand(map_def);

   // ... and check the accessibility of the files
   // and set corresponding variables
   if (access(dir_tmp, R_OK)) {
      if (gDebug > 0)
         ErrorInfo("GlbsToolCheckProxy: %s (%s)",
                   "cert directory not existing or not readable",
                    dir_tmp);
   } else {
     // set the corresponding variable
      if (setenv("X509_CERT_DIR", dir_def, 1)) {
         ErrorInfo("GlbsToolCheckProxy: unable to set X509_CERT_DIR ");
      }
   }
   if (dir_tmp) delete[] dir_tmp;

   if (access(map_tmp, R_OK)) {
      if (gDebug > 0)
          ErrorInfo("GlbsToolCheckProxy: %s (%s)",
                    "map file not existing or not readable",
                     map_tmp);
   } else {
     // set the corresponding variable
      if (setenv("GRIDMAP", map_def, 1)) {
         ErrorInfo("GlbsToolCheckProxy: unable to set GRIDMAP ");
      }
   }
   if (map_tmp) delete[] map_tmp;

   // If superuser check if cert and key files were specified;
   // if yes, check that their owner matches and determine its name
   // and user-id, to be used later for the proxy file name
   if (getuid() == 0) {

      char *cer_tmp  = 0;
      if (strlen(cer_def) && cer_def[0] != '*')
         cer_tmp = GlbsToolExpand(cer_def);
      char *key_tmp  = 0;
      if (strlen(key_def) && key_def[0] != '*')
         key_tmp =  GlbsToolExpand(key_def);

      if (cer_tmp && key_tmp) {

         struct stat stc, stk;
         if (stat(cer_tmp,&stc) == -1) {
            ErrorInfo("GlbsToolCheckProxy: stat error:"
                      " file %s (errno: %d) ",cer_tmp,errno);
         } else {
            if (stat(key_tmp,&stk) == -1) {
               ErrorInfo("GlbsToolCheckProxy: stat error:"
                         " file %s (errno: %d) ",key_tmp,errno);
            } else {
               if (stc.st_uid == stk.st_uid) {

                  // Set cert and key files
                  if (gDebug > 2)
                     ErrorInfo("GlbsToolCheckProxy: setting cert: %s ",cer_tmp);
                  if (setenv("X509_USER_CERT", cer_tmp, 1))
                     ErrorInfo("GlbsToolCheckProxy: unable to set X509_USER_CERT ");
                  if (gDebug > 2)
                     ErrorInfo("GlbsToolCheckProxy: setting key: %s ",key_tmp);
                  if (setenv("X509_USER_KEY", key_tmp, 1))
                     ErrorInfo("GlbsToolCheckProxy: unable to set X509_USER_KEY ");
                  // Change effective uid/gid of the current process to the
                  // ones of the certificate file owner
                  RpdSetUid(stc.st_uid);
               }
            }
         }
      }
      if (cer_tmp) delete[] cer_tmp;
      if (key_tmp) delete[] key_tmp;

   } else {
      // Needs to set this for consistency
      if (setenv("X509_USER_CERT", "/etc/grid-security/hostcert.pem", 1))
         ErrorInfo("GlbsToolCheckProxy: unable to set X509_USER_CERT ");
      if (setenv("X509_USER_KEY", "/etc/grid-security/hostkey.pem", 1))
         ErrorInfo("GlbsToolCheckProxy: unable to set X509_USER_KEY ");
   }

   // Now check if there is a proxy file associated with this user
   char proxy_file[256];
   SPrintf(proxy_file, 256, "/tmp/x509up_u%d", getuid());

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckProxy: testing Proxy file: %s",
                proxy_file);

   if (gDebug > 2)
      ErrorInfo("GlbsToolCheckProxy: uid:%d euid:%d gid:%d egid:%d",
                getuid(), geteuid(), getgid(), getegid());

   if (!access(proxy_file, R_OK)) {

      if (setenv("X509_USER_PROXY", proxy_file, 1))
         ErrorInfo("GlbsToolCheckProxy: unable to set X509_USER_PROXY ");

#ifdef R__GLBS22
      globus_gsi_cred_handle_t proxy_cred = 0;

      // Init proxy cred handle
      if (globus_gsi_cred_handle_init(&proxy_cred, 0)
          != GLOBUS_SUCCESS) {
          ErrorInfo("GlbsToolCheckProxy: %s",
                    "couldn't initialize proxy credential handle");
          return 1;
      }

      // Read proxies
      if (globus_gsi_cred_read_proxy(proxy_cred, proxy_file)
          != GLOBUS_SUCCESS) {
          ErrorInfo("GlbsToolCheckProxy: %s %s",
                    "couldn't read proxy from:", proxy_file);
          globus_gsi_cred_handle_destroy(proxy_cred);
          return 1;
      }

      // Get time left to expiration (in seconds) */
      time_t lifetime;
      if( globus_gsi_cred_get_lifetime(proxy_cred,&lifetime)
          != GLOBUS_SUCCESS) {
          ErrorInfo("GlbsToolCheckProxy: %s %s",
                    "couldn't get proxy remaining lifetime");
          globus_gsi_cred_handle_destroy(proxy_cred);
          return 1;
      }
      globus_gsi_cred_handle_destroy(proxy_cred);

      if (lifetime > 0) {

         if (lifetime < 3600)
            ErrorInfo("GlbsToolCheckProxy: WARNING: %s",
                      "proxy will soon expire (less than %d s)",
                       lifetime);

        // Get issuer to be sent back to the client
        X509 *xcert = 0;
        FILE *fcert = fopen(proxy_file, "r");
        if (fcert == 0 || !PEM_read_X509(fcert, &xcert, 0, 0)) {
           ErrorInfo("GlbsToolCheckProxy: unable to load user proxy certificate ");
           return 1;
        }
        fclose(fcert);
        *SubjName = X509_NAME_oneline(X509_get_issuer_name(xcert), 0, 0);
        if (gDebug > 3)
           ErrorInfo("GlbsToolCheckProxy: %s %s",
                   "Proxy Issuer:", *SubjName);

      } else {
         ErrorInfo("GlbsToolCheckProxy: ERROR: %s",
                     "proxy are invalid (expired)");
         return 1;
      }

#else
      // Old version: completly different ...
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
        *SubjName = X509_NAME_oneline(X509_get_issuer_name(pcd->ucert), 0, 0);
        if (gDebug > 3)
           ErrorInfo("GlbsToolCheckProxy: %s %s",
                   "Proxy Issuer:", *SubjName);

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
