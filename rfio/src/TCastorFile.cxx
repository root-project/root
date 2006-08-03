// @(#)root/net:$Name:  $:$Id: TCastorFile.cxx,v 1.14 2006/07/24 16:26:28 rdm Exp $
// Author: Fons Rademakers + Jean-Damien Durand 17/09/2003 + Ben Couturier 31/05/2005
// + Giulia Taurelli 26/04/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCastorFile                                                          //
//                                                                      //
// A TCastorFile is like a normal TNetFile except that it obtains the   //
// remote node (disk server) via the CASTOR API, once the disk server   //
// and the local file path are determined, the file will be accessed    //
// via the rootd daemon. File names have to be specified like:          //
//    castor:/castor/cern.ch/user/r/rdm/bla.root.                       //
//                                                                      //
// If Castor 2.1 is used the file names can also be specified           //
// in the following ways:                                               //
//                                                                      //
//  castor://stager_host:stager_port/?path=/castor/cern.ch/user/        //
//    r/rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION   //
//                                                                      //
//  castor://stager_host/?path=/castor/cern.ch/user/                    //
//    r/rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION   //
//                                                                      //
//  castor:///?path=/castor/cern.ch/user/                               //
//    r/rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION   //
//                                                                      //
// path is mandatory as parameter but all the other ones are optional.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "NetErrors.h"
#include "TCastorFile.h"
#include "TError.h"

#include <stdlib.h>
#include <errno.h>

#ifdef _WIN32
#include <WinDef.h>
#include <WinSock2.h>
#endif

#ifdef R__CASTOR2
#include <stager_api.h>       // For the new CASTOR 2 Stager
#include <RfioTURL.h>         // For the new url parsing
#endif
#define RFIO_KERNEL           // Get access to extra symbols in the headers
#include <stage_api.h>        // Dial with CASTOR stager
#include <rfio_api.h>         // Util routines from rfio
#include <Cns_api.h>          // Dial with CASTOR Name Server
#include <Cglobals.h>
#include <rfio_constants.h>

#define RFIO_USE_CASTOR_V2 "RFIO_USE_CASTOR_V2"
#define RFIO_HSM_BASETYPE  0x0
#define RFIO_HSM_CNS       RFIO_HSM_BASETYPE+1

extern "C" { int rfio_HsmIf_reqtoput (char *); }
extern "C" { int DLL_DECL rfio_parse(char *, char **, char **); }
extern "C" { int rfio_HsmIf_IsHsmFile (const char *); }
extern "C" { char *getconfent(char *, char *, int); }

#ifdef R__CASTOR2
extern int tStageHostKey;
extern int tStagePortKey;
extern int tSvcClassKey;
extern int tCastorVersionKey;

//______________________________________________________________________________
int TCastorFile::ParseAndSetGlobal()
{
   // This function does the parsing to deal with the new Turl
   // and set the global variables needed by castor.

   char **globalHost,  **globalSvc;
   int  *globalVersion, *globalPort;

   char *myHost,  *mySvcClass;
   int  myVersion, myPort;

   int ret;
   char *options, *myCastorVersion, *path2, *path1, *q1, *q;
   int versionNum;
   char *myTurl;
   int newTurl=0;

   globalHost = globalSvc = 0;
   myHost = mySvcClass = 0;
   versionNum = 0;
   globalVersion = globalPort = 0;
   myVersion = myPort = 0;

   myTurl = strdup(fUrl.GetFile()); // parsing of host and port not done by TUrl class.
   options = strdup(fUrl.GetOptions());

   // I parse the host and port and I save it into myHost and myPort

   // check to see if it is the new Turl or not

   path2 = strstr(options,"path=");
   if (path2) {
      newTurl = 1;
      path2 += 5; // to remove "path="
   }

   if (newTurl) {

      q = strstr(myTurl,":");
      q1 = strstr(myTurl,"/");

      if (myTurl!=q && myTurl!=q1) {
         if (q && q1) {
            *q='\0';
            q+=1;
            fUrl.SetHost(myTurl);
            *q1='\0';
            q1+=1;
            fUrl.SetPort(atoi(q));
         }
         if (!q && q1) {
            *q1='\0';
            q1+=1;
            fUrl.SetHost(myTurl);
         }
      } else {
         if(q1) q1+=1;
      }
      if (q1) {
         fUrl.SetFile(q1);
      } else {
         fUrl.SetFile("");
      }
   }

   free(myTurl);


   // Stage host
   ret=Cglobals_get(&tStageHostKey,(void **)&globalHost,sizeof(void*));
   if (ret<0) return -1;

   if (*globalHost) {
      free(*globalHost);
      *globalHost=0;
   }

   if (strcmp(fUrl.GetHost(),"")) {
      *globalHost=strdup(fUrl.GetHost());
   }

   // stage port
   ret=Cglobals_get(&tStagePortKey,(void **)&globalPort,sizeof(int));

   if (ret<0) {
      if (*globalHost) {
         free(*globalHost);
         *globalHost=0;
      }
      return -1;
   }
   *globalPort=0;
   if (fUrl.GetPort()>0) {
      *globalPort=fUrl.GetPort();
   }

   // From here I consider the Option given after ? (updating the right file path)
   // parsing of options given


   mySvcClass=strstr(options,"svcClass=");
   if (mySvcClass) {
      mySvcClass+=9; // to remove "svcClass="
   }

   myCastorVersion=strstr(options,"castorVersion=");
   if (myCastorVersion) {
      myCastorVersion+=14; // to remove "castorVersion="
   }

   if (mySvcClass) {
      q1=strstr(mySvcClass,"&");
      if (q1) {
         *q1='\0';
      }
   }
   if (myCastorVersion) {
      q1=strstr(myCastorVersion,"&");
      if (q1){
         *q1='\0';
      }
   }
   if (path2) {
      q1=strstr(path2,"&");
      if(q1) {
         *q1='\0';
      }
   }

   path1 = (char*)fUrl.GetFile();

   if (strcmp(path1,"") && path2) {
      // not possible to have the path twice
      if (*globalHost) {
         free(*globalHost);
         *globalHost=0;
      }
      free(options);
      return -1;
   }

   // ... only path as option could be specified for the new Turl
   // ... and path1 is used for the old one

   if (!strcmp(path1,"") && !path2) {
      // at least the path should be specified
      if (*globalHost) {
         free(*globalHost);
         *globalHost=0;
      }
      free(options);
      return(-1);
   }

   if (path2) {
      fUrl.SetFile(path2);
   }

   // Svc class set
   ret=Cglobals_get(&tSvcClassKey,(void **)&globalSvc,sizeof(void*));

   if (ret<0) {
      if(*globalHost){
         free(*globalHost);
         *globalHost=0;
      }
      free(options);
      return -1;
   }

   if (*globalSvc) {
      free(*globalSvc);
      *globalSvc=0;
   }

   if (mySvcClass && strcmp(mySvcClass,"")) {
      *globalSvc=strdup(mySvcClass);
   }

   // castor version
   ret=Cglobals_get(&tCastorVersionKey,(void **)&globalVersion,sizeof(int));

   if (ret<0) {
      serrno = EINVAL;
      if (*globalHost) {
         free(*globalHost);
         *globalHost=0;
      }
      if (*globalSvc) {
         free(*globalSvc);
         *globalSvc=0;
      }
      free(options);
      return -1;

   }
   *globalVersion=0;

   if (myCastorVersion) {
      if (!strcmp(myCastorVersion,"2")) {
         versionNum=2;
      }
      if (!strcmp(myCastorVersion,"1")) {
         versionNum=1;
      }

   }

   if (versionNum) {
      *globalVersion=versionNum;
   }

   ret=getDefaultForGlobal(globalHost,globalPort,globalSvc,globalVersion);
   if (ret<0) {
      if (*globalHost) {
         free(*globalHost);
         *globalHost=0;
      }
      if (*globalSvc) {
         free(*globalSvc);
         *globalSvc=0;
      }
      free(options);
      return -1;

   }
   free(options);
   path1=strdup(fUrl.GetFile());
   if (strstr(path1,"/castor")== path1) {
      fUrl.SetHost("");
      fUrl.SetPort(0);
   }
   free(path1);
   return 0;
}

//______________________________________________________________________________
static int UseCastor2API()
{
   // Function that checks whether we should use the old or new stager API.

   int *auxVal=0;
   int ret=Cglobals_get(& tCastorVersionKey, (void**) &auxVal,sizeof(int));
   if (ret==0) {
      return *auxVal==2?1:0;
   }
   return 0;
}

#else

//______________________________________________________________________________
static int UseCastor2API()
{
   // Function that checks whether we should use the old or new stager API.

   char *p;

   if (((p = getenv(RFIO_USE_CASTOR_V2)) == 0) &&
       ((p = getconfent("RFIO","USE_CASTOR_V2",0)) == 0)) {
      // Variable not set: compat mode
      return 0;
   }
   if ((strcmp(p,"YES") == 0) || (strcmp(p,"yes") == 0) || (atoi(p) == 1)) {
      // Variable set to yes or 1 but old CASTOR 1: compat mode + warning
      static int once = 0;
      if (!once) {
         ::Warning("UseCastor2API", "asked to use CASTOR 2, but linked with CASTOR 1");
         once = 1;
      }
      return 0;
   }
   // Variable set but not to 1 : compat mode
   return 0;
}
#endif


ClassImp(TCastorFile)

//______________________________________________________________________________
TCastorFile::TCastorFile(const char *url, Option_t *option, const char *ftitle,
                              Int_t compress, Int_t netopt)
      : TNetFile(url, ftitle, compress, kFALSE)
{
   // Create a TCastorFile. A TCastorFile is like a normal TNetFile except
   // that it obtains the remote node (disk server) via the CASTOR API, once
   // the disk server and the local file path are determined, the file will
   // be accessed via the rootd daemon. File names have to be specified like:
   //    castor:/castor/cern.ch/user/r/rdm/bla.root.
   // The other arguments are the same as for TNetFile and TFile.

   fIsCastor  = kFALSE;
   fWrittenTo = kFALSE;

   // file is always created by stage_out_hsm() and therefore
   // exists when opened by rootd
   TString opt = option;
   opt.ToUpper();
   if (opt == "NEW" || opt == "CREATE")
      opt = "RECREATE";

   Create(url, opt, netopt);
}

//______________________________________________________________________________
void TCastorFile::FindServerAndPath()
{
   // Find the CASTOR disk server and internal file path.

#ifdef R__CASTOR2
   int ret=ParseAndSetGlobal();

   if (ret<0) {
      Error("FindServerAndPath", "can't parse the turl given");
      return;
   }
#endif

   if (!UseCastor2API()) {

      struct stgcat_entry *stcp_output = 0;

      if (rfio_HsmIf_IsHsmFile(fUrl.GetFile()) == RFIO_HSM_CNS) {
         // This is a CASTOR file
         int flags = O_RDONLY;
         struct Cns_filestat st;
         int rc;
         char stageoutbuf[1025];
         char stageerrbuf[1025];

         // Check with internal stage limits - preventing overflow afterwards
         if (strlen(fUrl.GetFile()) > STAGE_MAX_HSMLENGTH) {
            serrno = ENAMETOOLONG;
            Error("FindServerAndPath", "can't open %s, error %d (%s)", fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         // Prepare the flags
         if (fOption == "CREATE" || fOption == "RECREATE" || fOption == "UPDATE")
            flags |= O_RDWR;
         if (fOption == "CREATE" || fOption == "RECREATE")
            flags |= O_CREAT | O_TRUNC;

         // Check if an existing file is going to be updated
         memset(&st, 0, sizeof(st));
         rc = Cns_stat(fUrl.GetFile(), &st);

         // Make sure that filesize is 0 if file doesn't exist
         // or that we will create (stage_out) if O_TRUNC.
         if (rc == -1 || ((flags & O_TRUNC) != 0))
            st.filesize = 0;

         // Makes sure stage api does not write automatically to stdout/stderr
         if (stage_setoutbuf(stageoutbuf, 1024) != 0) {
            Error("FindServerAndPath", "can't open %s, stage_setoutbuf, error %d (%s)",
                  fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }
         if (stage_seterrbuf(stageerrbuf, 1024) != 0) {
            Error("FindServerAndPath", "can't open %s, stage_seterrbuf, error %d (%s)",
                  fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         struct stgcat_entry stcp_input;
         int nstcp_output;

         memset(&stcp_input, 0, sizeof(struct stgcat_entry));
         strcpy(stcp_input.u1.h.xfile, fUrl.GetFile());
         if (flags == O_RDONLY || st.filesize > 0) {
         // Do a recall
            if (stage_in_hsm((u_signed64) 0,          // Ebusy is possible...
                             (int) flags,             // open flags
                             (char *) 0,              // hostname
                             (char *) 0,              // pooluser
                             (int) 1,                 // nstcp_input
                             (struct stgcat_entry *) &stcp_input, // stcp_input
                             (int *) &nstcp_output,   // nstcp_output
                             (struct stgcat_entry **) &stcp_output, // stcp_output
                             (int) 0,                 // nstpp_input
                             (struct stgpath_entry *) 0 // stpp_input
                            ) != 0) {
               Error("FindServerAndPath", "can't open %s, stage_in_hsm error %d (%s)",
                     fUrl.GetFile(), serrno, sstrerror(serrno));
               return;
            }
         } else {
            // Do a creation
            if (stage_out_hsm((u_signed64) 0,          // Ebusy is possible...
                              (int) flags,             // open flags
                              (mode_t) 0666,           // open mode (c.f. also umask)
                              // Note: This is S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH, c.f. fopen(2)
                              (char *) 0,              // hostname
                              (char *) 0,              // pooluser
                              (int) 1,                 // nstcp_input
                              (struct stgcat_entry *) &stcp_input, // stcp_input
                              (int *) &nstcp_output,   // nstcp_output
                              (struct stgcat_entry **) &stcp_output, // stcp_output
                              (int) 0,                       // nstpp_input
                              (struct stgpath_entry *) 0     // stpp_input
                             ) != 0) {
               Error("FindServerAndPath", "can't open %s, stage_out_hsm error %d (%s)",
               fUrl.GetFile(), serrno, sstrerror(serrno));
               return;
            }
         }
         if ((nstcp_output != 1) || (stcp_output == 0) ||
            (*(stcp_output->ipath) == '\0')) {
            // Impossible
            serrno = SEINTERNAL;
            if (stcp_output != 0) free(stcp_output);
            Error("FindServerAndPath", "can't open %s, error %d (%s)",
            fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         // Parse orig string to get disk server host
         char *filename;
         char *realhost = 0;
         rfio_parse(stcp_output->ipath, &realhost, &filename);
         if (realhost == 0) {
            serrno = SEINTERNAL;
            Error("FindServerAndPath", "can't open %s, get disk server hostname from %s error %d (%s)",
                  fUrl.GetFile(), stcp_output->ipath, errno, sstrerror(serrno));
            free(stcp_output);
            return;
         }
         // Save real host and internal path
         fDiskServer = realhost;
         if (filename[0] != '/') {
            // Make file 'local' to the host
            fInternalPath  = "/";
            fInternalPath += filename;
         } else {
            fInternalPath = filename;
         }

         if (st.filesize == 0) {
            // Will force notification to stage when the file is closed
            fWrittenTo = kTRUE;
         }
      }

      // Set the protocol prefix for TNetFile.
      // For the cern.ch domain we set the default authentication
      // method to UidGid, i.e. as for rfiod; for this we need
      // the full FQDN or address in "nnn.mmm.iii.jjj" form
      // (it can be changed by a proper directive in $HOME/.rootauthrc)
      TString r;
      TString fqdn;
      TInetAddress addr = gSystem->GetHostByName(fDiskServer);
      if (addr.IsValid()) {
         fqdn = addr.GetHostName();
         if (fqdn == "UnNamedHost")
            fqdn = addr.GetHostAddress();
         if (fqdn.EndsWith(".cern.ch") || fqdn.BeginsWith("137.138."))
            r = "rootug://";
         else
            r = "root://";
      } else
         r = "root://";

      // Update fUrl with new path
      r += fDiskServer + "/";
      r += fInternalPath;
      TUrl rurl(r);
      fUrl = rurl;

      Info("FindServerAndPath"," fDiskServer: %s, r: %s", fDiskServer.Data(), r.Data());

      // Now ipath is not null and contains the real internal path on the disk
      // server 'host', e.g. it is fDiskServer:fInternalPath
      fInternalPath = stcp_output->ipath;
      free(stcp_output);

   } else {

#ifdef R__CASTOR2
      // We use the new stager API
      int flags = O_RDONLY;
      int rc;
      struct stage_io_fileresp *response = 0;
      char *requestId = 0, *url = 0;
      char stageerrbuf[1025];

      // Prepare the flags
      if (fOption == "CREATE" || fOption == "RECREATE" || fOption == "UPDATE")
         flags |= O_RDWR;
      if (fOption == "CREATE" || fOption == "RECREATE")
         flags |= O_CREAT | O_TRUNC;

      stage_seterrbuf(stageerrbuf, 1024);

      int* auxVal;
      char ** auxPoint;
      struct stage_options opts;
      opts.stage_host=0;
      opts.stage_port=0;
      opts.service_class=0;
      opts.stage_version=0;

      ret=Cglobals_get(& tStageHostKey, (void**) &auxPoint,sizeof(void*));
      if(ret==0){
         opts.stage_host=*auxPoint;
      }
      ret=Cglobals_get(& tStagePortKey, (void**) &auxVal,sizeof(int));
      if(ret==0){
         opts.stage_port=*auxVal;
      }
      opts.stage_version=2;
      ret=Cglobals_get(& tSvcClassKey, (void**) &auxPoint,sizeof(void*));
      if (ret==0){
         opts.service_class=*auxPoint;
      }

      rc = stage_open(0,
                      MOVER_PROTOCOL_ROOT,
                      (fUrl.GetFile()),
                      flags,
                      (mode_t) 0666,
                      0,
                      &response,
                      &requestId,
                      &opts); // global values used as options

      if (rc != 0) {
         Error("FindServerAndPath", "stage_open failed: %s (%s)",
               sstrerror(serrno), stageerrbuf);
         if (response) free(response);
         if (requestId) free(requestId);
         return;
      }

      if (response == 0) {
         Error("FindServerAndPath", "response was null for %s (Request %s) %d/%s",
               fUrl.GetFile(), requestId,
               serrno, sstrerror(serrno));
         if (requestId) free(requestId);
         return;
      }

      if (response->errorCode != 0) {
         serrno = response->errorCode;
         Error("FindServerAndPath", "error getting file %s (Request %s) %d/%s",
               fUrl.GetFile(), requestId,
               serrno, sstrerror(serrno));
         free(response);
         if (requestId) free(requestId);
         return;
      }

      url = stage_geturl(response);

      if (url == 0) {
         Error("FindServerAndPath", "error getting file %s (Request %s) %d/%s",
               fUrl.GetFile(), requestId,
               serrno, sstrerror(serrno));
         free(response);
         if (requestId) free(requestId);
         return;
      }

      TUrl rurl(url);
      // Set the protocol prefix for TNetFile.
      // For the cern.ch domain we set the default authentication
      // method to UidGid, i.e. as for rfiod; for this we need
      // the full FQDN or address in "nnn.mmm.iii.jjj" form
      // (it can be changed by a proper directive in $HOME/.rootauthrc)
      TString p;
      TString fqdn = rurl.GetHostFQDN();
      if (fqdn.EndsWith(".cern.ch") || fqdn.BeginsWith("137.138."))
         p = "rootug";
      else
         p = "root";

      // Update protocol and fUrl
      rurl.SetProtocol(p);
      fUrl = rurl;

      if (response) free(response);
      if (url) free(url);
      if (requestId) free(requestId);
#endif

   }

   fIsCastor = kTRUE;
}

//______________________________________________________________________________
Int_t TCastorFile::SysClose(Int_t fd)
{
   // Close currently open file.

   Int_t r = TNetFile::SysClose(fd);

   if (!UseCastor2API()) {
      if (fIsCastor && fWrittenTo) {
         // CASTOR file was created or modified
         rfio_HsmIf_reqtoput((char *)fInternalPath.Data());
         fWrittenTo = kFALSE;
      }
   }

   return r;
}

//______________________________________________________________________________
Bool_t TCastorFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (TNetFile::WriteBuffer(buf, len))
      return kTRUE;

   if (!UseCastor2API()) {
      if (fIsCastor && !fWrittenTo && len > 0) {
         stage_hsm_t hsmfile;

         // Change status of file in stage catalog from STAGED to STAGEOUT
         memset(&hsmfile, 0, sizeof(hsmfile));
         strcpy(hsmfile.upath, fInternalPath);
         if (stage_updc_filchg(0, &hsmfile) < 0) {
         Error("WriteBuffer", "error calling stage_updc_filchg");
         return kTRUE;
         }
         fWrittenTo = kTRUE;
      }
   }

   return kFALSE;
}

//______________________________________________________________________________
void TCastorFile::ConnectServer(Int_t *stat, EMessageTypes *kind, Int_t netopt,
                                Int_t tcpwindowsize, Bool_t forceOpen,
                                Bool_t forceRead)
{
   // Connect to remote rootd server on CASTOR disk server.

   FindServerAndPath();

   // Continue only if successful
   if (fIsCastor) {
      TNetFile::ConnectServer(stat, kind, netopt, tcpwindowsize, forceOpen, forceRead);
   } else {
      // Failure: fill these to signal it to TNetFile
      *stat = kErrFileOpen;
      *kind = kROOTD_ERR;
   }
}
