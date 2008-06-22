/******************************************************************************/
/*                                                                            */
/*                       X r d C S 2 D C M 2 c s . c c                        */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCS2DCM2csCVSID = "$Id$";

#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include "Xrd/XrdScheduler.hh"
#include "Xrd/XrdTrace.hh"

#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetPeer.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdCS2/XrdCS2DCM.hh"
#include "XrdCS2/XrdCS2DCMFile.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

extern XrdScheduler      XrdSched;

extern XrdSysError       XrdLog;

extern XrdOucTrace       XrdTrace;
 
extern XrdCS2DCM         XrdCS2d;

extern XrdNet            XrdCS2Net;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdCS2Job : XrdJob
{
public:

     void DoIt() {if (Pfn) XrdCS2d.Stage(Tid, FileID, Mode, Lfn, Pfn);
                     else  XrdCS2d.Event(Tid, FileID, Mode, Lfn);
                  delete this;
                 }

          XrdCS2Job(char *tid, char *fileid, char *mode,
                    char *lfn, char *pfn, int freeDo=1)
                   {Tid    = tid;
                    FileID = fileid;
                    strcpy(Mode,  mode);
                    Lfn = lfn; Pfn = pfn;
                    doFree = freeDo;
                   }
         ~XrdCS2Job() {if (doFree)
                          {free(Tid); free(FileID); free(Lfn);
                           if (Pfn) free(Pfn);
                          }
                      }
private:
char *Tid;
char *FileID;
char  Mode[8];
char *Lfn;
char *Pfn;
int   doFree;
};

/******************************************************************************/
/*                              d o E v e n t s                               */
/******************************************************************************/
  
void XrdCS2DCM::doEvents()
{
   const char *Miss = 0, *TraceID = "doEvents";
   char *Eid, *tp, *Tid, *Lfn;

// Each request comes in as
// From ofs:    <traceid> {closer | closew | fwrite} <lfn>
//
// From XrdCS2e: <reqid>:<fileid> prep <physpath>
//
   while((tp = Events.GetLine()))
        {TRACE(DEBUG, "Event: '" <<tp <<"'");
         Tid = Eid = 0;
               if (!(tp = Events.GetToken())) Miss = "traceid";
         else {Tid = strdup(tp);
               if (!(tp = Events.GetToken())) Miss = "eventid";
         else {Eid = strdup(tp);
               if (!(tp = Events.GetToken())) Miss = "lfn";
         else {Lfn = strdup(tp);
               Miss = 0;
              } } }

         if (Miss) {XrdLog.Emsg("doEvents", "Missing", Miss, "in event.");
                    if (Tid) free(Tid);
                    continue;
                   }

         XrdSched.Schedule((XrdJob *)new XrdCS2Job(Tid,Eid,(char *)"",Lfn,0));
        }

// If we exit then we lost the connection
//
   XrdLog.Emsg("doEvents", "Exiting; lost event connection to xrootd!");
   exit(8);
}

/******************************************************************************/
/*                            d o M e s s a g e s                             */
/******************************************************************************/

void XrdCS2DCM::doMessages()
{
   const char *Miss = 0, *TraceID = "doMessages";
   XrdNetPeer      myPeer;
   XrdOucTokenizer Msg(0);
   char *tp, *cmd, *arg1, *arg2;

// Make sure we can really receive messages
//
   if (!udpPort) 
      {XrdLog.Emsg("main", "No udp port specified, udp event msgs ignored.");
       return;
      }

// Get udp messages in an endless loop:
//
// addfile <physpath> <fid>
// addlink <physpath> <lfn>
// rmfile  <fid>
// rmlink  <lfn>
//
   while(XrdCS2Net.Accept(myPeer))
        {Msg.Attach(myPeer.InetBuff->data);
         while((tp = Msg.GetLine()))
              {TRACE(DEBUG, "Message: '" <<tp <<"'");
                    if (!(cmd  = Msg.GetToken())) Miss = "command";
               else if (!(arg1 = Msg.GetToken())) Miss = "arg 1";
               else if (*cmd == 'a'
                    &&  !(arg2 = Msg.GetToken())) Miss = "arg 2";
               else Miss = 0;

               if (Miss) {XrdLog.Emsg("doMsg", "Missing", Miss, "in request.");
                          break;
                         }
               XrdSched.Schedule((XrdJob *)new XrdCS2Job(arg1,cmd,(char *)"",arg2,0));
              }
        }
}

/******************************************************************************/
/*                            d o R e q u e s t s                             */
/******************************************************************************/
  
void XrdCS2DCM::doRequests()
{
   const char *Miss = 0, *TraceID = "doRequests";
   char *Fid, Fsize[24], Mode[8], *tp, *Tid, *Lfn, *Pfn;

   memset(Fsize, 0, sizeof(Fsize));
   Mode[sizeof(Mode)-1] = '\0';

// Each request comes in as
// <traceid> <fileid> {r|w[c][t]} <lfn> <pfn>
//
   while((tp = Request.GetLine()))
        {TRACE(DEBUG, "Request: '" <<tp <<"'");
         Tid = Fid = Lfn = 0;
               if (!(tp = Request.GetToken())) Miss = "traceid";
         else {Tid = strdup(tp);
               if (!(tp = Request.GetToken())) Miss = "file id";
         else {Fid = strdup(tp);
               if (!(tp = Request.GetToken())) Miss = "file size";
         else {strncpy(Fsize, tp, sizeof(Fsize)-1);
               if (!(tp = Request.GetToken())) Miss = "mode";
         else {strncpy(Mode, tp, sizeof(Mode)-1);
               if (!(tp = Request.GetToken())) Miss = "lfn";
         else {Lfn = strdup(tp);
               if (!(tp = Request.GetToken())) Miss = "pfn";
         else {Pfn = strdup(tp);
               Miss = 0;
              } } } } } }

         if (!Miss && !strcmp(Fid, "$cs2.fid")) Miss = "actual file id";

         if (Miss) {XrdLog.Emsg("doReq", "Missing", Miss, "in request.");
                    if (Tid) free(Tid);
                    if (Fid) free(Fid);
                    if (Lfn) free(Lfn);
                    continue;
                   }

         XrdSched.Schedule((XrdJob *)new XrdCS2Job(Tid,Fid,Mode,Lfn,Pfn));
        }

// If we exit then we lost the connection
//
   XrdLog.Emsg("doRequests", "Exiting; lost request connection to xrootd!");
   exit(8);
}

/******************************************************************************/
/*                                 E v e n t                                  */
/******************************************************************************/
  
void XrdCS2DCM::Event(const char *Tid, const char *Eid, const char *Mode,
                      const char *Lfn)
{

// Process the event
//
        if (!strcmp("closer",  Eid)
        ||  !strcmp("closew",  Eid)) Release(Tid, Lfn);
   else if (!strcmp("fwrite",  Eid))
           {char thePath[2048];
            makeFname(thePath, APath, APlen, Lfn);
            XrdCS2DCMFile::Modify(thePath);
           }
   else if (!strcmp("prep",    Eid)
        ||  !strcmp("addfile", Eid)) Prep(Tid, Lfn);
   else if (!strcmp("addlink", Eid)) addLink(Tid, Lfn);
   else if (!strcmp("rmfile",  Eid)) unPrep(Tid);
   else if (!strcmp("rmlink",  Eid)) delLink(Tid);
   else XrdLog.Emsg("Event", "Received unknown event -", Tid, Eid);
}

/******************************************************************************/
/*                                 S t a g e                                  */
/******************************************************************************/
  
void XrdCS2DCM::Stage(const char *Tid, char *Fid, char *Mode,
                            char *Lfn, char *Pfn)
{
   const char *TraceID = "Stage";
   char thePath[2048], xeqPath[1024], oMode = 'r', *op;
   char *ofsEvent = getenv("XRDOFSEVENTS");
   int thePathLen = makeFname(thePath, APath, APlen, Lfn);

                                  //12345678901234
   struct iovec iox[10]= {{(char *)"#!/bin/sh\n",          10}, // 0
                          {(char *)"Rfn=",                  4}, // 1
                          {thePath,                thePathLen}, // 2
                          {(char *)"\nPfn=",                5}, // 3
                          {Pfn,                   strlen(Pfn)}, // 4
                          {(char *)"\nLfn=",                5}, // 5
                          {Lfn,                   strlen(Lfn)}, // 6
                          {(char *)"\nEfn=",                5}, // 7
                          {ofsEvent,         strlen(ofsEvent)}, // 8
                          {(char *)"\n",                    1}};// 9
   int Oflags, fnfd, rc;

// Convert mode to open type flags
//
   op = Mode; Oflags = 0;
   while(*op)
        {switch(*op++)
               {case 'r': Oflags  = O_RDONLY; oMode = 'r'; break;
                case 'w': Oflags |= O_RDWR;   oMode = 'w'; break;
                case 'c': Oflags |= O_CREAT;  oMode = 'c'; break;
                case 't': Oflags |= O_TRUNC;               break;
                case 'x': Oflags |= O_EXCL;                break;
                default:  XrdLog.Emsg("Stage", "Invalid mode:", Mode, Lfn);
                          failRequest(Pfn);
                          return;
               }
       }

// Make the directory structure for the upcomming symlink
//
   if ((rc = XrdOucUtils::makePath(Pfn,0770)))
      {XrdLog.Emsg("Stage", rc, "create directory path for", Pfn);
       return;
      }

// Create a file that will hold this information
//
   LockDir();
   if (!XrdCS2DCMFile::Create(thePath, oMode, Pfn))
      {failRequest(Pfn);
       UnLockDir();
       return;
      }

// Construct name of the script that will create the symlink and append the
// the subreqid to the database file we just created.
//
   strcpy(xeqPath,      PPath);
   strcpy(xeqPath+PPlen,Fid);

// Construct the script that will create the symlink to the physical file
//
   do {fnfd = open(xeqPath, O_RDWR|O_CREAT, S_IRUSR|S_IXUSR);}
      while( fnfd < 0 && errno == EINTR);
   if (fnfd < 0)
      {XrdLog.Emsg("Stage", errno, "create script", xeqPath);
       failRequest(Pfn);
       UnLockDir();
       return;
      }

// Write the information into the file
//
   if (writev(fnfd, iox, 10) < 0)
      {XrdLog.Emsg("Stage", errno, "write script", xeqPath);
       failRequest(Pfn);
       UnLockDir();
       return;
      }

// All done here
//
   close(fnfd);
   UnLockDir();

// Now we can schedule the I/O. The I/O has already been scheduled for file
// creation events. So, don't do it twice.
//
   TRACE(DEBUG, Tid <<" open mode " << Mode <<" file " <<Fid <<' ' <<Lfn);
   if (!(Oflags & O_CREAT) && !CS2_Open(Tid, Fid, Lfn, Oflags, 0))
      {failRequest(Pfn);
       return;
      }
}

/******************************************************************************/
/*                           f a i l R e q u e s t                            */
/******************************************************************************/
  
void XrdCS2DCM::failRequest(char *Pfn)
{
   char buff[2048];
   int rc, fd, PfnLen = strlen(Pfn);

// Construct a fail file name
//
   strcpy(buff, Pfn);
   strcpy(buff+PfnLen, ".fail");

// Add a fail file to keep staging this file at bay
//
   if ((rc = XrdOucUtils::makePath(Pfn,0770)))
      XrdLog.Emsg("failRequest", rc, "create directory path for", Pfn);
      else {do {fd = open(buff, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);}
               while(fd < 0 && errno == EINTR);
            if (fd > 0) close(fd);
           }
}

/******************************************************************************/
/*                             m a k e F n a m e                              */
/******************************************************************************/
  
int XrdCS2DCM::makeFname(char *thePath, const char *pfxPath, int pfxPlen,
                                        const char *fn)
{
   char *tp;

// Construct the filename of where we will record the RequestID and pfn
//
   strcpy(thePath, pfxPath);
   tp = thePath+pfxPlen;
   while(*fn) {*tp = (*fn == '/' ? '%' : *fn); tp++; fn++;}
   *tp = '\0';
   return strlen(thePath);
}

/******************************************************************************/
/*                               R e l e a s e                                */
/******************************************************************************/
  
int XrdCS2DCM::Release(const char *Tid, const char *Lfn, int Fail)
{
   const char *TraceID = "Release";
   XrdCS2DCMFile theFile;
   char *Pfn=0, newPath[2048], thePath[2048];
   int rc, isNew, isMod;
   unsigned long long reqID;

// Construct the name of the record file based on the Lfn
//
   makeFname(thePath, APath, APlen, Lfn);

// Process the file
//
   if ((rc = theFile.Init(thePath, (Fail ? UpTime : 0))))
      {if (rc == ENOENT)
          {TRACE(DEBUG, "Release file gone Tid=" <<Tid <<" path=" <<thePath);}
       return 0;
      }

// Prepare to process the subreqid's
//
   isMod = theFile.Modified();
   isNew = theFile.Mode() == 'c';
   Pfn   = theFile.Pfn();

// We consider the request as "failed" if the file was opened for create but a
// write never happened to the file.
//
   if (isNew && !isMod) Fail = 1;

// Issue updateFailed(), putFail(), putDone() or getDone() for each subreqid
//
   while((reqID = theFile.reqID()))
        {if (isMod)
            if (Fail) CS2_wFail(Tid, reqID, Pfn, isNew);
               else   CS2_wDone(Tid, reqID, Pfn);
            else      CS2_rDone(Tid, reqID, Lfn);
        }

// Move the file to the "closed" directory
//
   makeFname(newPath, CPath, CPlen, Lfn);
   rename(thePath, newPath);
   return 1;
}
