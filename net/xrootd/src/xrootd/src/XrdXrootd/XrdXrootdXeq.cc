/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d X e q . c c                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdXrootdXeqCVSID = "$Id$";

#include <stdio.h>

#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdLink.hh"
#include "XrdXrootd/XrdXrootdAio.hh"
#include "XrdXrootd/XrdXrootdCallBack.hh"
#include "XrdXrootd/XrdXrootdFile.hh"
#include "XrdXrootd/XrdXrootdFileLock.hh"
#include "XrdXrootd/XrdXrootdJob.hh"
#include "XrdXrootd/XrdXrootdMonitor.hh"
#include "XrdXrootd/XrdXrootdPio.hh"
#include "XrdXrootd/XrdXrootdPrepare.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdStats.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
#include "XrdXrootd/XrdXrootdXPath.hh"
  
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

extern XrdOucTrace *XrdXrootdTrace;

/******************************************************************************/
/*                      L o c a l   S t r u c t u r e s                       */
/******************************************************************************/
  
struct XrdXrootdFHandle
       {kXR_int32 handle;

        void Set(kXR_char *ch)
            {memcpy((void *)&handle, (const void *)ch, sizeof(handle));}
        XrdXrootdFHandle() {}
        XrdXrootdFHandle(kXR_char *ch) {Set(ch);}
       ~XrdXrootdFHandle() {}
       };

struct XrdXrootdSessID
       {unsigned int       Sid;
                 int       Pid;
                 int       FD;
        unsigned int       Inst;

        XrdXrootdSessID() {}
       ~XrdXrootdSessID() {}
       };

/******************************************************************************/
/*                         L o c a l   D e f i n e s                          */
/******************************************************************************/

#define CRED (const XrdSecEntity *)Client

#define TRACELINK Link

#define UPSTATS(x) SI->statsMutex.Lock(); SI->x++; SI->statsMutex.UnLock()
 
/******************************************************************************/
/*                              d o _ A d m i n                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Admin()
{
   return Response.Send(kXR_Unsupported, "admin request is not supported");
}
  
/******************************************************************************/
/*                               d o _ A u t h                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Auth()
{
    struct sockaddr netaddr;
    XrdSecCredentials cred;
    XrdSecParameters *parm = 0;
    XrdOucErrInfo     eMsg;
    const char *eText;
    int rc, n;

// Ignore authenticate requests if security turned off
//
   if (!CIA) return Response.Send();
   cred.size   = Request.header.dlen;
   cred.buffer = argp->buff;

// If we have no auth protocol, try to get it. Track number of times we got a
// protocol object as the read count (we will zero it out later).
//
   if (!AuthProt)
      {Link->Name(&netaddr);
       if (!(AuthProt = CIA->getProtocol(Link->Host(),netaddr,&cred,&eMsg)))
          {eText = eMsg.getErrText(rc);
           eDest.Emsg("Xeq", "User authentication failed;", eText);
           return Response.Send(kXR_NotAuthorized, eText);
          }
       AuthProt->Entity.tident = Link->ID; numReads++;
      }

// Now try to authenticate the client using the current protocol
//
   if (!(rc = AuthProt->Authenticate(&cred, &parm, &eMsg)))
      {const char *msg = (Status & XRD_ADMINUSER ? "admin login as"
                                                 : "login as");
       rc = Response.Send(); Status &= ~XRD_NEED_AUTH;
       Client = &AuthProt->Entity; numReads = 0;
       if (Client->name) 
          eDest.Log(SYS_LOG_01, "Xeq", Link->ID, msg, Client->name);
          else
          eDest.Log(SYS_LOG_01, "Xeq", Link->ID, msg, "nobody");
       return rc;
      }

// If we need to continue authentication, tell the client as much
//
   if (rc > 0)
      {TRACEP(LOGIN, "more auth requested; sz=" <<(parm ? parm->size : 0));
       if (parm) {rc = Response.Send(kXR_authmore, parm->buffer, parm->size);
                  delete parm;
                  return rc;
                 }
       eDest.Emsg("Xeq", "Security requested additional auth w/o parms!");
       return Response.Send(kXR_ServerError,"invalid authentication exchange");
      }

// Authentication failed. We will delete the authentication object and zero
// out the pointer. We can do this without any locks because this section is
// single threaded relative to a connection. To prevent guessing attacks, we
// wait a variable amount of time if there have been 3 or more tries.
//
   if (AuthProt) {AuthProt->Delete(); AuthProt = 0;}
   if ((n = numReads - 2) > 0) XrdSysTimer::Snooze(n > 5 ? 5 : n);

// We got an error, bail out.
//
   eText = eMsg.getErrText(rc);
   eDest.Emsg("Xeq", "User authentication failed;", eText);
   return Response.Send(kXR_NotAuthorized, eText);
}

/******************************************************************************/
/*                               d o _ B i n d                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Bind()
{
   XrdXrootdSessID *sp = (XrdXrootdSessID *)Request.bind.sessid;
   XrdXrootdProtocol *pp;
   XrdLink *lp;
   int i, pPid, rc;
   char buff[64], *cp, *dp;

// Update misc stats count
//
   UPSTATS(miscCnt);

// Find the link we are to bind to
//
   if (sp->FD <= 0 || !(lp = XrdLink::fd2link(sp->FD, sp->Inst)))
      return Response.Send(kXR_NotFound, "session not found");

// The link may have escaped so we need to hold this link and try again
//
   lp->Hold(1);
   if (lp != XrdLink::fd2link(sp->FD, sp->Inst))
      {lp->Hold(0);
       return Response.Send(kXR_NotFound, "session just closed");
      }

// Get the protocol associated with the link
//
   if (!(pp=dynamic_cast<XrdXrootdProtocol *>(lp->getProtocol()))||lp != pp->Link)
      {lp->Hold(0);
       return Response.Send(kXR_ArgInvalid, "session protocol not xroot");
      }

// Verify that the parent protocol is fully logged in
//
   if (!(pp->Status & XRD_LOGGEDIN) || (pp->Status & XRD_NEED_AUTH))
      {lp->Hold(0);
       return Response.Send(kXR_ArgInvalid, "session not logged in");
      }

// Verify that the bind is valid for the requestor
//
   if (sp->Pid != myPID || sp->Sid != pp->mySID)
      {lp->Hold(0);
       return Response.Send(kXR_ArgInvalid, "invalid session ID");
      }

// For now, verify that the request is comming from the same host
//
   if (strcmp(Link->Host(), lp->Host()))
      {lp->Hold(0);
       return Response.Send(kXR_NotAuthorized, "cross-host bind not allowed");
      }

// Find a slot for this path in parent protocol
//
   for (i = 1; i < maxStreams && pp->Stream[i]; i++) {}
   if (i >= maxStreams)
      {lp->Hold(0);
       return Response.Send(kXR_NoMemory, "bind limit exceeded");
      }

// Link this protocol to the parent
//
   pp->Stream[i] = this;
   Stream[0]     = pp;
   pp->isBound   = 1;
   PathID        = i;
   sprintf(buff, "FD %d#%d bound", Link->FDnum(), i);
   eDest.Log(SYS_LOG_01, "Xeq", buff, lp->ID);

// Construct a login name for this bind session
//
   cp = strdup(lp->ID);
   if ( (dp = rindex(cp, '@'))) *dp = '\0';
   if (!(dp = rindex(cp, '.'))) pPid = 0;
      else {*dp++ = '\0'; pPid = strtol(dp, (char **)NULL, 10);}
   Link->setID(cp, pPid);
   free(cp);
   CapVer = pp->CapVer;
   Status = XRD_BOUNDPATH;

// Get the required number of parallel I/O objects
//
   pioFree = XrdXrootdPio::Alloc(maxPio);

// There are no errors possible at this point unless the response fails
//
   buff[0] = static_cast<char>(i);
   if (!(rc = Response.Send(kXR_ok, buff, 1))) rc = -EINPROGRESS;

// Return but keep the link disabled
//
   lp->Hold(0);
   return rc;
}

/******************************************************************************/
/*                              d o _ c h m o d                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Chmod()
{
   int mode, rc;
   const char *opaque;
   XrdOucErrInfo myError(Link->ID);

// Check for static routing
//
   if (Route[RD_chmod].Port) 
      return Response.Send(kXR_redirect,Route[RD_chmod].Port,Route[RD_chmod].Host);

// Unmarshall the data
//
   mode = mapMode((int)ntohs(Request.chmod.mode));
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Modifying", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Modifying", argp->buff);

// Preform the actual function
//
   rc = osFS->chmod(argp->buff, (XrdSfsMode)mode, myError, CRED, opaque);
   TRACEP(FS, "chmod rc=" <<rc <<" mode=" <<std::oct <<mode <<std::dec <<' ' <<argp->buff);
   if (SFS_OK == rc) return Response.Send();

// An error occured
//
   return fsError(rc, myError);
}

/******************************************************************************/
/*                              d o _ C K s u m                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_CKsum(int canit)
{
   const char *opaque;
   char *args[3];

// Check if we support this operation
//
   if (!JobCKS)
      return Response.Send(kXR_Unsupported, "query chksum is not supported");

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Check summing", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Check summing", argp->buff);

// If this is a cancel request, do it now
//
   if (canit)
      {JobCKS->Cancel(argp->buff, &Response);
       return Response.Send();
      }

// Construct the argument list
//
   args[0] = JobCKT;
   args[1] = argp->buff;
   args[2] = 0;

// Preform the actual function
//
   return JobCKS->Schedule(argp->buff, (const char **)args, &Response,
                  ((CapVer & kXR_vermask) >= kXR_ver002 ? 0 : JOB_Sync));
}

/******************************************************************************/
/*                              d o _ C l o s e                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Close()
{
   XrdXrootdFile *fp;
   XrdXrootdFHandle fh(Request.close.fhandle);
   int rc;

// Keep statistics
//
   UPSTATS(miscCnt);

// Find the file object
//
   if (!FTab || !(fp = FTab->Get(fh.handle)))
      return Response.Send(kXR_FileNotOpen, 
                          "close does not refer to an open file");

// Serialize the link to make sure that any in-flight operations on this handle
// have completed (async mode or parallel streams)
//
   Link->Serialize();

// If we are monitoring, insert a close entry
//
   if (monFILE && Monitor) Monitor->Close(fp->FileID,fp->readCnt,fp->writeCnt);

// Do an explicit close of the file here; reflecting any errors
//
   rc = fp->XrdSfsp->close();
   TRACEP(FS, "close rc=" <<rc <<" fh=" <<fh.handle);
   if (SFS_OK != rc)
      return Response.Send(kXR_FSError, fp->XrdSfsp->error.getErrText());

// Delete the file from the file table; this will unlock/close the file
//
   FTab->Del(fh.handle);
   numFiles--;
   return Response.Send();
}

/******************************************************************************/
/*                            d o _ D i r l i s t                             */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Dirlist()
{
   int bleft, rc = 0, dlen, cnt = 0;
   char *buff, ebuff[4096];
   const char *opaque, *dname;
   XrdSfsDirectory *dp;

// Check for static routing
//
   if (Route[RD_dirlist].Port) 
      return Response.Send(kXR_redirect,Route[RD_dirlist].Port,Route[RD_dirlist].Host);

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Listing", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Listing", argp->buff);

// Get a directory object
//
   if (!(dp = osFS->newDir(Link->ID)))
      {snprintf(ebuff,sizeof(ebuff)-1,"Insufficient memory to open %s",argp->buff);
       eDest.Emsg("Xeq", ebuff);
       return Response.Send(kXR_NoMemory, ebuff);
      }

// First open the directory
//
   if ((rc = dp->open(argp->buff, CRED, opaque)))
      {rc = fsError(rc, dp->error); delete dp; return rc;}

// Start retreiving each entry and place in a local buffer with a trailing new
// line character (the last entry will have a null byte). If we cannot fit a
// full entry in the buffer, send what we have with an OKSOFAR and continue.
// This code depends on the fact that a directory entry will never be longer
// than sizeof( ebuff)-1; otherwise, an infinite loop will result. No errors
// are allowed to be reflected at this point.
//
  dname = 0;
  do {buff = ebuff; bleft = sizeof(ebuff);
      while(dname || (dname = dp->nextEntry()))
           {dlen = strlen(dname);
            if (dlen > 2 || dname[0] != '.' || (dlen == 2 && dname[1] != '.'))
               {if ((bleft -= (dlen+1)) < 0) break;
                strcpy(buff, dname); buff += dlen; *buff = '\n'; buff++; cnt++;
               }
            dname = 0;
           }
       if (dname) rc = Response.Send(kXR_oksofar, ebuff, buff-ebuff);
     } while(!rc && dname);

// Send the ending packet if we actually have one to send
//
   if (!rc) 
      {if (ebuff == buff) rc = Response.Send();
          else {*(buff-1) = '\0';
                rc = Response.Send((void *)ebuff, buff-ebuff);
               }
      }

// Close the directory
//
   dp->close();
   delete dp;
   if (!rc) {TRACEP(FS, "dirlist entries=" <<cnt <<" path=" <<argp->buff);}
   return rc;
}

/******************************************************************************/
/*                            d o _ E n d s e s s                             */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Endsess()
{
   XrdXrootdSessID *sp, sessID;
   int rc;

// Update misc stats count
//
   UPSTATS(miscCnt);

// Extract out the FD and Instance from the session ID
//
   sp = (XrdXrootdSessID *)Request.endsess.sessid;
   memcpy((void *)&sessID.Pid,  &sp->Pid,  sizeof(sessID.Pid));
   memcpy((void *)&sessID.FD,   &sp->FD,   sizeof(sessID.FD));
   memcpy((void *)&sessID.Inst, &sp->Inst, sizeof(sessID.Inst));

// Trace this request
//
   TRACEP(LOGIN, "endsess " <<sessID.Pid <<':' <<sessID.FD <<'.' <<sessID.Inst);

// If this session id does not refer to us, ignore the request
//
   if (sessID.Pid != myPID) return Response.Send();

// Terminate the indicated session, if possible. This could also be a self-termination.
//
   if ((sessID.FD == 0 && sessID.Inst == 0) 
   ||  !(rc = Link->Terminate(Link, sessID.FD, sessID.Inst))) return -1;

// Trace this request
//
   TRACEP(LOGIN, "endsess " <<sessID.Pid <<':' <<sessID.FD <<'.' <<sessID.Inst
          <<" rc=" <<rc <<" (" <<strerror(rc < 0 ? -rc : EAGAIN) <<")");

// Return result
//
   if (rc >  0)
      return (rc = Response.Send(kXR_wait, rc, "session still active")) ? rc:1;

   if (rc == -EACCES)return Response.Send(kXR_NotAuthorized, "not session owner");
   if (rc == -ESRCH) return Response.Send(kXR_NotFound, "session not found");
   if (rc == -ETIME) return Response.Send(kXR_Cancelled,"session not ended");

   return Response.Send();
}

/******************************************************************************/
/*                            d o   G e t f i l e                             */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Getfile()
{
   int gopts, buffsz;

// Keep Statistics
//
   UPSTATS(getfCnt);

// Unmarshall the data
//
   gopts  = int(ntohl(Request.getfile.options));
   buffsz = int(ntohl(Request.getfile.buffsz));

   return Response.Send(kXR_Unsupported, "getfile request is not supported");
}

/******************************************************************************/
/*                             d o _ L o c a t e                              */
/******************************************************************************/

int XrdXrootdProtocol::do_Locate()
{
   static XrdXrootdCallBack locCB("locate");
   int rc, opts, fsctl_cmd = SFS_FSCTL_LOCATE;
   const char *opaque;
   char *Path, *fn = argp->buff, opt[8], *op=opt;
   XrdOucErrInfo myError(Link->ID, &locCB, ReqID.getID());

// Unmarshall the data
//
   opts = (int)ntohs(Request.locate.options);

// Map the options
//
   if (opts & kXR_nowait)  {fsctl_cmd |= SFS_O_NOWAIT; *op++ = 'i';}
   if (opts & kXR_refresh) {fsctl_cmd |= SFS_O_RESET;  *op++ = 's';}
   *op = '\0';
   TRACEP(FS, "locate " <<opt <<' ' <<fn);

// Check for static routing
//
   if (Route[RD_locate].Port)
      return Response.Send(kXR_redirect, Route[RD_locate].Port,
                                         Route[RD_locate].Host);

// Check if this is a non-specific locate
//
        if (*fn != '*') Path = fn;
   else if (*(fn+1))    Path = fn+1;
   else                {Path = 0; 
                        fn = XPList.Next()->Path();
                        fsctl_cmd |= SFS_O_TRUNC;
                       }

// Prescreen the path
//
   if (Path)
      {if (rpCheck(Path, &opaque)) return rpEmsg("Locating", Path);
       if (!Squash(Path))          return vpEmsg("Locating", Path);
      }

// Preform the actual function
//
   rc = osFS->fsctl(fsctl_cmd, fn, myError, CRED);
   TRACEP(FS, "rc=" <<rc <<" locate " <<fn);
   return fsError(rc, myError);
}
  
/******************************************************************************/
/*                              d o _ L o g i n                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Login()
{
   static XrdSysMutex sessMutex;
   static unsigned int Sid = 0;
   XrdXrootdSessID sessID;
   int i, pid, rc, sendSID = 0;
   char uname[sizeof(Request.login.username)+1];

// Keep Statistics
//
   UPSTATS(miscCnt);

// Unmarshall the data
//
   pid = (int)ntohl(Request.login.pid);
   for (i = 0; i < (int)sizeof(Request.login.username); i++)
      {if (Request.login.username[i] == '\0' ||
           Request.login.username[i] == ' ') break;
       uname[i] = Request.login.username[i];
      }
   uname[i] = '\0';

// Make sure the user is not already logged in
//
   if (Status) return Response.Send(kXR_InvalidRequest,
                                    "duplicate login; already logged in");

// Establish the ID for this link
//
   Link->setID(uname, pid);
   CapVer = Request.login.capver[0];

// Establish the session ID if the client can handle it (protocol version > 0)
//
   if (CapVer && kXR_vermask)
      {sessID.FD   = Link->FDnum();
       sessID.Inst = Link->Inst();
       sessID.Pid  = myPID;
       sessMutex.Lock(); mySID = ++Sid; sessMutex.UnLock();
       sessID.Sid  = mySID;
       sendSID = 1;
      }

// Check if this is an admin login
//
   if (*(Request.login.role) & (kXR_char)kXR_useradmin)
      Status = XRD_ADMINUSER;

// Get the security token for this link. We will either get a token, a null
// string indicating host-only authentication, or a null indicating no
// authentication. We can then optimize of each case.
//
   if (CIA)
      {const char *pp=CIA->getParms(i, Link->Host());
       if (pp && i ) {if (!sendSID) rc = Response.Send((void *)pp, i);
                         else {struct iovec iov[3];
                               iov[1].iov_base = (char *)&sessID;
                               iov[1].iov_len  = sizeof(sessID);
                               iov[2].iov_base = (char *)pp;
                               iov[2].iov_len  = i;
                               rc = Response.Send(iov,3,int(i+sizeof(sessID)));
                              }
                      Status = (XRD_LOGGEDIN | XRD_NEED_AUTH);
                     }
          else {rc = (sendSID ? Response.Send((void *)&sessID, sizeof(sessID))
                              : Response.Send());
                Status = XRD_LOGGEDIN;
                if (pp) {Entity.tident = Link->ID; Client = &Entity;}
               }
      }
      else {rc = (sendSID ? Response.Send((void *)&sessID, sizeof(sessID))
                          : Response.Send());
            Status = XRD_LOGGEDIN;
           }

// Allocate a monitoring object, if needed for this connection
//
   if ((Monitor = XrdXrootdMonitor::Alloc()))
      {monFILE = XrdXrootdMonitor::monFILE;
       monIO   = XrdXrootdMonitor::monIO;
       if (XrdXrootdMonitor::monUSER)
          monUID = XrdXrootdMonitor::Map(XROOTD_MON_MAPUSER, Link->ID, 0);
      }

// Complete the rquestID object
//
   ReqID.setID(Request.header.streamid, Link->FDnum(), Link->Inst());

// Document this login
//
   if (!(Status & XRD_NEED_AUTH))
      eDest.Log(SYS_LOG_01, "Xeq", Link->ID, (Status & XRD_ADMINUSER
                            ? "admin login" : "login"));
   return rc;
}

/******************************************************************************/
/*                              d o _ M k d i r                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Mkdir()
{
   int mode, rc;
   const char *opaque;
   XrdOucErrInfo myError(Link->ID);

// Check for static routing
//
   if (Route[RD_mkdir].Port) 
      return Response.Send(kXR_redirect,Route[RD_mkdir].Port,Route[RD_mkdir].Host);

// Unmarshall the data
//
   mode = mapMode((int)ntohs(Request.mkdir.mode)) | S_IRWXU;
   if (Request.mkdir.options[0] & static_cast<unsigned char>(kXR_mkdirpath))
      mode |= SFS_O_MKPTH;
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Creating", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Creating", argp->buff);

// Preform the actual function
//
   rc = osFS->mkdir(argp->buff, (XrdSfsMode)mode, myError, CRED, opaque);
   TRACEP(FS, "rc=" <<rc <<" mkdir " <<std::oct <<mode <<std::dec <<' ' <<argp->buff);
   if (SFS_OK == rc) return Response.Send();

// An error occured
//
   return fsError(rc, myError);
}

/******************************************************************************/
/*                                 d o _ M v                                  */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Mv()
{
   int rc;
   const char *Opaque, *Npaque;
   char *oldp, *newp;
   XrdOucErrInfo myError(Link->ID);

// Check for static routing
//
   if (Route[RD_mv].Port) 
      return Response.Send(kXR_redirect,Route[RD_mv].Port,Route[RD_mv].Host);

// Find the space separator between the old and new paths
//
   oldp = newp = argp->buff;
   while(*newp && *newp != ' ') newp++;
   if (*newp) {*newp = '\0'; newp++;
               while(*newp && *newp == ' ') newp++;
              }

// Get rid of relative paths and multiple slashes
//
   if (rpCheck(oldp, &Opaque)) return rpEmsg("Renaming",    oldp);
   if (rpCheck(newp, &Npaque)) return rpEmsg("Renaming to", newp);
   if (!Squash(oldp))          return vpEmsg("Renaming",    oldp);
   if (!Squash(newp))          return vpEmsg("Renaming to", newp);

// Check if new path actually specified here
//
   if (*newp == '\0')
      Response.Send(kXR_ArgMissing, "new path specfied for mv");

// Preform the actual function
//
   rc = osFS->rename(oldp, newp, myError, CRED, Opaque, Npaque);
   TRACEP(FS, "rc=" <<rc <<" mv " <<oldp <<' ' <<newp);
   if (SFS_OK == rc) return Response.Send();

// An error occured
//
   return fsError(rc, myError);
}

/******************************************************************************/
/*                            d o _ O f f l o a d                             */
/******************************************************************************/

int XrdXrootdProtocol::do_Offload(int pathID, int isWrite)
{
   XrdSysSemaphore isAvail(0);
   XrdXrootdProtocol *pp;
   XrdXrootdPio      *pioP;
   kXR_char streamID[2];

// Verify that the path actually exists
//
   if (pathID >= maxStreams || !(pp = Stream[pathID]))
      return Response.Send(kXR_ArgInvalid, "invalid path ID");

// Verify that this path is still functional
//
   pp->streamMutex.Lock();
   if (pp->isDead || pp->isNOP)
      {pp->streamMutex.UnLock();
       return Response.Send(kXR_ArgInvalid, 
       (pp->isDead ? "path ID is not functional"
                   : "path ID is not connected"));
      }

// Grab the stream ID
//
   Response.StreamID(streamID);

// Try to schedule this operation. In order to maximize the I/O overlap, we
// will wait until the stream gets control and will have a chance to start
// reading from the device or from the network.
//
   do{if (!pp->isActive)
         {pp->myFile   = myFile;
          pp->myOffset = myOffset;
          pp->myIOLen  = myIOLen;
          pp->myBlen   = 0;
          pp->doWrite  = static_cast<char>(isWrite);
          pp->doWriteC = 0;
          pp->Resume   = &XrdXrootdProtocol::do_OffloadIO;
          pp->isActive = 1;
          pp->reTry    = &isAvail;
          pp->Response.Set(streamID);
          pp->streamMutex.UnLock();
          Link->setRef(1);
          Sched->Schedule((XrdJob *)(pp->Link));
          isAvail.Wait();
          return 0;
         }

      if ((pioP = pp->pioFree)) break;
      pp->reTry = &isAvail;
      pp->streamMutex.UnLock();
      TRACEP(FS, (isWrite ? 'w' : 'r') <<" busy path " <<pathID <<" offs=" <<myOffset);
      isAvail.Wait();
      TRACEP(FS, (isWrite ? 'w' : 'r') <<" free path " <<pathID <<" offs=" <<myOffset);
      pp->streamMutex.Lock();
      if (pp->isNOP)
         {pp->streamMutex.UnLock();
          return Response.Send(kXR_ArgInvalid, "path ID is not connected");
         }
      } while(1);

// Fill out the queue entry and add it to the queue
//
   pp->pioFree = pioP->Next; pioP->Next = 0;
   pioP->Set(myFile, myOffset, myIOLen, streamID, static_cast<char>(isWrite));
   if (pp->pioLast) pp->pioLast->Next = pioP;
      else          pp->pioFirst      = pioP;
   pp->pioLast = pioP;
   pp->streamMutex.UnLock();
   return 0;
}

/******************************************************************************/
/*                          d o _ O f f l o a d I O                           */
/******************************************************************************/

int XrdXrootdProtocol::do_OffloadIO()
{
   XrdSysSemaphore *sesSem;
   XrdXrootdPio    *pioP;
   int rc;

// Entry implies that we just got scheduled and are marked as active. Hence
// we need to post the session thread so that it can pick up the next request.
// We can manipulate the semaphore pointer without a lock as the only other
// thread that can manipulate the pointer is the waiting session thread.
//
   if (!doWriteC && (sesSem = reTry)) {reTry = 0; sesSem->Post();}
  
// Perform all I/O operations on a parallel stream (suppress async I/O).
//
   do {if (!doWrite) rc = do_ReadAll(0);
          else if ( (rc = (doWriteC ? do_WriteCont() : do_WriteAll()) ) > 0)
                  {Resume = &XrdXrootdProtocol::do_OffloadIO;
                   doWriteC = 1;
                   return rc;
                  }
       streamMutex.Lock();
       if (rc || !(pioP = pioFirst)) break;
       if (!(pioFirst = pioP->Next)) pioLast = 0;
       myFile   = pioP->myFile;
       myOffset = pioP->myOffset;
       myIOLen  = pioP->myIOLen;
       doWrite  = pioP->isWrite;
       doWriteC = 0;
       Response.Set(pioP->StreamID);
       pioP->Next = pioFree; pioFree = pioP;
       if (reTry) {reTry->Post(); reTry = 0;}
       streamMutex.UnLock();
      } while(1);

// There are no pending operations or the link died
//
   if (rc) isNOP = 1;
   isActive = 0;
   Stream[0]->Link->setRef(-1);
   if (reTry) {reTry->Post(); reTry = 0;}
   streamMutex.UnLock();
   return -EINPROGRESS;
}

/******************************************************************************/
/*                               d o _ O p e n                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Open()
{
   static XrdXrootdCallBack openCB("open file");
   int fhandle;
   int rc, mode, opts, openopts, mkpath = 0, doforce = 0, compchk = 0;
   int popt, retStat = 0;
   const char *opaque;
   char usage, ebuff[2048];
   char *fn = argp->buff, opt[16], *op=opt, isAsync = '\0';
   XrdSfsFile *fp;
   XrdXrootdFile *xp;
   struct stat statbuf;
   struct ServerResponseBody_Open myResp;
   int resplen = sizeof(myResp.fhandle);
   struct iovec IOResp[3];  // Note that IOResp[0] is completed by Response

// Keep Statistics
//
   UPSTATS(openCnt);

// Unmarshall the data
//
   mode = (int)ntohs(Request.open.mode);
   opts = (int)ntohs(Request.open.options);

// Map the mode and options
//
   mode = mapMode(mode) | S_IRUSR | S_IWUSR; usage = 'r';
        if (opts & kXR_open_read)  
           {openopts  = SFS_O_RDONLY;  *op++ = 'r';}
   else if (opts & kXR_open_updt)   
           {openopts  = SFS_O_RDWR;    *op++ = 'u'; usage = 'w';}
   else    {openopts  = SFS_O_RDONLY;  *op++ = 'r';}

        if (opts & kXR_new)
           {openopts |= SFS_O_CREAT;   *op++ = 'n';
            if (opts & kXR_mkdir)     {*op++ = 'm'; mkpath = 1;
                                       mode |= SFS_O_MKPTH;
                                      }
           }
   else if (opts & kXR_delete)
           {openopts  = SFS_O_TRUNC;   *op++ = 'd';
            if (opts & kXR_mkdir)     {*op++ = 'm'; mkpath = 1;
                                       mode |= SFS_O_MKPTH;
                                      }
           }
   if (opts & kXR_compress)        
           {openopts |= SFS_O_RAWIO;   *op++ = 'c'; compchk = 1;}
   if (opts & kXR_force)              {*op++ = 'f'; doforce = 1;}
   if ((opts & kXR_async || as_force) && !as_noaio)
                                      {*op++ = 'a'; isAsync = '1';}
   if (opts & kXR_refresh)            {*op++ = 's'; openopts |= SFS_O_RESET;
                                       UPSTATS(Refresh);
                                      }
   if (opts & kXR_retstat)            {*op++ = 't'; retStat = 1;}
   if (opts & kXR_posc)               {*op++ = 'p'; openopts |= SFS_O_POSC;}
   *op = '\0';
   TRACEP(FS, "open " <<opt <<' ' <<fn);

// Check if opaque data has been provided
//
   if (rpCheck(fn, &opaque)) return rpEmsg("Opening", fn);

// Check if static redirection applies
//
   if (Route[RD_open1].Host && (popt = RPList.Validate(fn)))
      return Response.Send(kXR_redirect,Route[popt].Port,Route[popt].Host);

// Validate the path
//
   if (!(popt = Squash(fn))) return vpEmsg("Opening", fn);

// Get a file object
//
   if (!(fp = osFS->newFile(Link->ID)))
      {snprintf(ebuff, sizeof(ebuff)-1,"Insufficient memory to open %s",fn);
       eDest.Emsg("Xeq", ebuff);
       return Response.Send(kXR_NoMemory, ebuff);
      }

// The open is elegible for a defered response, indicate we're ok with that
//
   fp->error.setErrCB(&openCB, ReqID.getID());

// Open the file
//
   if ((rc = fp->open(fn, (XrdSfsFileOpenMode)openopts,
                     (mode_t)mode, CRED, opaque)))
      {rc = fsError(rc, fp->error); delete fp; return rc;}

// Obtain a hyper file object
//
   if (!(xp=new XrdXrootdFile(Link->ID,fp,usage,isAsync,Link->sfOK,&statbuf)))
      {delete fp;
       snprintf(ebuff, sizeof(ebuff)-1, "Insufficient memory to open %s", fn);
       eDest.Emsg("Xeq", ebuff);
       return Response.Send(kXR_NoMemory, ebuff);
      }

// Serialize the link
//
   Link->Serialize();
   *ebuff = '\0';

// Lock this file
//
   if (!(popt & XROOTDXP_NOLK) && (rc = Locker->Lock(xp, doforce)))
      {const char *who;
       if (rc > 0) who = (rc > 1 ? "readers" : "reader");
          else {   rc = -rc;
                   who = (rc > 1 ? "writers" : "writer");
               }
       snprintf(ebuff, sizeof(ebuff)-1,
                "%s file %s is already opened by %d %s; open denied.",
                ('r' == usage ? "Input" : "Output"), fn, rc, who);
       delete fp; xp->XrdSfsp = 0; delete xp;
       eDest.Emsg("Xeq", ebuff);
       return Response.Send(kXR_FileLocked, ebuff);
      }

// Create a file table for this link if it does not have one
//
   if (!FTab) FTab = new XrdXrootdFileTable();

// Insert this file into the link's file table
//
   if (!FTab || (fhandle = FTab->Add(xp)) < 0)
      {delete xp;
       snprintf(ebuff, sizeof(ebuff)-1, "Insufficient memory to open %s", fn);
       eDest.Emsg("Xeq", ebuff);
       return Response.Send(kXR_NoMemory, ebuff);
      }

// Document forced opens
//
   if (doforce)
      {int rdrs, wrtrs;
       Locker->numLocks(xp, rdrs, wrtrs);
       if (('r' == usage && wrtrs) || ('w' == usage && rdrs) || wrtrs > 1)
          {snprintf(ebuff, sizeof(ebuff)-1,
             "%s file %s forced opened with %d reader(s) and %d writer(s).",
             ('r' == usage ? "Input" : "Output"), fn, rdrs, wrtrs);
           eDest.Emsg("Xeq", ebuff);
          }
      }

// Determine if file is compressed
//
   if (!compchk) 
      {resplen = sizeof(myResp.fhandle);
       memset(&myResp, 0, sizeof(myResp));
      }
      else {int cpsize;
            fp->getCXinfo((char *)myResp.cptype, cpsize);
            if (cpsize) {myResp.cpsize = static_cast<kXR_int32>(htonl(cpsize));
                         resplen = sizeof(myResp);
                        } else myResp.cpsize = 0;
           }

// If client wants a stat in open, return the stat information
//
   if (retStat)
      {retStat = StatGen(statbuf, ebuff);
       IOResp[1].iov_base = (char *)&myResp; IOResp[1].iov_len = sizeof(myResp);
       IOResp[2].iov_base =         ebuff;   IOResp[2].iov_len = retStat;
       resplen = sizeof(myResp) + retStat;
      }

// If we are monitoring, send off a path to dictionary mapping
//
   if (monFILE && Monitor) 
      {xp->FileID = Monitor->Map(XROOTD_MON_MAPPATH, Link->ID, fn);
       Monitor->Open(xp->FileID, statbuf.st_size);
      }

// Insert the file handle
//
   memcpy((void *)myResp.fhandle,(const void *)&fhandle,sizeof(myResp.fhandle));
   numFiles++;

// Respond
//
   if (retStat)  return Response.Send(IOResp, 3, resplen);
      else       return Response.Send((void *)&myResp, resplen);
}

/******************************************************************************/
/*                               d o _ P i n g                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Ping()
{

// Keep Statistics
//
   UPSTATS(miscCnt);

// This is a basic nop
//
   return Response.Send();
}

/******************************************************************************/
/*                            d o _ P r e p a r e                             */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Prepare()
{
   int rc, hport, pathnum = 0;
   const char *opaque;
   char opts, hname[32], reqid[64], nidbuff[512], *path;
   XrdOucErrInfo myError(Link->ID);
   XrdOucTokenizer pathlist(argp->buff);
   XrdOucTList *pFirst=0, *pP, *pLast = 0;
   XrdOucTList *oFirst=0, *oP, *oLast = 0;
   XrdOucTListHelper pHelp(&pFirst), oHelp(&oFirst);
   XrdXrootdPrepArgs pargs(0, 0);
   XrdSfsPrep fsprep;

// Grab the options
//
   opts = Request.prepare.options;

// Check for static routing
//
   if (Route[RD_prepstg].Port && ((opts & kXR_stage) || (opts & kXR_cancel)))
      return Response.Send(kXR_redirect,Route[RD_prepstg].Port,Route[RD_prepstg].Host);
   if (Route[RD_prepare].Port)
      return Response.Send(kXR_redirect,Route[RD_prepare].Port,Route[RD_prepare].Host);

// Get a request ID for this prepare and check for static routine
//
   if (opts & kXR_stage && !(opts & kXR_cancel)) 
      {XrdOucReqID::ID(reqid, sizeof(reqid)); 
       fsprep.opts = Prep_STAGE | (opts & kXR_coloc ? Prep_COLOC : 0);
      }
      else {reqid[0] = '*'; reqid[1] = '\0';  fsprep.opts = 0;}

// Initialize the fsile system prepare arg list
//
   fsprep.reqid   = reqid;
   fsprep.paths   = 0;
   fsprep.oinfo   = 0;
   fsprep.opts   |= Prep_PRTY0 | (opts & kXR_fresh ? Prep_FRESH : 0);
   fsprep.notify  = 0;

// Check if this is a cancel request
//
   if (opts & kXR_cancel)
      {if (!(path = pathlist.GetLine()))
          return Response.Send(kXR_ArgMissing, "Prepare requestid not specified");
       if (!XrdOucReqID::isMine(path, hport, hname, sizeof(hname)))
          {if (!hport) return Response.Send(kXR_ArgInvalid,
                             "Prepare requestid owned by an unknown server");
           TRACEI(REDIR, Response.ID() <<"redirecting to " << hname <<':' <<hport);
           return Response.Send(kXR_redirect, hport, hname);
          }
       fsprep.reqid = path;
       if (SFS_OK != (rc = osFS->prepare(fsprep, myError, CRED)))
          return fsError(rc, myError);
       rc = Response.Send();
       XrdXrootdPrepare::Logdel(path);
       return rc;
      }

// Cycle through all of the paths in the list
//
   while((path = pathlist.GetLine()))
        {if (rpCheck(path, &opaque)) return rpEmsg("Preparing", path);
         if (!Squash(path))          return vpEmsg("Preparing", path);
         pP = new XrdOucTList(path, pathnum);
         (pLast ? (pLast->next = pP) : (pFirst = pP)); pLast = pP;
         oP = new XrdOucTList(opaque, 0);
         (oLast ? (oLast->next = oP) : (oFirst = oP)); oLast = oP;
         pathnum++;
        }

// Make sure we have at least one path
//
   if (!pFirst)
      return Response.Send(kXR_ArgMissing, "No prepare paths specified");

// Issue the prepare
//
   if (opts & kXR_notify)
      {fsprep.notify  = nidbuff;
       sprintf(nidbuff, Notify, Link->FDnum(), Link->ID);
       fsprep.opts = (opts & kXR_noerrs ? Prep_SENDAOK : Prep_SENDACK);
      }
   if (opts & kXR_wmode) fsprep.opts |= Prep_WMODE;
   fsprep.paths = pFirst;
   fsprep.oinfo = oFirst;
   if (SFS_OK != (rc = osFS->prepare(fsprep, myError, CRED)))
      return fsError(rc, myError);

// Perform final processing
//
   if (!(opts & kXR_stage)) rc = Response.Send();
      else {rc = Response.Send(reqid, strlen(reqid));
            pargs.reqid=reqid;
            pargs.user=Link->ID;
            pargs.paths=pFirst;
            XrdXrootdPrepare::Log(pargs);
            pargs.reqid = 0;
           }
   return rc;
}
  
/******************************************************************************/
/*                           d o _ P r o t o c o l                            */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Protocol()
{
    static ServerResponseBody_Protocol Resp
                 = {static_cast<kXR_int32>(htonl(XROOTD_VERSBIN)),
                    static_cast<kXR_int32>(htonl(kXR_DataServer))};

// Keep Statistics
//
   UPSTATS(miscCnt);

// Return info
//
    if (isRedir) Resp.flags = static_cast<kXR_int32>(htonl(kXR_LBalServer));
    return Response.Send((void *)&Resp, sizeof(Resp));
}

/******************************************************************************/
/*                            d o _ P u t f i l e                             */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Putfile()
{
   int popts, buffsz;

// Keep Statistics
//
   UPSTATS(putfCnt);

// Unmarshall the data
//
   popts  = int(ntohl(Request.putfile.options));
   buffsz = int(ntohl(Request.putfile.buffsz));

   return Response.Send(kXR_Unsupported, "putfile request is not supported");
}

/******************************************************************************/
/*                              d o _ Q c o n f                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Qconf()
{
   XrdOucTokenizer qcargs(argp->buff);
   char *val, buff[1024], *bp=buff;
   int n, bleft = sizeof(buff);

// Get the first argument
//
   if (!qcargs.GetLine() || !(val = qcargs.GetToken()))
      return Response.Send(kXR_ArgMissing, "query config argument not specified.");

// Trace this query variable
//
   do {TRACEP(DEBUG, "query config " <<val);
       if (bleft < 32) break;

   // Now determine what the user wants to query
   //
        if (!strcmp("bind_max", val))
           {n = sprintf(bp, "%d\n", maxStreams-1);
            bp += n; bleft -= n;
           }
   else if (!strcmp("pio_max", val))
           {n = sprintf(bp, "%d\n", maxPio+1);
            bp += n; bleft -= n;
           }
   else if (!strcmp("readv_ior_max", val))
           {n = sprintf(bp, "%d\n", maxTransz - (int)sizeof(readahead_list));
            bp += n; bleft -= n;
           }
   else if (!strcmp("readv_iov_max", val)) 
           {n = sprintf(bp, "%d\n", maxRvecsz);
            bp += n; bleft -= n;
           }
   else if (!strcmp("wan_port", val) && WANPort)
           {n = sprintf(bp, "%d\n", WANPort);
            bp += n; bleft -= n;
           }
   else if (!strcmp("wan_window", val) && WANPort)
           {n = sprintf(bp, "%d\n", WANWindow);
            bp += n; bleft -= n;
           }
   else if (!strcmp("window", val) && Window)
           {n = sprintf(bp, "%d\n", Window);
            bp += n; bleft -= n;
           }
   else {n = strlen(val);
         if (bleft <= n) break;
         strcpy(bp, val); bp +=n; *bp = '\n'; bp++;
         bleft -= (n+1);
        }
   } while((val = qcargs.GetToken()));

// Make sure all ended well
//
   if (val) 
      return Response.Send(kXR_ArgTooLong, "too many query config arguments.");

// All done
//
   return Response.Send(buff, sizeof(buff) - bleft);
}
  
/******************************************************************************/
/*                                d o _ Q f h                                 */
/******************************************************************************/

int XrdXrootdProtocol::do_Qfh()
{
   static const int fsctl_cmd1 = SFS_FCTL_STATV;
   static XrdXrootdCallBack qryCB("query");
   XrdOucErrInfo myError(Link->ID, &qryCB, ReqID.getID());
   XrdXrootdFHandle fh(Request.query.fhandle);
   XrdXrootdFile *fp;
   short qopt = (short)ntohs(Request.query.infotype);
   int rc, fsctl_cmd;

// Update misc stats count
//
   UPSTATS(miscCnt);

// Perform the appropriate query
//
   switch(qopt)
         {case kXR_Qvisa:   fsctl_cmd = fsctl_cmd1;
                            break;
          default:          return Response.Send(kXR_ArgMissing, 
                                   "Required query argument not present");
         }

// Find the file object
//
   if (!FTab || !(fp = FTab->Get(fh.handle)))
      return Response.Send(kXR_FileNotOpen,
                           "query does not refer to an open file");

// Preform the actual function
//
   rc = fp->XrdSfsp->fctl(fsctl_cmd, 0, myError);
   TRACEP(FS, "query rc=" <<rc <<" fh=" <<fh.handle);

// Return appropriately
//
   if (SFS_OK != rc) return fsError(rc, myError);
   return Response.Send();
}
  
/******************************************************************************/
/*                            d o _ Q o p a q u e                             */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Qopaque(short qopt)
{
   XrdOucErrInfo myError(Link->ID);
   XrdSfsFSctl myData;
   const char *opaque, *Act, *AData;
   int fsctl_cmd, rc, dlen = Request.query.dlen;

// Process unstructured as well as structured (path/opaque) requests
//
   if (qopt == kXR_Qopaque)
      {myData.Arg1 = argp->buff; myData.Arg1Len = dlen;
       myData.Arg2 = 0;          myData.Arg1Len = 0;
       fsctl_cmd = SFS_FSCTL_PLUGIO;
       Act = " qopaque '"; AData = "...";
      } else {
       // Check for static routing (this falls under stat)
       //
       if (Route[RD_stat].Port)
          return Response.Send(kXR_redirect,Route[RD_stat].Port,Route[RD_stat].Host);

       // Prescreen the path
       //
       if (rpCheck(argp->buff, &opaque)) return rpEmsg("Querying", argp->buff);
       if (!Squash(argp->buff))          return vpEmsg("Querying", argp->buff);

       // Setup arguments
       //
       myData.Arg1    = argp->buff;
       myData.Arg1Len = (opaque ? opaque - argp->buff - 1    : dlen);
       myData.Arg2    = opaque;
       myData.Arg2Len = (opaque ? argp->buff + dlen - opaque : 0);
       fsctl_cmd = SFS_FSCTL_PLUGIN;
       Act = " qopaquf '"; AData = argp->buff;
      }

// Preform the actual function using the supplied arguments
//
   rc = osFS->FSctl(fsctl_cmd, myData, myError, CRED);
   TRACEP(FS, "rc=" <<rc <<Act <<AData <<"'");
   if (rc == SFS_OK) Response.Send("");
   return fsError(rc, myError);
}

/******************************************************************************/
/*                             d o _ Q s p a c e                              */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Qspace()
{
   static const int fsctl_cmd = SFS_FSCTL_STATLS;
   XrdOucErrInfo myError(Link->ID);
   const char *opaque;
   int n, rc;

// Check for static routing
//
   if (Route[RD_stat].Port) 
      return Response.Send(kXR_redirect,Route[RD_stat].Port,Route[RD_stat].Host);

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Stating", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Stating", argp->buff);

// Add back the opaque info
//
   if (opaque)
      {n = strlen(argp->buff); argp->buff[n] = '?';
       if ((argp->buff)+n != opaque-1) strcpy(&argp->buff[n+1], opaque);
      }

// Preform the actual function using the supplied logical FS name
//
   rc = osFS->fsctl(fsctl_cmd, argp->buff, myError, CRED);
   TRACEP(FS, "rc=" <<rc <<" qspace '" <<argp->buff <<"'");
   if (rc == SFS_OK) Response.Send("");
   return fsError(rc, myError);
}

/******************************************************************************/
/*                              d o _ Q u e r y                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Query()
{
    short qopt = (short)ntohs(Request.query.infotype);

// Perform the appropriate query
//
   switch(qopt)
         {case kXR_QStats: return SI->Stats(Response,
                              (Request.header.dlen ? argp->buff : "a"));
          case kXR_Qcksum:  return do_CKsum(0);
          case kXR_Qckscan: return do_CKsum(1);
          case kXR_Qconfig: return do_Qconf();
          case kXR_Qspace:  return do_Qspace();
          case kXR_Qxattr:  return do_Qxattr();
          case kXR_Qopaque:
          case kXR_Qopaquf: return do_Qopaque(qopt);
          default:          break;
         }

// Whatever we have, it's not valid
//
   return Response.Send(kXR_ArgInvalid, 
                        "Invalid information query type code");
}

/******************************************************************************/
/*                             d o _ Q x a t t r                              */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Qxattr()
{
   static XrdXrootdCallBack statCB("stat");
   static const int fsctl_cmd = SFS_FSCTL_STATXA;
   int rc;
   const char *opaque;
   XrdOucErrInfo myError(Link->ID, &statCB, ReqID.getID());

// Check for static routing
//
   if (Route[RD_stat].Port) 
      return Response.Send(kXR_redirect,Route[RD_stat].Port,Route[RD_stat].Host);

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Stating", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Stating", argp->buff);

// Preform the actual function
//
   rc = osFS->fsctl(fsctl_cmd, argp->buff, myError, CRED);
   TRACEP(FS, "rc=" <<rc <<" qxattr " <<argp->buff);
   return fsError(rc, myError);
}
  
/******************************************************************************/
/*                               d o _ R e a d                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Read()
{
   int pathID, retc;
   XrdXrootdFHandle fh(Request.read.fhandle);
   numReads++;

// We first handle the pre-read list, if any. We do it this way because of
// a historical glitch in the protocol. One should really not piggy back a
// pre-read on top of a read, though it is allowed.
//
   if (!Request.header.dlen) pathID = 0;
      else if (do_ReadNone(retc, pathID)) return retc;

// Unmarshall the data
//
   myIOLen  = ntohl(Request.read.rlen);
              n2hll(Request.read.offset, myOffset);

// Find the file object
//
   if (!FTab || !(myFile = FTab->Get(fh.handle)))
      return Response.Send(kXR_FileNotOpen,
                           "read does not refer to an open file");

// Short circuit processing is read length is zero
//
   TRACEP(FS, pathID <<" fh=" <<fh.handle <<" read " <<myIOLen <<'@' <<myOffset);
   if (!myIOLen) return Response.Send();

// If we are monitoring, insert a read entry
//
   if (monIO && Monitor) Monitor->Add_rd(myFile->FileID, Request.read.rlen,
                                         Request.read.offset);

// See if an alternate path is required, offload the read
//
   if (pathID) return do_Offload(pathID, 0);

// Now read all of the data (do pre-reads first)
//
   return do_ReadAll();
}

/******************************************************************************/
/*                            d o _ R e a d A l l                             */
/******************************************************************************/

// myFile   = file to be read
// myOffset = Offset at which to read
// myIOLen  = Number of bytes to read from file and write to socket
  
int XrdXrootdProtocol::do_ReadAll(int asyncOK)
{
   int rc, xframt, Quantum = (myIOLen > maxBuffsz ? maxBuffsz : myIOLen);
   char *buff;

// If this file is memory mapped, short ciruit all the logic and immediately
// transfer the requested data to minimize latency.
//
   if (myFile->isMMapped)
      {     if (myOffset >= myFile->fSize) return Response.Send();
       else if (myOffset+myIOLen <= myFile->fSize)
               return Response.Send(myFile->mmAddr+myOffset, myIOLen);
       else    return Response.Send(myFile->mmAddr+myOffset,
                                    myFile->fSize -myOffset);
      }

// If we are sendfile enabled, then just send the file if possible
//
   if (myFile->sfEnabled && myIOLen >= as_minsfsz
   &&  myOffset+myIOLen <= myFile->fSize)
      return Response.Send(myFile->fdNum, myOffset, myIOLen);

// If we are in async mode, schedule the read to ocur asynchronously
//
   if (asyncOK && myFile->AsyncMode)
      {if (myIOLen >= as_miniosz && Link->UseCnt() < as_maxperlnk)
          if ((rc = aio_Read()) != -EAGAIN) return rc;
       SI->AsyncRej++;
      }

// Make sure we have a large enough buffer
//
   if (!argp || Quantum < halfBSize || Quantum > argp->bsize)
      {if ((rc = getBuff(1, Quantum)) <= 0) return rc;}
      else if (hcNow < hcNext) hcNow++;
   buff = argp->buff;

// Now read all of the data
//
   do {if ((xframt = myFile->XrdSfsp->read(myOffset, buff, Quantum)) <= 0) break;
       myFile->readCnt += xframt;
       if (xframt >= myIOLen) return Response.Send(buff, xframt);
       if (Response.Send(kXR_oksofar, buff, xframt) < 0) return -1;
       myOffset += xframt; myIOLen -= xframt;
       if (myIOLen < Quantum) Quantum = myIOLen;
      } while(myIOLen);

// Determine why we ended here
//
   if (xframt == 0) return Response.Send();
   return Response.Send(kXR_FSError, myFile->XrdSfsp->error.getErrText());
}

/******************************************************************************/
/*                           d o _ R e a d N o n e                            */
/******************************************************************************/
  
int XrdXrootdProtocol::do_ReadNone(int &retc, int &pathID)
{
   XrdXrootdFHandle fh;
   int ralsz = Request.header.dlen;
   struct read_args *rargs=(struct read_args *)(argp->buff);
   struct readahead_list *ralsp = (readahead_list *)(rargs+sizeof(read_args));

// Return the pathid
//
   pathID = static_cast<int>(rargs->pathid);
   if ((ralsz -= sizeof(read_args)) <= 0) return 0;

// Make sure that we have a proper pre-read list
//
   if (ralsz%sizeof(readahead_list))
      {Response.Send(kXR_ArgInvalid, "Invalid length for read ahead list");
       return 1;
      }

// Run down the pre-read list
//
   while(ralsz > 0)
        {myIOLen  = ntohl(ralsp->rlen);
                    n2hll(ralsp->offset, myOffset);
         memcpy((void *)&fh.handle, (const void *)ralsp->fhandle,
                  sizeof(fh.handle));
         TRACEP(FS, "fh=" <<fh.handle <<" read " <<myIOLen <<'@' <<myOffset);
         if (!FTab || !(myFile = FTab->Get(fh.handle)))
            {retc = Response.Send(kXR_FileNotOpen,
                             "preread does not refer to an open file");
             return 1;
            }
         myFile->XrdSfsp->read(myOffset, myIOLen);
         ralsz -= sizeof(struct readahead_list);
         ralsp++;
         numReads++;
        };

// All done
//
   return 0;
}

/******************************************************************************/
/*                               d o _ R e a d V                              */
/******************************************************************************/
  
int XrdXrootdProtocol::do_ReadV()
{
// This will read multiple buffers at the same time in an attempt to avoid
// the latency in a network. The information with the offsets and lengths
// of the information to read is passed as a data buffer... then we decode
// it and put all the individual buffers in a single one (it's up to the)
// client to interpret it. Code originally developed by Leandro Franco, CERN.
//
   const int hdrSZ     = sizeof(readahead_list);
   XrdXrootdFHandle currFH, lastFH((kXR_char *)"\xff\xff\xff\xff");
   struct readahead_list rdVec[maxRvecsz];
   long long totLen;
   int rdVecNum, rdVecLen = Request.header.dlen;
   int i, rc, xframt, Quantum, Qleft;
   char *buffp;

// Compute number of elements in the read vector and make sure we have no
// partial elements.
//
   rdVecNum = rdVecLen / sizeof(readahead_list);
   if ( (rdVecLen <= 0) || (rdVecNum*hdrSZ != rdVecLen) )
      {Response.Send(kXR_ArgInvalid, "Read vector is invalid");
       return 0;
      }

// Make sure that we can copy the read vector to our local stack. We must impose 
// a limit on it's size. We do this to be able to reuse the data buffer to 
// prevent cross-cpu memory cache synchronization.
//
   if (rdVecLen > static_cast<int>(sizeof(rdVec)))
      {Response.Send(kXR_ArgTooLong, "Read vector is too long");
       return 0;
      }
   memcpy(rdVec, argp->buff, rdVecLen);

// Run down the list and compute the total size of the read. No individual
// read may be greater than the maximum transfer size.
//
   totLen = rdVecLen; xframt = maxTransz - hdrSZ;
   for (i = 0; i < rdVecNum; i++) 
       {totLen += (rdVec[i].rlen = ntohl(rdVec[i].rlen));
        if (rdVec[i].rlen > xframt)
           {Response.Send(kXR_NoMemory, "Single readv transfer is too large");
            return 0;
           }
       }

// We limit the total size of the read to be 2GB for convenience
//
   if (totLen > 0x7fffffffLL)
      {Response.Send(kXR_NoMemory, "Total readv transfer is too large");
       return 0;
      }
   if ((Quantum = static_cast<int>(totLen)) > maxTransz) Quantum = maxTransz;
   
// Now obtain the right size buffer
//
   if ((Quantum < halfBSize && Quantum > 1024) || Quantum > argp->bsize)
      {if ((rc = getBuff(1, Quantum)) <= 0) return rc;}
      else if (hcNow < hcNext) hcNow++;

// Check that we really have at least one file open. This needs to be done 
// only once as this code runs in the control thread.
//
   if (!FTab) return Response.Send(kXR_FileNotOpen,
                              "readv does not refer to an open file");

// Run down the pre-read list. Each read element is prefixed by the verctor
// element. We also break the reads into Quantum sized units. We do the
//
   Qleft = Quantum; buffp = argp->buff;
   for (i = 0; i < rdVecNum; i++)
       {
        // Every request could come from a different file
        //
        currFH.Set(rdVec[i].fhandle);
        if (currFH.handle != lastFH.handle)
           {if (!(myFile = FTab->Get(currFH.handle)))
               return Response.Send(kXR_FileNotOpen,
                               "readv does not refer to an open file");
               else lastFH.handle = currFH.handle;
           }
      
        // Read in the vector, segmenting as needed. Note that we gaurantee
        // that a single readv element will never need to be segmented.
        //
        myIOLen  = rdVec[i].rlen;
        n2hll(rdVec[i].offset, myOffset);
        if (Qleft < (myIOLen + hdrSZ))
           {if (Response.Send(kXR_oksofar,argp->buff,Quantum-Qleft) < 0)
               return -1;
            Qleft = Quantum;
            buffp = argp->buff;
           }
        TRACEP(FS,"fh=" <<currFH.handle <<" readV " << myIOLen <<'@' <<myOffset);
        if ((xframt = myFile->XrdSfsp->read(myOffset,buffp+hdrSZ,myIOLen)) < 0)
           break;
        myFile->readCnt += xframt; numReads++;
        rdVec[i].rlen = htonl(xframt);
        memcpy(buffp, &rdVec[i], hdrSZ);
        buffp += (xframt+hdrSZ); Qleft -= (xframt+hdrSZ);
       }
   
// Determine why we ended here
//
   if (i >= rdVecNum)
      return Response.Send(argp->buff, Quantum-Qleft);
   return Response.Send(kXR_FSError, myFile->XrdSfsp->error.getErrText());
}
  
/******************************************************************************/
/*                                 d o _ R m                                  */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Rm()
{
   int rc;
   const char *opaque;
   XrdOucErrInfo myError(Link->ID);

// Check for static routing
//
   if (Route[RD_rm].Port) 
      return Response.Send(kXR_redirect,Route[RD_rm].Port,Route[RD_rm].Host);

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Removing", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Removing", argp->buff);

// Preform the actual function
//
   rc = osFS->rem(argp->buff, myError, CRED, opaque);
   TRACEP(FS, "rc=" <<rc <<" rm " <<argp->buff);
   if (SFS_OK == rc) return Response.Send();

// An error occured
//
   return fsError(rc, myError);
}

/******************************************************************************/
/*                              d o _ R m d i r                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Rmdir()
{
   int rc;
   const char *opaque;
   XrdOucErrInfo myError(Link->ID);

// Check for static routing
//
   if (Route[RD_rmdir].Port) 
      return Response.Send(kXR_redirect,Route[RD_rmdir].Port,Route[RD_rmdir].Host);

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Removing", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Removing", argp->buff);

// Preform the actual function
//
   rc = osFS->remdir(argp->buff, myError, CRED, opaque);
   TRACEP(FS, "rc=" <<rc <<" rmdir " <<argp->buff);
   if (SFS_OK == rc) return Response.Send();

// An error occured
//
   return fsError(rc, myError);
}

/******************************************************************************/
/*                                d o _ S e t                                 */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Set()
{
   XrdOucTokenizer setargs(argp->buff);
   char *val, *rest;

// Get the first argument
//
   if (!setargs.GetLine() || !(val = setargs.GetToken(&rest)))
      return Response.Send(kXR_ArgMissing, "set argument not specified.");

// Trace this set
//
   TRACEP(DEBUG, "set " <<val <<' ' <<rest);

// Now determine what the user wants to set
//
        if (!strcmp("appid", val))
           {while(*rest && *rest == ' ') rest++;
            eDest.Emsg("Xeq", Link->ID, "appid", rest);
            return Response.Send();
           }
   else if (!strcmp("monitor", val)) return do_Set_Mon(setargs);

// All done
//
   return Response.Send(kXR_ArgInvalid, "invalid set parameter");
}

/******************************************************************************/
/*                            d o _ S e t _ M o n                             */
/******************************************************************************/

// Process: set monitor {off | on} [appid] | info [info]}

int XrdXrootdProtocol::do_Set_Mon(XrdOucTokenizer &setargs)
{
  char *val, *appid;
  kXR_unt32 myseq = 0;

// Get the first argument
//
   if (!(val = setargs.GetToken(&appid)))
      return Response.Send(kXR_ArgMissing,"set monitor argument not specified.");

// For info requests, nothing changes. However, info events must have been
// enabled for us to record them. Route the information via the static
// monitor entry, since it knows how to forward the information.
//
   if (!strcmp(val, "info"))
      {if (appid && XrdXrootdMonitor::monINFO)
          {while(*appid && *appid == ' ') appid++;
           if (strlen(appid) > 1024) appid[1024] = '\0';
           if (*appid) myseq = XrdXrootdMonitor::Map(XROOTD_MON_MAPINFO,
                               Link->ID, appid);
          }
       return Response.Send((void *)&myseq, sizeof(myseq));
      }

// Determine if on do appropriate processing
//
   if (!strcmp(val, "on"))
      {if (Monitor || (Monitor = XrdXrootdMonitor::Alloc(1)))
          {if (appid && XrdXrootdMonitor::monIO)
              {while(*appid && *appid == ' ') appid++;
               if (*appid) Monitor->appID(appid);
              }
           monIO   =  XrdXrootdMonitor::monIO;
           monFILE =  XrdXrootdMonitor::monFILE;
           if (XrdXrootdMonitor::monUSER && !monUID)
              monUID = XrdXrootdMonitor::Map(XROOTD_MON_MAPUSER, Link->ID, 0);
          }
       return Response.Send();
      }

// Determine if off and do appropriate processing
//
   if (!strcmp(val, "off"))
      {if (Monitor)
          {if (appid && XrdXrootdMonitor::monIO)
              {while(*appid && *appid == ' ') appid++;
               if (*appid) Monitor->appID(appid);
              }
           Monitor->unAlloc(Monitor); Monitor = 0; monIO = monFILE = 0;
          }
       return Response.Send();
      }

// Improper request
//
   return Response.Send(kXR_ArgInvalid, "invalid set monitor argument");
}
  
/******************************************************************************/
/*                               d o _ S t a t                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Stat()
{
   static XrdXrootdCallBack statCB("stat");
   static const int fsctl_cmd = SFS_FSCTL_STATFS;
   int rc;
   const char *opaque;
   char xxBuff[256];
   struct stat buf;
   XrdOucErrInfo myError(Link->ID, &statCB, ReqID.getID());

// Check for static routing
//
   if (Route[RD_stat].Port) 
      return Response.Send(kXR_redirect,Route[RD_stat].Port,Route[RD_stat].Host);

// Prescreen the path
//
   if (rpCheck(argp->buff, &opaque)) return rpEmsg("Stating", argp->buff);
   if (!Squash(argp->buff))          return vpEmsg("Stating", argp->buff);

// Preform the actual function
//
   if (Request.stat.options & kXR_vfs)
      {rc = osFS->fsctl(fsctl_cmd, argp->buff, myError, CRED);
       TRACEP(FS, "rc=" <<rc <<" statfs " <<argp->buff);
       if (rc == SFS_OK) Response.Send("");
      } else {
       rc = osFS->stat(argp->buff, &buf, myError, CRED, opaque);
       TRACEP(FS, "rc=" <<rc <<" stat " <<argp->buff);
       if (rc == SFS_OK) return Response.Send(xxBuff, StatGen(buf, xxBuff));
      }
   return fsError(rc, myError);
}

/******************************************************************************/
/*                              d o _ S t a t x                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Statx()
{
   static XrdXrootdCallBack statxCB("xstat");
   int rc;
   const char *opaque;
   char *path, *respinfo = argp->buff;
   mode_t mode;
   XrdOucErrInfo myError(Link->ID, &statxCB, ReqID.getID());
   XrdOucTokenizer pathlist(argp->buff);

// Cycle through all of the paths in the list
//
   while((path = pathlist.GetLine()))
        {if (rpCheck(path, &opaque)) return rpEmsg("Stating", path);
         if (!Squash(path))          return vpEmsg("Stating", path);
         rc = osFS->stat(path, mode, myError, CRED, opaque);
         TRACEP(FS, "rc=" <<rc <<" stat " <<path);
         if (rc != SFS_OK)                    return fsError(rc, myError);
            else {if (mode == (mode_t)-1)    *respinfo = (char)kXR_offline;
                     else if (S_ISDIR(mode)) *respinfo = (char)kXR_isDir;
                             else            *respinfo = (char)kXR_file;
                 }
         respinfo++;
        }

// Return result
//
   return Response.Send(argp->buff, respinfo-argp->buff);
}

/******************************************************************************/
/*                               d o _ S y n c                                */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Sync()
{
   int rc;
   XrdXrootdFile *fp;
   XrdXrootdFHandle fh(Request.sync.fhandle);

// Keep Statistics
//
   UPSTATS(syncCnt);

// Find the file object
//
   if (!FTab || !(fp = FTab->Get(fh.handle)))
      return Response.Send(kXR_FileNotOpen,"sync does not refer to an open file");

// Sync the file
//
   rc = fp->XrdSfsp->sync();
   TRACEP(FS, "sync rc=" <<rc <<" fh=" <<fh.handle);
   if (SFS_OK != rc)
      return Response.Send(kXR_FSError, fp->XrdSfsp->error.getErrText());

// Respond that all went well
//
   return Response.Send();
}

/******************************************************************************/
/*                           d o _ T r u n c a t e                            */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Truncate()
{
   XrdXrootdFile *fp;
   XrdXrootdFHandle fh(Request.truncate.fhandle);
   long long theOffset;
   int rc;

// Unmarshall the data
//
   n2hll(Request.truncate.offset, theOffset);

// Check if this is a truncate for an open file (no path given)
//
   if (!Request.header.dlen)
      {
       // Update misc stats count
       //
          UPSTATS(miscCnt);

      // Find the file object
      //
         if (!FTab || !(fp = FTab->Get(fh.handle)))
            return Response.Send(kXR_FileNotOpen,
                                     "trunc does not refer to an open file");

     // Truncate the file
     //
        rc = fp->XrdSfsp->truncate(theOffset);
        TRACEP(FS, "trunc rc=" <<rc <<" sz=" <<theOffset <<" fh=" <<fh.handle);
        if (SFS_OK != rc)
           return Response.Send(kXR_FSError, fp->XrdSfsp->error.getErrText());

   } else {

       XrdOucErrInfo myError(Link->ID);
       const char *opaque;

    // Verify the path and extract out the opaque information
    //
       if (rpCheck(argp->buff,&opaque)) return rpEmsg("Truncating",argp->buff);
       if (!Squash(argp->buff))         return vpEmsg("Truncating",argp->buff);

    // Preform the actual function
    //
       rc = osFS->truncate(argp->buff, (XrdSfsFileOffset)theOffset, myError,
                           CRED, opaque);
       TRACEP(FS, "rc=" <<rc <<" trunc " <<theOffset <<' ' <<argp->buff);
       if (SFS_OK != rc) return fsError(rc, myError);
   }

// Respond that all went well
//
   return Response.Send();
}
  
/******************************************************************************/
/*                              d o _ W r i t e                               */
/******************************************************************************/
  
int XrdXrootdProtocol::do_Write()
{
   int retc, pathID;
   XrdXrootdFHandle fh(Request.write.fhandle);
   numWrites++;

// Unmarshall the data
//
   myIOLen  = Request.header.dlen;
              n2hll(Request.write.offset, myOffset);
   pathID   = static_cast<int>(Request.write.pathid);

// Find the file object
//
   if (!FTab || !(myFile = FTab->Get(fh.handle)))
      {if (argp) return do_WriteNone();
       Response.Send(kXR_FileNotOpen,"write does not refer to an open file");
       return Link->setEtext("write protcol violation");
      }

// If we are monitoring, insert a write entry
//
   if (monIO && Monitor) Monitor->Add_wr(myFile->FileID, Request.write.dlen,
                                         Request.write.offset);

// If zero length write, simply return
//
   TRACEP(FS, "fh=" <<fh.handle <<" write " <<myIOLen <<'@' <<myOffset);
   if (myIOLen <= 0) return Response.Send();

// See if an alternate path is required
//
   if (pathID) return do_Offload(pathID, 1);

// If we are in async mode, schedule the write to occur asynchronously
//
   if (myFile->AsyncMode && !as_syncw)
      {if (myStalls > as_maxstalls) myStalls--;
          else if (myIOLen >= as_miniosz && Link->UseCnt() < as_maxperlnk)
                  {if ((retc = aio_Write()) != -EAGAIN)
                      {if (retc == -EIO) return do_WriteNone();
                          else return retc;
                      }
                  }
       SI->AsyncRej++;
      }

// Just to the i/o now
//
   myFile->writeCnt += myIOLen; // Optimistically correct
   return do_WriteAll();
}
  
/******************************************************************************/
/*                           d o _ W r i t e A l l                            */
/******************************************************************************/

// myFile   = file to be written
// myOffset = Offset at which to write
// myIOLen  = Number of bytes to read from socket and write to file
  
int XrdXrootdProtocol::do_WriteAll()
{
   int rc, Quantum = (myIOLen > maxBuffsz ? maxBuffsz : myIOLen);

// Make sure we have a large enough buffer
//
   if (!argp || Quantum < halfBSize || Quantum > argp->bsize)
      {if ((rc = getBuff(0, Quantum)) <= 0) return rc;}
      else if (hcNow < hcNext) hcNow++;

// Now write all of the data (XrdXrootdProtocol.C defines getData())
//
   while(myIOLen > 0)
        {if ((rc = getData("data", argp->buff, Quantum)))
            {if (rc > 0) 
                {Resume = &XrdXrootdProtocol::do_WriteCont;
                 myBlast = Quantum;
                 myStalls++;
                }
             return rc;
            }
         if (myFile->XrdSfsp->write(myOffset, argp->buff, Quantum) < 0)
            {myIOLen  = myIOLen-Quantum;
             return do_WriteNone();
            }
         myOffset += Quantum; myIOLen -= Quantum;
         if (myIOLen < Quantum) Quantum = myIOLen;
        }

// All done
//
   return Response.Send();
}

/******************************************************************************/
/*                          d o _ W r i t e C o n t                           */
/******************************************************************************/

// myFile   = file to be written
// myOffset = Offset at which to write
// myIOLen  = Number of bytes to read from socket and write to file
// myBlast  = Number of bytes already read from the socket
  
int XrdXrootdProtocol::do_WriteCont()
{

// Write data that was finaly finished comming in
//
   if (myFile->XrdSfsp->write(myOffset, argp->buff, myBlast) < 0)
      {myIOLen  = myIOLen-myBlast;
       return do_WriteNone();
      }
    myOffset += myBlast; myIOLen -= myBlast;

// See if we need to finish this request in the normal way
//
   if (myIOLen > 0) return do_WriteAll();
   return Response.Send();
}
  
/******************************************************************************/
/*                          d o _ W r i t e N o n e                           */
/******************************************************************************/
  
int XrdXrootdProtocol::do_WriteNone()
{
   int rlen, blen = (myIOLen > argp->bsize ? argp->bsize : myIOLen);

// Discard any data being transmitted
//
   TRACEP(REQ, "discarding " <<myIOLen <<" bytes");
   while(myIOLen > 0)
        {rlen = Link->Recv(argp->buff, blen, readWait);
         if (rlen  < 0) return Link->setEtext("link read error");
         myIOLen -= rlen;
         if (rlen < blen) 
            {myBlen   = 0;
             Resume   = &XrdXrootdProtocol::do_WriteNone;
             return 1;
            }
         if (myIOLen < blen) blen = myIOLen;
        }

// Send our the error message and return
//
   return Response.Send(kXR_FSError, myFile->XrdSfsp->error.getErrText());
}
  
/******************************************************************************/
/*                       U t i l i t y   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               f s E r r o r                                */
/******************************************************************************/
  
int XrdXrootdProtocol::fsError(int rc, XrdOucErrInfo &myError)
{
   int ecode;
   const char *eMsg = myError.getErrText(ecode);

// Process standard errors
//
   if (rc == SFS_ERROR)
      {SI->errorCnt++;
       rc = mapError(ecode);
       return Response.Send((XErrorCode)rc, eMsg);
      }

// Process the redirection (error msg is host:port)
//
   if (rc == SFS_REDIRECT)
      {SI->redirCnt++;
       if (ecode <= 0) ecode = (ecode ? -ecode : Port);
       TRACEI(REDIR, Response.ID() <<"redirecting to " << eMsg <<':' <<ecode);
       return Response.Send(kXR_redirect, ecode, eMsg);
      }

// Process the deferal. We also synchronize sending the deferal response with
// sending the actual defered response by calling Done() in the callback object.
// This allows the requestor of he callback know that we actually send the
// kXR_waitresp to the end client and avoid violating time causality.
//
   if (rc == SFS_STARTED)
      {SI->stallCnt++;
       if (ecode <= 0) ecode = 1800;
       TRACEI(STALL, Response.ID() <<"delaying client up to " <<ecode <<" sec");
       rc = Response.Send(kXR_waitresp, ecode, eMsg);
       if (myError.getErrCB()) myError.getErrCB()->Done(ecode, &myError);
       return (rc ? rc : 1);
      }

// Process the data response
//
   if (rc == SFS_DATA)
      {if (ecode) return Response.Send((void *)eMsg, ecode);
          else    return Response.Send();
      }

// Process the deferal
//
   if (rc >= SFS_STALL)
      {SI->stallCnt++;
       TRACEI(STALL, Response.ID() <<"stalling client for " <<rc <<" sec");
       return (rc = Response.Send(kXR_wait, rc, eMsg)) ? rc : 1;
      }

// Unknown conditions, report it
//
   {char buff[32];
    SI->errorCnt++;
    sprintf(buff, "%d", rc);
    eDest.Emsg("Xeq", "Unknown error code", buff, eMsg);
    return Response.Send(kXR_ServerError, eMsg);
   }
}
  
/******************************************************************************/
/*                               g e t B u f f                                */
/******************************************************************************/
  
int XrdXrootdProtocol::getBuff(const int isRead, int Quantum)
{

// Check if we need to really get a new buffer
//
   if (!argp || Quantum > argp->bsize) hcNow = hcPrev;
      else if (Quantum >= halfBSize || hcNow-- > 0) return 1;
              else if (hcNext >= hcMax) hcNow = hcMax;
                      else {int tmp = hcPrev;
                            hcNow   = hcNext;
                            hcPrev  = hcNext;
                            hcNext  = tmp+hcNext;
                           }

// Get a new buffer
//
   if (argp) BPool->Release(argp);
   if ((argp = BPool->Obtain(Quantum))) halfBSize = argp->bsize >> 1;
      else return Response.Send(kXR_NoMemory, (isRead ?
                                "insufficient memory to read file" :
                                "insufficient memory to write file"));

// Success
//
   return 1;
}

/******************************************************************************/
/*                              m a p E r r o r                               */
/******************************************************************************/
  
int XrdXrootdProtocol::mapError(int rc)
{
    if (rc < 0) rc = -rc;
    switch(rc)
       {case ENOENT:       return kXR_NotFound;
        case EPERM:        return kXR_NotAuthorized;
        case EACCES:       return kXR_NotAuthorized;
        case EIO:          return kXR_IOError;
        case ENOMEM:       return kXR_NoMemory;
        case ENOBUFS:      return kXR_NoMemory;
        case ENOSPC:       return kXR_NoSpace;
        case ENAMETOOLONG: return kXR_ArgTooLong;
        case ENETUNREACH:  return kXR_noserver;
        case ENOTBLK:      return kXR_NotFile;
        case EISDIR:       return kXR_isDirectory;
        case EEXIST:       return kXR_InvalidRequest;
        case ETXTBSY:      return kXR_inProgress;
        default:           return kXR_FSError;
       }
}

/******************************************************************************/
/*                               m a p M o d e                                */
/******************************************************************************/

#define Map_Mode(x,y) if (Mode & kXR_ ## x) newmode |= S_I ## y

int XrdXrootdProtocol::mapMode(int Mode)
{
   int newmode = 0;

// Map the mode in the obvious way
//
   Map_Mode(ur, RUSR); Map_Mode(uw, WUSR);  Map_Mode(ux, XUSR);
   Map_Mode(gr, RGRP); Map_Mode(gw, WGRP);  Map_Mode(gx, XGRP);
   Map_Mode(or, ROTH);                      Map_Mode(ox, XOTH);

// All done
//
   return newmode;
}
  
/******************************************************************************/
/*                               r p C h e c k                                */
/******************************************************************************/
  
int XrdXrootdProtocol::rpCheck(char *fn, const char **opaque)
{
   char *cp;

   if (*fn != '/') return 1;

   if (!(cp = index(fn, '?'))) *opaque = 0;
      else {*cp = '\0'; *opaque = cp+1;
            if (!**opaque) *opaque = 0;
           }

   while ((cp = index(fn, '/')))
         {fn = cp+1;
          if (fn[0] == '.' && fn[1] == '.' && fn[2] == '/') return 1;
         }
   return 0;
}
  
/******************************************************************************/
/*                                r p E m s g                                 */
/******************************************************************************/
  
int XrdXrootdProtocol::rpEmsg(const char *op, char *fn)
{
   char buff[2048];
   snprintf(buff,sizeof(buff)-1,"%s relative path '%s' is disallowed.",op,fn);
   buff[sizeof(buff)-1] = '\0';
   return Response.Send(kXR_NotAuthorized, buff);
}
 
/******************************************************************************/
/*                                S q u a s h                                 */
/******************************************************************************/
  
int XrdXrootdProtocol::Squash(char *fn)
{
   char *ofn, *ifn = fn;

   while(*ifn)
        {if (*ifn == '/')
            if (*(ifn+1) == '/'
            || (*(ifn+1) == '.' && *(ifn+1) && *(ifn+2) == '/')) break;
         ifn++;
        }

   if (!*ifn) return XPList.Validate(fn, ifn-fn);

   ofn = ifn;
   while(*ifn) {*ofn = *ifn++;
                while(*ofn == '/')
                   {while(*ifn == '/') ifn++;
                    if (ifn[0] == '.' && ifn[1] == '/') ifn += 2;
                       else break;
                   }
                ofn++;
               }
   *ofn = '\0';

   return XPList.Validate(fn, ofn-fn);
}

/******************************************************************************/
/*                               S t a t G e n                                */
/******************************************************************************/
  
#define XRDXROOTD_STAT_CLASSNAME XrdXrootdProtocol
#include "XrdXrootd/XrdXrootdStat.icc"

/******************************************************************************/
/*                                v p E m s g                                 */
/******************************************************************************/
  
int XrdXrootdProtocol::vpEmsg(const char *op, char *fn)
{
   char buff[2048];
   snprintf(buff,sizeof(buff)-1,"%s path '%s' is disallowed.",op,fn);
   buff[sizeof(buff)-1] = '\0';
   return Response.Send(kXR_NotAuthorized, buff);
}
