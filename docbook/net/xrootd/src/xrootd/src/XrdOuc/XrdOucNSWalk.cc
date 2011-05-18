/******************************************************************************/
/*                                                                            */
/*                       X r d O u c N S W a l k . c c                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOucNSWalkCVSID = "$Id$";

#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>

#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOucNSWalk::XrdOucNSWalk(XrdSysError *erp, const char *dpath,
                                             const char *lkfn, int opts,
                                             XrdOucTList *xlist)
{
// Set the required fields
//
   eDest = erp;
   DList = new XrdOucTList(dpath);
   if (lkfn) LKFn = strdup(lkfn);
      else   LKFn = 0;
   Opts = opts;
   DPfd = LKfd = -1;
   errOK= opts & skpErrs;
   DEnts= 0;
   edCB = 0;

// Copy the exclude list if one exists
//
   if (!xlist) XList = 0;
      else while(xlist)
                {XList = new XrdOucTList(xlist->text,xlist->ival,XList);
                 xlist = xlist->next;
                }
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdOucNSWalk::~XrdOucNSWalk()
{
   XrdOucTList *tP;

   if (LKFn) free(LKFn);

   while((tP = DList)) {DList = tP->next; delete tP;}

   while((tP = XList)) {XList = tP->next; delete tP;}
}

/******************************************************************************/
/*                                 I n d e x                                  */
/******************************************************************************/
  
XrdOucNSWalk::NSEnt *XrdOucNSWalk::Index(int &rc, const char **dPath)
{
   XrdOucTList *tP;
   NSEnt *eP;

// Sequence the directory
//
   rc = 0; *DPath = '\0';
   while((tP = DList))
        {setPath(tP->text);
         DList = tP->next; delete tP;
         if (LKFn && (rc = LockFile())) break;
         rc = Build();
         if (LKfd >= 0) close(LKfd);
         if (DEnts || (rc && !errOK)) break;
         if (edCB && isEmpty) edCB->isEmpty(&dStat, DPath, LKFn);
        }

// Return the result
//
   eP = DEnts; DEnts = 0;
   if (dPath) *dPath = DPath;
   return eP;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                a d d E n t                                 */
/******************************************************************************/
  
void XrdOucNSWalk::addEnt(XrdOucNSWalk::NSEnt *eP)
{
   static const int retIxLO = retIDLO | retIILO;

// Complete the entry
//
   if (Opts & noPath) {eP->Path = strdup(File); eP->File = eP->Path;}
      else {eP->Path = strdup(DPath);
            eP->File = eP->Path + (File - DPath);
           }
    eP->Plen = (eP->File - eP->Path) + strlen(eP->File);

// Chain the entry into the list
//
   if (!(Opts & retIxLO)) {eP->Next = DEnts; DEnts = eP;}
      else {NSEnt *pP = 0, *nP = DEnts;
            if (Opts & retIDLO)
               while(nP && eP->Plen < nP->Plen) {pP = nP; nP = nP->Next;}
               else
               while(nP && eP->Plen > nP->Plen) {pP = nP; nP = nP->Next;}
            if (pP) {eP->Next = nP; pP->Next = eP;}
               else {eP->Next = nP; DEnts    = eP;}
           }
}

/******************************************************************************/
/*                                 B u i l d                                  */
/******************************************************************************/
  
int XrdOucNSWalk::Build()
{
   struct Helper {XrdOucNSWalk::NSEnt *P;
                  DIR                 *D;
                  int                  F;
                                       Helper() : P(0), D(0), F(-1) {}
                                      ~Helper() {if (P)   delete P;
                                                 if (D)   closedir(D);
                                                 if (F>0) close(F);
                                                }
                 } theEnt;
   struct dirent  *dp;
   int             rc = 0, getLI = Opts & retLink;
   int             nEnt = 0, xLKF = 0, chkED = (edCB != 0) && (LKFn != 0);

// Initialize the empty flag prior to doing anything else
//
   isEmpty = 0;

// If we can optimize with a directory file descriptor, get one
//
#ifdef HAVE_FSTATAT
   if ((DPfd = open(DPath, O_RDONLY)) < 0) rc = errno;
      else theEnt.F = DPfd;
#else
   DPfd = -1;
#endif

// Open the directory
//
   if (!(theEnt.D = opendir(DPath)))
      {rc = errno;
       if (eDest) eDest->Emsg("Build", rc, "open directory", DPath);
       return rc;
      }

// Process the entries
//
   errno = 0;
   while((dp = readdir(theEnt.D)))
        {if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) continue;
         strcpy(File, dp->d_name); nEnt++;
         if (!theEnt.P) theEnt.P = new NSEnt();
         rc = getStat(theEnt.P, getLI);
         switch(theEnt.P->Type)
               {case NSEnt::isDir:
                     if (Opts & Recurse && (!getLI || !isSymlink())
                     &&  (!XList || !inXList(File)))
                        DList = new XrdOucTList(DPath, 0, DList);
                     if (!(Opts & retDir)) continue;
                     break;
                case NSEnt::isFile:
                     if ((chkED && !xLKF && (xLKF = !strcmp(File, LKFn)))
                     ||  !(Opts & retFile)) continue;
                     break;
                case NSEnt::isLink:
                     if ((rc = getLink(theEnt.P)))
                        memset(&theEnt.P->Stat, 0, sizeof(struct stat));
                        else if ((Opts & retStat) && (rc = getStat(theEnt.P)))
                                {theEnt.P->Type = NSEnt::isLink; rc = 0;}
                     break;
                case NSEnt::isMisc:
                     if (!(Opts & retMisc)) continue;
                     break;
                default:
                     if (!rc) rc = EINVAL;
                     break;
               }
         errno = 0;
         if (rc) {if (errOK) continue; return rc;}
         addEnt(theEnt.P); theEnt.P = 0; 
        }

// All done, check if we reached EOF or there is an error
//
   *File = '\0';
   if ((rc = errno) && !errOK)
      {eDest->Emsg("Build", rc, "reading directory", DPath); return rc;}

// Check if we need to do a callback for an empty directory
//
   if (edCB && xLKF == nEnt && !DEnts)
      {if ((DPfd < 0 ? !stat(DPath, &dStat) : !fstat(DPfd, &dStat))) isEmpty=1;
          else eDest->Emsg("Build", errno, "stating directory", DPath);
      }
   return 0;
}

/******************************************************************************/
/*                               g e t L i n k                                */
/******************************************************************************/

int XrdOucNSWalk::getLink(XrdOucNSWalk::NSEnt *eP)
{
   char lnkbuff[2048];
   int rc;

   if ((rc = readlink(DPath, lnkbuff, sizeof(lnkbuff))) < 0)
      {rc = errno;
       if (eDest) eDest->Emsg("getLink", rc, "read link of", DPath);
       return rc;
      }

   eP->Lksz = rc;
   eP->Link = (char *)malloc(rc+1);
   memcpy(eP->Link, lnkbuff, rc);
   *(eP->Link+rc) = '\0';
   return 0;
}
  
/******************************************************************************/
/*                               g e t S t a t                                */
/******************************************************************************/
  
int XrdOucNSWalk::getStat(XrdOucNSWalk::NSEnt *eP, int doLstat)
{
   int rc;

// The following code either uses fstatat() or regular stat()
//
#ifdef HAVE_FSTATAT
do{rc = fstatat(DPfd, File, &(eP->Stat), (doLstat ? AT_SYMLINK_NOFOLLOW : 0));
#else
do{rc = doLstat ? lstat(DPath, &(eP->Stat)) : stat(DPath, &(eP->Stat));
#endif
  } while(rc && errno == EINTR);

// Check for errors
//
   if (rc)
      {rc = errno;
       if (eDest && rc != ENOENT && rc != ELOOP)
          eDest->Emsg("getStat", rc, "stat", DPath);
       memset(&eP->Stat, 0, sizeof(struct stat));
       eP->Type = NSEnt::isBad;
       return rc;
      }

// Set appropraite type
//
        if ((eP->Stat.st_mode & S_IFMT) == S_IFDIR) eP->Type = NSEnt::isDir;
   else if ((eP->Stat.st_mode & S_IFMT) == S_IFREG) eP->Type = NSEnt::isFile;
   else if ((eP->Stat.st_mode & S_IFMT) == S_IFLNK) eP->Type = NSEnt::isLink;
   else                                             eP->Type = NSEnt::isMisc;

   return 0;
}
  
/******************************************************************************/
/*                               i n X L i s t                                */
/******************************************************************************/
  
int XrdOucNSWalk::inXList(const char *dName)
{
    XrdOucTList *xTP = XList, *pTP = 0;

// Search for the directory entry
//
    while(xTP && strcmp(DPath, xTP->text)) {pTP = xTP; xTP = xTP->next;}

// If not found return false. Otherwise, delete the entry and return true.
//
   if (!xTP) return 0;
   if (pTP) pTP->next = xTP->next;
      else      XList = xTP->next;
   delete xTP;
   return 1;
}
  
/******************************************************************************/
/*                             i s S y m l i n k                              */
/******************************************************************************/
  
int XrdOucNSWalk::isSymlink()
{
   struct stat buf;
   int rc;


// The following code either uses fstatat() or regular stat()
//
#ifdef HAVE_FSTATAT
do{rc = fstatat(DPfd, File, &buf, AT_SYMLINK_NOFOLLOW);
#else
do{rc = lstat(DPath, &buf);
#endif
  } while(rc && errno == EINTR);

// Check for errors
//
   if (rc) return 0;
   return (buf.st_mode & S_IFMT) == S_IFLNK;
}

/******************************************************************************/
/*                              L o c k F i l e                               */
/******************************************************************************/
  
int XrdOucNSWalk::LockFile()
{
   FLOCK_t lock_args;
   int rc;

// Construct the path and open the file
//
   strcpy(File, LKFn);
   do {LKfd = open(DPath, O_RDWR);} while(LKfd < 0 && errno == EINTR);
   if (LKfd < 0)
      {if (errno == ENOENT) {*File = '\0'; return 0;}
          {rc = errno;
           if (eDest) eDest->Emsg("LockFile", rc, "open", DPath);
           *File = '\0'; return rc;
          }
      }

// Establish locking options
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type = F_WRLCK;

// Perform action.
//
   do {rc = fcntl(LKfd,F_SETLKW,&lock_args);}
       while(rc < 0 && errno == EINTR);
   if (rc < 0)
       {rc = -errno;
        if (eDest) eDest->Emsg("LockFile", errno, "lock", DPath);
       }

// All done
//
   *File = '\0';
   return rc;
}

/******************************************************************************/
/*                               s e t P a t h                                */
/******************************************************************************/
  
void XrdOucNSWalk::setPath(char *newpath)
{
   int n;

   strcpy(DPath, newpath);
   n = strlen(newpath);
   if (DPath[n-1] != '/')
      {DPath[n++] = '/'; DPath[n] = '\0';}
   File = DPath+n;
}
