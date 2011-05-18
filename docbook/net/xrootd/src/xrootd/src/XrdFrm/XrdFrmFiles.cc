/******************************************************************************/
/*                                                                            */
/*                        X r d F r m F i l e s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include <errno.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysPlatform.hh"

const char *XrdFrmFilesCVSID = "$Id$";

using namespace XrdFrm;

/******************************************************************************/
/*                   C l a s s   X r d F r m F i l e s e t                    */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmFileset::XrdFrmFileset(XrdFrmFileset *sP, XrdOucTList *diP)
              : Next(sP), dInfo(diP), dlkFD(-1), flkFD(-1)
{  memset(File, 0, sizeof(File));
   if (diP) diP->ival[dRef]++;
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdFrmFileset::~XrdFrmFileset()
{
   int i;

// Unlock any locked files
//
   UnLock();

// Delete all the table entries
//
   for (i = 0; i < XrdOssPath::sfxNum; i++) if(File[i]) delete File[i];

// If there is a shared directory buffer, decrease reference count, delete if 0
//
   if (dInfo && ((dInfo->ival[dRef] -= 1) <= 0)) delete dInfo;
}

/******************************************************************************/
/*                               d i r P a t h                                */
/******************************************************************************/
  
int XrdFrmFileset::dirPath(char *dBuff, int dBlen)
{
   char *dP = 0;
   int   dN = 0, i;

// If we have a shared directory pointer, use that as directory information
// Otherwise, get it from one of the files in the fileset.
//
   if (dInfo) {dP = dInfo->text; dN = dInfo->ival[dLen];}
      else {for (i = 0; i < XrdOssPath::sfxNum; i++)
                if (File[i])
                   {dP = File[i]->Path;
                    dN = File[i]->File - File[i]->Path;
                    break;
                   }
           }

// Copy out the directory path
//
   if (dBlen > dN && dP) strncpy(dBuff, dP, dN);
      else dN = 0;
   *(dBuff+dN) = '\0';
   return dN;
}

/******************************************************************************/
/*                               R e f r e s h                                */
/******************************************************************************/
  
int XrdFrmFileset::Refresh(int isMig, int doLock)
{
   XrdOucNSWalk::NSEnt *bP = baseFile(), *lP = lockFile();
   char pBuff[MAXPATHLEN+1], *fnP, *pnP = pBuff;
   int n;

// Get the directory path for this entry
//
   if (!(n = dirPath(pBuff, sizeof(pBuff)-1))) return 0;
   fnP = pBuff+n;

// If we need to lock the entry, do so. We also check if file is in use
//
   if (doLock && bP)
      {strcpy(fnP, baseFile()->File);
       if (chkLock(pBuff)) return 0;
       if (lP && dlkFD < 0)
          {strcpy(fnP, Config.lockFN);
           if ((dlkFD = getLock(pBuff)) < 0) return 0;
           strcpy(fnP, lockFile()->File);
           if ((flkFD = getLock(pBuff,0,1)) < 0)
              {close(dlkFD); dlkFD = -1; return 0;}
          }
      }

// Do a new stat call on each relevant file (pin file excluded for isMig)
//
   if (bP)
      {if (bP->Link) pnP = bP->Link;
         else strcpy(fnP, bP->File);
       if (stat(pnP, &(bP->Stat)))
          {Say.Emsg("Refresh", errno, "stat", pnP); UnLock(); return 0;}
      }

   if (lP)
      {strcpy(fnP, lP->File);
       if (stat(pBuff, &(lP->Stat)))
          {Say.Emsg("Refresh", errno, "stat", pBuff); UnLock(); return 0;}
      }

   if (!isMig && (bP = pinFile()))
      {strcpy(fnP, bP->File);
       if (stat(pBuff, &(bP->Stat)) && errno == ENOENT)
          {delete bP; File[XrdOssPath::isPin] = 0;}
      }

// All done
//
   return 1;
}

/******************************************************************************/
/*                                U n L o c k                                 */
/******************************************************************************/
  
void XrdFrmFileset::UnLock()
{
   if (flkFD >= 0) {close(flkFD); flkFD = -1;}
   if (dlkFD >= 0) {close(dlkFD); dlkFD = -1;}
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               c h k L o c k                                */
/******************************************************************************/

// Returns 0 if no lock exists o/w returns 1 or -1;
  
int XrdFrmFileset::chkLock(const char *Path)
{
   FLOCK_t lock_args;
   int rc, lokFD;

// Open the file appropriately
//
   if ((lokFD = open(Path, O_RDONLY)) < 0)
      {Say.Emsg("chkLock", errno, "open", Path); return -1;}

// Initialize the lock arguments
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type = F_WRLCK;

// Now check if the lock can be obtained
//
   do {rc = fcntl(lokFD, F_GETLK, &lock_args);} while(rc < 0 && errno == EINTR);

// Determine the result
//
   if (rc) Say.Emsg("chkLock", errno, "lock", Path);
      else rc = (lock_args.l_type == F_UNLCK ? 0 : 1);

// All done
//
   close(lokFD);
   return rc;
}

/******************************************************************************/
/*                               g e t L o c k                                */
/******************************************************************************/
  
int XrdFrmFileset::getLock(char *Path, int Shared, int noWait)
{
   static const int AMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
   FLOCK_t lock_args;
   int oFlags = (Shared ? O_RDONLY : O_RDWR|O_CREAT);
   int rc, lokFD, Act;

// Open the file appropriately
//
   if ((lokFD = open(Path, oFlags, AMode)) < 0)
      {Say.Emsg("getLock", errno, "open", Path); return -1;}

// Initialize the lock arguments
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type = (Shared ? F_RDLCK : F_WRLCK);
   Act = (noWait ? F_SETLK : F_SETLKW);

// Now obtain the lock
//
   do {rc = fcntl(lokFD, Act, &lock_args);} while(rc < 0 && errno == EINTR);

// Determine the result
//
   if (rc)
      {if (errno != EAGAIN) Say.Emsg("getLock", errno, "lock", Path);
       close(lokFD); return -1;
      }

// All done
//
   return lokFD;
}

/******************************************************************************/
/*                                  M k f n                                   */
/******************************************************************************/
  
const char *XrdFrmFileset::Mkfn(XrdOucNSWalk::NSEnt *fP)
{

// If we have no file for this, return the null string
//
   if (!fP) return "";

// If we have no shared directory pointer, return the full path
//
   if (!dInfo) return fP->Path;

// Construct the name in a non-renterant way (this is documented)
//
   strcpy(dInfo->text+dInfo->ival[dLen], fP->File);
   return dInfo->text;
}
  
/******************************************************************************/
/*                     C l a s s   X r d F r m F i l e s                      */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmFiles::XrdFrmFiles(const char *dname, int opts,
                        XrdOucTList *XList, XrdOucNSWalk::CallBack *cbP)
            : nsObj(&Say, dname, Config.lockFN,
                    XrdOucNSWalk::retFile | XrdOucNSWalk::retLink
                   |XrdOucNSWalk::retStat | XrdOucNSWalk::skpErrs
                   |XrdOucNSWalk::retIILO
                   | (opts & CompressD  ?   XrdOucNSWalk::noPath  : 0)
                   | (opts & Recursive  ?   XrdOucNSWalk::Recurse : 0), XList),
              fsList(0), manMem(opts & NoAutoDel ? Hash_keep : Hash_default),
              shareD(opts & CompressD)
{

// Set Call Back method
//
   nsObj.setCallBack(cbP);
}

/******************************************************************************/
/*                                   G e t                                    */
/******************************************************************************/
  
XrdFrmFileset *XrdFrmFiles::Get(int &rc, int noBase)
{
   XrdOucNSWalk::NSEnt *nP;
   XrdFrmFileset *fsetP;
   const char *dPath;

// Check if we have something to return
//
do{while ((fsetP = fsList))
         {fsList = fsetP->Next; fsetP->Next = 0;
          if (fsetP->File[XrdOssPath::isBase] || noBase) {rc = 0; return fsetP;}
             else if (manMem) delete fsetP;
         }

// Start with next directory (we return when no directories left).
//
   do {if (!(nP = nsObj.Index(rc, &dPath))) return 0;
       fsTab.Purge(); fsList = 0;
      } while(!Process(nP, dPath));

  } while(1);

// To keep the compiler happy
//
   return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/
  
int XrdFrmFiles::Process(XrdOucNSWalk::NSEnt *nP, const char *dPath)
{
   XrdOucNSWalk::NSEnt *fP;
   XrdFrmFileset       *sP;
   XrdOucTList         *dP = 0;
   char *dotP;
   int fType;

// If compressed directories wanted, then setup a shared directory buffer
// Warning! We use a hard-coded value for maximum filename length instead of
//          constantly calling pathconf().
//
   if (shareD)
      {int n = strlen(dPath);
       char *dBuff = (char *)malloc(n+264);
       strcpy(dBuff, dPath);
       dP = new XrdOucTList;
       dP->text = dBuff;
       dP->ival[XrdFrmFileset::dLen] = n;
       dP->ival[XrdFrmFileset::dRef] = 0;
      }

// Process the file list
//
   while((fP = nP))
        {nP = fP->Next; fP->Next = 0;
         if (!strcmp(fP->File, Config.lockFN)) {delete fP; continue;}
         if ((fType = (int)XrdOssPath::pathType(fP->File))
         && (dotP = rindex(fP->File, '.'))) *dotP = '\0';
            else dotP = 0;
         if (!(sP = fsTab.Find(fP->File)))
            {sP = fsList = new XrdFrmFileset(fsList, dP);
             fsTab.Add(fP->File, sP, 0, manMem);
            }
         if (dotP) *dotP = '.';
         sP->File[fType] = fP;
        }

// Indicate whether we have anything here
//
   if (fsList) return 1;
   if (dP) delete dP;
   return 0;
}
