/******************************************************************************/
/*                                                                            */
/*                        X r d F r m F i l e s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

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

using namespace XrdFrm;

/******************************************************************************/
/*                   C l a s s   X r d F r m F i l e s e t                    */
/******************************************************************************/
/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/

XrdOucHash<char>  XrdFrmFileset::BadFiles;
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmFileset::XrdFrmFileset(XrdFrmFileset *sP, XrdOucTList *diP)
              : Next(sP), dInfo(diP)
{  memset(File, 0, sizeof(File));
   if (diP) diP->ival[dRef]++;
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdFrmFileset::~XrdFrmFileset()
{
   int i;

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
   class fdClose
        {public:
         int Num;
             fdClose() : Num(-1) {}
            ~fdClose() {if (Num >= 0) close(Num);}
        } fnFD;
   XrdOucNSWalk::NSEnt *lP, *bP = baseFile();
   char pBuff[MAXPATHLEN+1], *fnP, *pnP = pBuff;
   int n, lkFD = -1;

// Get the directory path for this entry
//
   if (!(n = dirPath(pBuff, sizeof(pBuff)-1))) return 0;
   fnP = pBuff+n;

// If we need to lock the entry, do so. We also check if file is in use
//
   if (doLock && bP)
      {strcpy(fnP, baseFile()->File);
       if (!(lkFD = chkLock(pBuff))) return 0;
       fnFD.Num = lkFD;
      }

// Do a new stat call on each relevant file (pin file excluded for isMig)
//
   if (bP)
      {if (bP->Link) pnP = bP->Link;
         else strcpy(fnP, bP->File);
       if (stat(pnP, &(bP->Stat)))
          {Say.Emsg("Refresh", errno, "stat", pnP); return 0;}
      }

   if (!isMig) pinInfo.Get(pnP, lkFD);

   if ((lP = lockFile()))
      {strcpy(fnP, lP->File);
       if (stat(pBuff, &(lP->Stat)))
          {Say.Emsg("Refresh", errno, "stat", pBuff); return 0;}
       cpyInfo.Attr.cpyTime = static_cast<long long>(lP->Stat.st_mtime);
      } else if (cpyInfo.Get(pnP, lkFD) <= 0) cpyInfo.Attr.cpyTime = 0;

// All done
//
   return 1;
}

/******************************************************************************/
/*                                S c r e e n                                 */
/******************************************************************************/

int XrdFrmFileset::Screen(int needLF)
{
   const char *What = 0, *badFN = 0;

// Verify that we have all the relevant files (old mode only)
//
   if (!Config.runNew && !baseFile())
      {if (Config.Fix)
          {if (lockFile()) Remfix("Lock", lockPath());
           if ( pinFile()) Remfix("Pin",  pinPath());
           return 0;
          }
            if (lockFile()) badFN = lockPath();
       else if ( pinFile()) badFN = pinPath();
       else return 0;
       What = "No base file for";
      }

// If no errors from above, try to get the copy time for this file
//
   if (!What)
      {if (!needLF || setCpyTime()) return 1;
       What = Config.runNew ? "no copy time xattr for" : "no lock file for";
       badFN = basePath();
      }

// Issue message if we haven't issued one before
//
   if (!BadFiles.Add(badFN, 0, 0, Hash_data_is_key))
      Say.Emsg("Screen", What, badFN);
   return 0;
}

/******************************************************************************/
/*                            s e t C p y T i m e                             */
/******************************************************************************/
  
int XrdFrmFileset::setCpyTime(int Refresh)
{
   XrdOucNSWalk::NSEnt *lP;

// In new run mode the copy time comes from the extended attributes
//
   if (Config.runNew) return cpyInfo.Get(basePath()) > 0;

// If there is no lock file, indicate so
//
   if (!(lP = lockFile())) return 0;

// Use the lock file as the source of information
//
   if (Refresh && stat(lockPath(), &(lP->Stat)))
      {Say.Emsg("setCpyTime", errno, "stat", lockPath()); return 0;}
   cpyInfo.Attr.cpyTime = static_cast<long long>(lP->Stat.st_mtime);
   return 1;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               c h k L o c k                                */
/******************************************************************************/

// Returns 0 if lock exists or an error occurred, o/w returns fd for the file.
  
int XrdFrmFileset::chkLock(const char *Path)
{
   FLOCK_t lock_args;
   int rc, lokFD;

// Open the file appropriately
//
   if ((lokFD = open(Path, O_RDONLY)) < 0)
      {Say.Emsg("chkLock", errno, "open", Path); return 0;}

// Initialize the lock arguments
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type = F_WRLCK;

// Now check if the lock can be obtained
//
   do {rc = fcntl(lokFD, F_GETLK, &lock_args);} while(rc < 0 && errno == EINTR);

// Determine the result
//
   if (!rc) return lokFD;
   Say.Emsg("chkLock", errno, "lock", Path);
   close(lokFD);
   return 0;
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
/*                                R e m f i x                                 */
/******************************************************************************/

void XrdFrmFileset::Remfix(const char *fType, const char *fPath)
{

// Remove the offending file
//
   if (unlink(fPath)) Say.Emsg("Remfix", errno, "remove orphan", fPath);
      Say.Emsg("Remfix", fType, "file orphan fixed; removed", fPath);
}
  
/******************************************************************************/
/*                     C l a s s   X r d F r m F i l e s                      */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmFiles::XrdFrmFiles(const char *dname, int opts,
                        XrdOucTList *XList, XrdOucNSWalk::CallBack *cbP)
            : nsObj(&Say, dname, 0,
                    XrdOucNSWalk::retFile | XrdOucNSWalk::retLink
                   |XrdOucNSWalk::retStat | XrdOucNSWalk::skpErrs
                   |XrdOucNSWalk::retIILO
                   | (opts & CompressD  ?   XrdOucNSWalk::noPath  : 0)
                   | (opts & Recursive  ?   XrdOucNSWalk::Recurse : 0), XList),
              fsList(0), manMem(opts & NoAutoDel ? Hash_keep : Hash_default),
              shareD(opts & CompressD), getCPT(opts & GetCpyTim)
{

// Set Call Back method
//
   nsObj.setCallBack(cbP);
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdFrmFiles::~XrdFrmFiles()
{
   XrdFrmFileset *fsetP;

// If manual memory is wante then we must delete any unreturned objects
//
   if (manMem)
       while((fsetP = fsList))
            {fsList = fsetP->Next; fsetP->Next = 0; delete fsetP;}
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
               if (fsetP->File[XrdOssPath::isBase])
                  {if (getCPT) fsetP->setCpyTime();
                   rc = 0; return fsetP;
                  }
          else if (noBase) {rc = 0; return fsetP;}
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
/*                              C o m p l a i n                               */
/******************************************************************************/
  
void XrdFrmFiles::Complain(const char *dPath)
{
   static const int OneDay = 24*60*60;
   static XrdOucHash<char> dTab;

// We want to complain about old=style directories only once every 24 hours
//
   if (dTab.Add(dPath, 0, OneDay, Hash_data_is_key)) return;

// Complain about this directory
//
   Say.Emsg("Complain","Found old-style files in directory", dPath);
   Say.Emsg("Complain","In new run mode, migrate & purge will skip them.");
}

/******************************************************************************/
/*                               o l d F i l e                                */
/******************************************************************************/

int XrdFrmFiles::oldFile(XrdOucNSWalk::NSEnt *fP, XrdOucTList *dP, int fType)
{
   char pBuff[MAXPATHLEN+8], *pnP = pBuff, *fnP;

// Ignore (for now): '.anew', '.fail', or '.pfn'
//
   if (fType == XrdOssPath::isAnew
   ||  fType == XrdOssPath::isFail
   ||  fType == XrdOssPath::isPfn) return 0;

// If this is not a directory lock file, indicate we should complain
//
   if (fType >= 0) return 1;
  
// This is a directory lock file, quietly remove it (we no longer use them)
//
   if (!dP) pnP = fP->Path;
      else {strcpy(pBuff, dP->text);
            fnP = pBuff + dP->ival[XrdFrmFileset::dLen];
            *fnP++ = '/';
            strcpy(fnP, fP->File);
           }
   unlink(pnP);
   return 0;
}

/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/
  
int XrdFrmFiles::Process(XrdOucNSWalk::NSEnt *nP, const char *dPath)
{
   XrdOucNSWalk::NSEnt *fP;
   XrdFrmFileset       *sP;
   XrdOucTList         *dP = 0;
   char *dotP;
   int fType, noDLKF = 1, runOldFault = 0;

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
         if (noDLKF && !strcmp(fP->File, Config.lockFN))
            {oldFile(fP, dP, -1); delete fP; noDLKF = 0; continue;}
         if (!(fType = (int)XrdOssPath::pathType(fP->File))
         ||  !(dotP = rindex(fP->File, '.'))) dotP = 0;
            else {if (Config.runNew)
                     {runOldFault |= oldFile(fP, dP, fType);
                      delete fP; continue;
                     }
                  *dotP = '\0';
                 }
         if (!(sP = fsTab.Find(fP->File)))
            {sP = fsList = new XrdFrmFileset(fsList, dP);
             fsTab.Add(fP->File, sP, 0, manMem);
            }
         if (dotP) *dotP = '.';
         sP->File[fType] = fP;
        }

// If we found on old-style file while in runNew, complain
//
   if (runOldFault) Complain(dPath);

// Indicate whether we have anything here
//
   if (fsList) return 1;
   if (dP) delete dP;
   return 0;
}
