/******************************************************************************/
/*                                                                            */
/*                   X r d F r m A d m i n F i l e s . c c                    */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdFrm/XrdFrmXAttr.hh"

#include "XrdOuc/XrdOucXAttr.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                                c k A t t r                                 */
/******************************************************************************/
  
char XrdFrmAdmin::ckAttr(int What, const char *Lfn, char *Pfn, int Pfnsz)
{
   struct stat Stat;
   const char *Msg = (What & mkLF ? "mark " : (What & mkMF ? "mmap " : "pin "));
   char Buff[80], Resp;

// Get the actual pfn for the base file
//
   if (!Config.LocalPath(Lfn, Pfn, Pfnsz)) {finalRC = 4; return 0;}

// Get file state
//
   if (stat(Pfn, &Stat))
      {Emsg(errno, "ckAttr ", Msg, Lfn); return 0;}

// If this is not a directory, then all is well
//
   if ((Stat.st_mode & S_IFMT) != S_IFDIR)
      {if (!Opt.All) return 'f';
       Emsg(ENOTDIR, "ckAttr ", Msg, "files in ", Lfn);
       return 0;
      }

// Make sure the whole directory is being considered
//
   if (Opt.All || Opt.Recurse) return 'd';

// Ask what we should do
//
   sprintf(Buff, "%s ALL files in directory ", Msg);
   Buff[0] = toupper(*Buff);
   if ((Resp = XrdFrmUtils::Ask('n', Buff, Lfn)) == 'y') return 'd';
   return (Resp == 'a' ? 'a' : 0);
}

/******************************************************************************/
/*                                m k M a r k                                 */
/******************************************************************************/
  
int XrdFrmAdmin::mkMark(const char *Lfn)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char Pfn[MAXPATHLEN+8], Resp;
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec = 0, Adj = (Opt.MPType == 'p' ? 0 : -113);

// Check what we are dealing with
//
   if (!(Resp = ckAttr(mkLF, Lfn, Pfn, sizeof(Pfn)-8))) return 1;
   if (Resp == 'a') return 0;

// If this is a file then do one file
//
   if (Resp == 'f')
      {if (XrdFrmUtils::updtCpy(Pfn, Adj)) numFiles++;
       return 1;
      }

// Process the directory
//
   fP = new XrdFrmFiles(Pfn, opts | XrdFrmFiles::NoAutoDel);
   while((sP = fP->Get(ec,1)))
        {if (sP->baseFile()
         &&  XrdFrmUtils::updtCpy(sP->basePath(), Adj)) numFiles++;
         delete sP;
        }

// All done
//
   if (ec) finalRC = 4;
   delete fP;
   return 1;
}
  
/******************************************************************************/
/*                                m k M m a p                                 */
/******************************************************************************/
  
int XrdFrmAdmin::mkMmap(const char *Lfn)
{
   XrdOucXAttr<XrdFrmXAttrMem> memInfo;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   const char *bFn;
   char Pfn[MAXPATHLEN+8], Resp;
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec, doSet = 0;

// Check what we are dealing with
//
   if (!(Resp = ckAttr(mkMF, Lfn, Pfn, sizeof(Pfn)-8))) return 1;
   if (Resp == 'a') return 0;

// Construct the proper mmap attribute
//
   if (!Opt.Local)
      {      doSet = memInfo.Attr.Flags  = XrdFrmXAttrMem::memMap;
       if (Opt.Fix)  memInfo.Attr.Flags |= XrdFrmXAttrMem::memLock;
       if (Opt.Keep) memInfo.Attr.Flags |= XrdFrmXAttrMem::memKeep;
      }

// If this is a file then do one file
//
   if (Resp == 'f')
      {if ((doSet ? !memInfo.Set(Pfn) : !memInfo.Del(Pfn))) numFiles++;
       return 1;
      }

// Process the directory
//
   fP = new XrdFrmFiles(Pfn, opts | XrdFrmFiles::NoAutoDel);
   while((sP = fP->Get(ec,1)))
        {if (sP->baseFile() && (bFn = sP->basePath())
         &&  (doSet ? !memInfo.Set(bFn) : !memInfo.Del(bFn))) numFiles++;
         delete sP;
        }


// All done
//
   if (ec) finalRC = 4;
   delete fP;
   return 1;
}
  
/******************************************************************************/
/*                                 m k P i n                                  */
/******************************************************************************/
  
int XrdFrmAdmin::mkPin(const char *Lfn)
{
   XrdOucXAttr<XrdFrmXAttrPin> pinInfo;
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   const char *bFn;
   char Pfn[MAXPATHLEN+8], Resp;
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec, doSet;

// Check what we are dealing with
//
   if (!(Resp = ckAttr(mkPF, Lfn, Pfn, sizeof(Pfn)-8))) return 1;
   if (Resp == 'a') return 0;

// Construct the proper pin attribute
//
        if (Opt.ktAlways)   pinInfo.Attr.Flags = XrdFrmXAttrPin::pinPerm;
   else if (Opt.KeepTime)
           {pinInfo.Attr.pinTime = static_cast<long long>(Opt.KeepTime);
            if (Opt.ktIdle) pinInfo.Attr.Flags = XrdFrmXAttrPin::pinIdle;
               else         pinInfo.Attr.Flags = XrdFrmXAttrPin::pinKeep;
           }
   doSet = (Opt.ktAlways || Opt.KeepTime ? 1 : 0);

// If this is a file then do one file
//
   if (Resp == 'f')
      {if ((doSet ? !pinInfo.Set(Pfn) : !pinInfo.Del(Pfn))) numFiles++;
       return 1;
      }

// Process the directory
//
   fP = new XrdFrmFiles(Pfn, opts | XrdFrmFiles::NoAutoDel);
   while((sP = fP->Get(ec,1)))
        {if (sP->baseFile() && (bFn = sP->basePath())
         &&  (doSet ? !pinInfo.Set(bFn) : !pinInfo.Del(bFn))) numFiles++;
         delete sP;
        }

// All done
//
   if (ec) finalRC = 4;
   delete fP;
   return 1;
}

/******************************************************************************/
/*                      O b s o l e t e   M e t h o d s                       */
/******************************************************************************/
/******************************************************************************/
/*                                m k F i l e                                 */
/******************************************************************************/
  
int XrdFrmAdmin::mkFile(int What, const char *Path, const char *Data, int DLen)
{
   static const mode_t Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH;
   struct stat Stat;
   time_t Tid;
   uid_t  Uid;
   gid_t  Gid;
   char *pfn, baseFN[1038], tempFN[1038];
   int rc, theFD;

// Check if we are handling hidden files
//
   if (!Config.lockFN || *Config.lockFN != '.') pfn = baseFN;
      else {*baseFN = '.'; pfn = baseFN+1;}

// Get the actual pfn for the base file
//
   if (What & isPFN) strcpy(pfn, Path);
      else if (!Config.LocalPath(Path, pfn, sizeof(baseFN)-6)) return 0;

// Make sure the base file exists
//
   if (stat(pfn, &Stat)) {Emsg(errno,"stat pfn ",pfn); return 0;}

// Add the appropriate suffix
//
   strcat(baseFN, (What & mkLF ? ".lock" : ".pin"));
   strcpy(tempFN, baseFN);
   strcat(tempFN, ".TEMP");

// Check if we need to merely delete the pin file
//
   if ((What & mkPF) && !Opt.ktAlways && !Opt.KeepTime)
      {if (unlink(baseFN)) {Emsg(errno, "remove pfn ", tempFN); return 0;}
       return 1;
      }

// Open the file, possibly creating it
//
   if ((theFD = open(tempFN, O_RDWR | O_CREAT | O_TRUNC, Mode)) < 0)
      {Emsg(errno, "open pfn ", tempFN); return 0;}

// If we need to write some data into the file
//
   if (Data && DLen)
      {do {rc = write(theFD, Data, DLen);
           if (rc < 0) {if (errno != EINTR) break;}
              else {Data += rc; DLen -= rc;}
          } while(DLen > 0);
       if (rc< 0) {Emsg(errno, "write pfn ", tempFN);
                   close(theFD); unlink(tempFN); return 0;
                  }
      }

// Set correct ownership
//
   Uid = (int(Opt.Uid) < 0 ? Stat.st_uid : Opt.Uid);
   Gid = (int(Opt.Gid) < 0 ? Stat.st_gid : Opt.Gid);
   if (Stat.st_uid != Uid || Stat.st_gid != Gid)
      {do {rc = fchown(theFD, Uid, Gid);} while(rc && errno == EINTR);
       if (rc) {Emsg(errno, "set uid/gid for pfn ", tempFN);
                close(theFD); unlink(tempFN); return 0;
               }
      }

// Set the file time (mig -> lock < file; prg -> lock > file)
//
   if (What & mkLF) {Tid = Stat.st_mtime + (Opt.MPType == 'p' ? +113 : -113);}
      else {Tid = (DLen || Opt.ktAlways ? time(0) : Opt.KeepTime);
            if (Opt.ktAlways)
               {do {rc = fchmod(theFD, Mode|S_ISUID);} while(rc && errno == EINTR);
                    if (rc) {Emsg(errno, "set mode for pfn ", tempFN);
                    close(theFD); unlink(tempFN); return 0;
                   }
               }
           }
   close(theFD);
   if (!XrdFrmUtils::Utime(tempFN,Tid)) {unlink(tempFN); return 0;}

// Finish up
//
   if (rename(tempFN, baseFN))
      {Emsg(errno, "rename pfn ", tempFN);
       unlink(tempFN);
       return 0;
      }
   return 1;
}

/******************************************************************************/
/*                                m k L o c k                                 */
/******************************************************************************/
  
int XrdFrmAdmin::mkLock(const char *Lfn)
{
   XrdFrmFileset *sP;
   XrdFrmFiles   *fP;
   char Pfn[MAXPATHLEN+8], Resp;
   int opts = (Opt.Recurse ? XrdFrmFiles::Recursive : 0);
   int ec = 0;

// Check what we are dealing with
//
   if (!(Resp = mkStat(mkLF, Lfn, Pfn, sizeof(Pfn)-8))) return 1;
   if (Resp == 'a') return 0;

// If this is a file then do one file
//
   if (Resp == 'f')
      {if (mkFile(mkLF|isPFN, Pfn)) numFiles++;
       return 1;
      }

// Process the directory
//
   fP = new XrdFrmFiles(Pfn, opts | XrdFrmFiles::NoAutoDel);
   while((sP = fP->Get(ec,1)))
        {if (sP->baseFile() && mkFile(mkLF|isPFN, sP->basePath())) numFiles++;
         delete sP;
        }

// All done
//
   if (ec) finalRC = 4;
   delete fP;
   return 1;
}

/******************************************************************************/
/*                                m k S t a t                                 */
/******************************************************************************/
  
char XrdFrmAdmin::mkStat(int What, const char *Lfn, char *Pfn, int Pfnsz)
{
   struct stat Stat;
   const char *Msg = (What & mkLF ? "create lock file for "
                                  : "create pin file for ");
   const char *Msh = (What & mkLF ? "create lock files in "
                                  : "create pin files in ");
   char Resp;

// Get the actual pfn for the base file
//
   if (!Config.LocalPath(Lfn, Pfn, Pfnsz)) {finalRC = 4; return 0;}

// Get file state
//
   if (stat(Pfn, &Stat))
      {Emsg(errno, "create ", Msg, Lfn); return 0;}

// If this is not a directory, then all is well
//
   if ((Stat.st_mode & S_IFMT) != S_IFDIR)
      {if (!Opt.All) return 'f';
       Emsg(ENOTDIR, "create ", Msh, Lfn);
       return 0;
      }

// Make sure the whole directory is being considered
//
   if (Opt.All || Opt.Recurse) return 'd';

// Ask what we should do
//
   Msg = (What & mkLF ? "Apply makelf to ALL files in directory "
                      : "Apply pin to ALL files in directory ");
   if ((Resp = XrdFrmUtils::Ask('n', Msg, Lfn)) == 'y') return 'd';
   return (Resp == 'a' ? 'a' : 0);
}
