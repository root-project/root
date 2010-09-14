/******************************************************************************/
/*                                                                            */
/*                   X r d F r m A d m i n R e l o c . c c                    */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdFrmAdminRelockCVSID = "$Id$";

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <utime.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                                 R e l o c                                  */
/******************************************************************************/

int XrdFrmAdmin::Reloc(char *srcLfn, char *Space)
{
   static const int crOpts = (O_CREAT|O_EXCL)<<8;
   class relocRecover
        {public:
         char *Lfn;
               relocRecover() : Lfn(0) {}
              ~relocRecover() {if (Lfn) Config.ossFS->Unlink(Lfn);}
         } Recover;

   XrdOucTList   *pP;
   XrdOucEnv      myEnv;
   struct stat    srcStat, lokStat;
   struct utimbuf tBuff;
   char trgLfn[1032], trgPfn[1032], trgSpace[XrdOssSpace::minSNbsz];
   char srcLnk[1032], srcPfn[1032], srcSpace[XrdOssSpace::minSNbsz];
   char lokPfn[1032], ASize[32], *fsTarget = 0;
   int  rc, srcLsz = 0;

// Obtain the target space information, verify that it exists
//
   if (!(pP = ParseSpace(Space, &fsTarget))) return 4;
   strcpy(trgSpace, Space);
   if (fsTarget) *(fsTarget-1) = ':';

// Get the pfn for the incomming path
//
   if (!Config.LocalPath(srcLfn, srcPfn, sizeof(srcPfn)-8))
      {finalRC = 4; return 0;}

// Make sure the source file exists and get its attributes
//
   if (   lstat(srcPfn, &srcStat)) {Emsg(errno, "stat ", srcLfn); return 0;}
   if ((srcStat.st_mode & S_IFMT) == S_IFLNK)
      {if (stat(srcPfn, &srcStat)) {Emsg(errno, "stat ", srcLfn); return 0;}
       if ((srcLsz = readlink(srcPfn, srcLnk, sizeof(srcLnk)-1) < 0))
          {Emsg(errno, "read link ", srcLfn); return 0;}
       srcLnk[srcLsz] = '\0';
      } else *srcLnk = 0;
   XrdOssPath::getCname(srcPfn, srcSpace);

// Check this operation really makes sense
//
   if (!strcmp(srcSpace, trgSpace)
   ||  (fsTarget && !strncmp(fsTarget, srcLnk, strlen(fsTarget))))
      {Emsg(srcLfn, " already in space ", Space); return 0;}

// Get the current lock file time
//
   strcpy(lokPfn, srcPfn); strcat(lokPfn, ".lock");
   if (stat(lokPfn, &lokStat)) *lokPfn = '\0';

// Generate the target lfn and pfn
//
   strcpy(trgLfn, srcLfn); strcat(trgLfn, ".anew");
   if (!Config.LocalPath(trgLfn, trgPfn, sizeof(trgPfn)))
      {finalRC = 4; return 0;}

// Set environmental variables
//
   sprintf(ASize,"%lld", static_cast<long long>(srcStat.st_size));
   myEnv.Put("oss.asize", ASize);
   myEnv.Put("oss.cgroup",Space);

// Allocate a new file in the target space
//
   rc = Config.ossFS->Create("admin",trgLfn,srcStat.st_mode&S_IAMB,myEnv,crOpts);
   if (rc) {Emsg(rc, "create placeholder for ", trgLfn); return 0;}

// Now copy the source file to the target location. While we could possibly
// have done a rename, this could have potentially disrupted access to the file.
// Perform the reloc based on src/trg location
//
   Recover.Lfn = trgPfn;
   if (!RelocCP(srcPfn, trgPfn, srcStat.st_size)) return 0;

// Set the time of the file to it's original value
//
   tBuff.actime = srcStat.st_atime;
   tBuff.modtime= srcStat.st_mtime;
   if (utime(trgPfn, &tBuff)) Emsg(errno, "set mtime for ", trgPfn);

// Set the lock file time (do not let the reloc complete unless we can)
//
   if (*lokPfn)
      {strcpy(lokPfn, trgPfn); strcat(lokPfn, ".lock");
       tBuff.actime = lokStat.st_atime;
       tBuff.modtime= lokStat.st_mtime;
       if (utime(lokPfn, &tBuff)) Emsg(errno, "set mtime for ", srcLfn);
       return 0;
      }

// Rename the new file to the old file
//
   if ((rc = Config.ossFS->Rename(trgLfn, srcLfn)))
      {Emsg(-rc, "rename ", trgPfn); return 0;}
   Recover.Lfn = 0;

// Now adjust space as needed
//
   XrdOssSpace::Adjust(trgSpace,  srcStat.st_size, XrdOssSpace::Pstg);
   XrdOssSpace::Adjust(srcSpace, -srcStat.st_size, XrdOssSpace::Purg);

// If the source was another cache file syste, we need to remove the remnants
//
   if (srcLsz)
      {if (symlink(srcLnk, trgPfn))
          {Emsg(errno, "create symlink to ", srcLnk); return 0;}
       if ((rc = Config.ossFS->Unlink(trgLfn)))
          {Emsg(errno, "remove ", trgPfn); return 0;}
      }

// All done
//
   Msg(srcLfn, " relocated from space ", srcSpace, " to ", Space);
   return 0;
}

/******************************************************************************/
/*                               R e l o c C P                                */
/******************************************************************************/
  
int XrdFrmAdmin::RelocCP(const char *inFn, const char *outFn, off_t inSz)
{
   static const size_t segSize = 1024*1024;
   class ioFD
        {public:
         int FD;
             ioFD() : FD(-1) {}
            ~ioFD() {if (FD >= 0) close(FD);}
        } In, Out;

   char *inBuff, ioBuff[segSize], *bP;
   off_t  inOff=0, Offset=0, Size=inSz, outSize=segSize, inSize=segSize;
   size_t ioSize;
   ssize_t rLen;

// Open the input file
//
   if ((In.FD = open(inFn, O_RDONLY)) < 0)
      {Emsg(errno, "open ", inFn); return 1;}

// Open the output file
//
   if ((Out.FD = open(outFn, O_WRONLY)) < 0)
      {Emsg(errno, "open ", outFn); return 1;}

// We now copy 1MB segments using direct I/O
//
   ioSize = (Size < (int)segSize ? Size : segSize);
   while(Size)
        {if ((inBuff = (char *)mmap(0, ioSize, PROT_READ,
                       MAP_NORESERVE|MAP_PRIVATE, In.FD, Offset)) == MAP_FAILED)
            {Emsg(errno, "memory map ", inFn); break;}
         if (!RelocWR(outFn, Out.FD, inBuff, ioSize, Offset)) break;
         Size -= ioSize; Offset += ioSize;
         if (munmap(inBuff, ioSize) < 0)
            {Emsg(errno, "unmap memory for ", inFn); break;}
         if (Size < (int)segSize) ioSize = Size;
        }

// Return if all went well, otherwise check if we can recover
//
   if (!Size || Size != inSz) return Size == 0;
   Msg("Trying traditional copy....");

// Do a traditional copy
//
   inSize = (inSz < (int)segSize ? Size : segSize);
   while(Size)
        {if (Size < (int)ioSize) outSize = inSize = Size;
         bP = ioBuff;
         while(inSize)
              {if ((rLen = pread(In.FD, bP, inSize, inOff)) < 0)
                  {if (errno == EINTR) continue;
                      else {Emsg(errno, "read ", inFn); return 0;}
                  }
               bP += rLen; inSize -= rLen; inOff += rLen;
              }
         if (!RelocWR(outFn, Out.FD, ioBuff, outSize, Offset)) return 0;
         Size -= outSize; Offset += outSize;
        }

// Success
//
   return 1;
}

/******************************************************************************/
  
int XrdFrmAdmin::RelocWR(const char *outFn,
                         int oFD, char *Buff, size_t BLen, off_t BOff)
{
   ssize_t wLen;

// Copy out a segment
//
   while(BLen)
        {if ((wLen = pwrite(oFD, Buff, BLen, BOff)) < 0)
            {if (errno == EINTR) continue;
                else break;
            }
         Buff += wLen; BLen -= wLen; BOff += wLen;
        }

// Check for errors
//
   if (!BLen) return 1;
   Emsg(errno, "write ", outFn);
   return 0;
}
