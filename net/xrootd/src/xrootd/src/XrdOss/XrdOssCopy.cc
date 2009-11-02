/******************************************************************************/
/*                                                                            */
/*                         X r d O s s C o p y . c c                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$
 
const char *XrdOssCopyCVSID = "$Id$";

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
  
#include "XrdOss/XrdOssCopy.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*                  E r r o r   R o u t i n g   O b j e c t                   */
/******************************************************************************/
  
extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;

/******************************************************************************/
/* Public:                          C o p y                                   */
/******************************************************************************/
  
off_t XrdOssCopy::Copy(const char *inFn, const char *outFn, int outFD)
{
   static const size_t segSize = 1024*1024;
   class ioFD
        {public:
         int FD;
             ioFD(int fd=-1) : FD(fd) {}
            ~ioFD() {if (FD >= 0) close(FD);}
        } In, Out(outFD);

   struct utimbuf tBuff;
   struct stat buf;
   char *inBuff, *bP;
   off_t  Offset=0, fileSize;
   size_t ioSize, copySize;
   ssize_t rLen;

// Open the input file
//
   if ((In.FD = open(inFn, O_RDONLY)) < 0)
      {OssEroute.Emsg("Copy", errno, "open", inFn); return -1;}

// Get the input filesize
//
   if (fstat(In.FD, &buf))
      {OssEroute.Emsg("Copy", errno, "stat", outFn); return -1;}
   copySize = fileSize = buf.st_size;

// We now copy 1MB segments using direct I/O
//
   ioSize = (fileSize < (off_t)segSize ? fileSize : segSize);
   while(copySize)
        {if ((inBuff = (char *)mmap(0, ioSize, PROT_READ, 
                       MAP_NORESERVE|MAP_PRIVATE, In.FD, Offset)) == MAP_FAILED)
            {OssEroute.Emsg("Copy", errno, "memory map", inFn); break;}
         if (!Write(outFn, Out.FD, inBuff, ioSize, Offset)) break;
         copySize -= ioSize; Offset += ioSize;
         if (munmap(inBuff, ioSize) < 0)
            {OssEroute.Emsg("Copy", errno, "unmap memory for", inFn); break;}
         if (copySize < segSize) ioSize = copySize;
        }

// Return if all went well, otherwise check if we can recover
//
   if (!copySize)            return fileSize;
   if ((off_t)copySize != fileSize) return -1;
   OssEroute.Emsg("Copy", "Trying traditional copy for", inFn, "...");

// Do a traditional copy (note that we didn't copy anything yet)
//
  {char ioBuff[segSize];
   off_t rdSize, wrSize = segSize, inOff=0;
   while(copySize)
        {if (copySize < segSize) rdSize = wrSize = copySize;
            else rdSize = segSize;
         bP = ioBuff;
         while(rdSize)
              {do {rLen = pread(In.FD, bP, rdSize, inOff);}
                  while(rLen < 0 && errno == EINTR);
               if (rLen <= 0)
                  {OssEroute.Emsg("Copy",rLen ? errno : ECANCELED,"read",inFn);
                   return -1;
                  }
               bP += rLen; rdSize -= rLen; inOff += rLen;
              }
         if (!Write(outFn, Out.FD, ioBuff, wrSize, Offset)) return -1;
         copySize -= wrSize; Offset += wrSize;
        }
   }

// Now set the time on the file to the original time
//
   tBuff.actime = buf.st_atime;
   tBuff.modtime= buf.st_mtime;
   if (utime(outFn, &tBuff)) 
      OssEroute.Emsg("Copy", errno, "set mtime for", outFn);

// Success
//
   return fileSize;
}

/******************************************************************************/
/* private:                        W r i t e                                  */
/******************************************************************************/
  
int XrdOssCopy::Write(const char *outFn,
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
   OssEroute.Emsg("Copy", errno, "write", outFn);
   return 0;
}
