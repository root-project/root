/******************************************************************************/
/*                                                                            */
/*                     X r d P o s i x S t r e a m . c c                      */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//           $Id$

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdSys/XrdSysPthread.hh"
#include "XrdPosix/XrdPosixExtern.hh"
#include "XrdPosix/XrdPosixLinkage.hh"
#include "XrdPosix/XrdPosixStream.hh"
#include "XrdPosix/XrdPosixXrootd.hh"
 
/******************************************************************************/
/*                   G l o b a l   D e c l a r a t i o n s                    */
/******************************************************************************/
  
extern XrdPosixLinkage Xunix;

extern XrdPosixRootVec xinuX;

       XrdPosixStream  streamX;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdPosixStream::XrdPosixStream()
{

   memset(myFiled, 0, sizeof(myFiled));
}

/******************************************************************************/
/*                                F c l o s e                                 */
/******************************************************************************/

int XrdPosixStream::Fclose(FILE *stream)
{
   int nullfd = fileno(stream);

// Close the associated file
//
   if (nullfd < MaxFiles && myFiled[nullfd])
      {FileMutex.Lock();
       if (myFiled[nullfd]) xinuX.Close(myFiled[nullfd]);
           myFiled[nullfd] = 0;
       FileMutex.UnLock();
      }

// Now close the stream
//
   return Xunix.Fclose(stream);
}

/******************************************************************************/
/*                                 F o p e n                                  */
/******************************************************************************/

#define ISMODE(x) !strcmp(mode, x)
  
FILE *XrdPosixStream::Fopen(const char *path, const char *mode)
{
   int nullfd, fd, omode;
   FILE *stream;

// Translate the mode flags
//
        if (ISMODE("r")  || ISMODE("rb"))                   omode = O_RDONLY;
   else if (ISMODE("w")  || ISMODE("wb"))                   omode = O_WRONLY;
   else if (ISMODE("a")  || ISMODE("ab"))                   omode = O_APPEND;
   else if (ISMODE("r+") || ISMODE("rb+") || ISMODE("r+b")) omode = O_RDWR;
   else if (ISMODE("w+") || ISMODE("wb+") || ISMODE("w+b")) omode = O_RDWR;
// else if (ISMODE("a+") || ISMODE("ab+") || ISMODE("a+b")) omode = unsupported;
   else {errno = EINVAL; return 0;}

// First obtain a free stream
//
   if (!(stream = fopen("/dev/null", mode))) return stream;
   nullfd = fileno(stream);

// Now open the file
//
   if ((fd = xinuX.Open(path, omode)) < 0) {fclose(stream); return 0;}

// Record the slot number
//
   FileMutex.Lock();
   myFiled[nullfd] = fd;
   FileMutex.UnLock();

// All done
//
   return stream;
}
