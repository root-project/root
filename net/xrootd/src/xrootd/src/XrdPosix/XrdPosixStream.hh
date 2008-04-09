#ifndef __XRDPOSIXSTREAM__
#define __XRDPOSIXSTREAM__
/******************************************************************************/
/*                                                                            */
/*                     X r d P o s i x S t r e a m . h h                      */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//           $Id$

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "XrdSys/XrdSysPthread.hh"

class XrdPosixStream
{
public:

       int   Fclose(FILE *stream);

       FILE *Fopen(const char *path, const char *mode);

inline int   myFD(int fildes)
                 {return (fildes > MaxFiles || !myFiled[fildes])
                         ? fildes : myFiled[fildes];
                 }
      XrdPosixStream();
     ~XrdPosixStream() {}

private:

XrdSysMutex FileMutex;

static const int MaxFiles = 256;

             int myFiled[MaxFiles];
};
#endif
