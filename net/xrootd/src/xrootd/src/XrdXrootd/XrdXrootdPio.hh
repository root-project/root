#ifndef __XRDXROOTDPIO__
#define __XRDXROOTDPIO__
/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d P i o . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$
  
#include "XProtocol/XPtypes.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdXrootdFile;

class XrdXrootdPio
{
public:

       XrdXrootdPio      *Next;
       XrdXrootdFile     *myFile;
       long long          myOffset;
       int                myIOLen;
       kXR_char           StreamID[2];
       char               isWrite;

static XrdXrootdPio      *Alloc(int n=1);

inline XrdXrootdPio      *Clear(XrdXrootdPio *np=0)
                               {static const kXR_char zed[2] = {0,0};
                                Set(0, 0, 0, zed,'\0');
                                Next = np; return this;
                               }

       void               Recycle();

inline void               Set(XrdXrootdFile *theFile, long long theOffset,
                             int theIOLen, const kXR_char *theSID, char theW)
                             {myFile      = theFile;
                              myOffset    = theOffset;
                              myIOLen     = theIOLen;
                              StreamID[0] = theSID[0]; StreamID[1] = theSID[1];
                              isWrite     = theW;
                             }

                          XrdXrootdPio(XrdXrootdPio *np=0) {Clear(np);}
                         ~XrdXrootdPio() {}

private:

static const int          FreeMax = 256;
static XrdSysMutex        myMutex;
static XrdXrootdPio      *Free;
static int                FreeNum;
};
#endif
