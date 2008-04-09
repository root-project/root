#ifndef __XrdBuffer_H__
#define __XrdBuffer_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d B u f f e r . h h                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$ 

#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                            x r d _ B u f f e r                             */
/******************************************************************************/

class XrdBuffer
{
public:

char *   buff;     // -> buffer
int      bsize;    // size of this buffer

         XrdBuffer(char *bp, int sz, int ix)
                      {buff = bp; bsize = sz; bindex = ix; next = 0;}

        ~XrdBuffer() {if (buff) free(buff);}

         friend class XrdBuffManager;
private:

XrdBuffer *next;
       int  bindex;
static int  pagesz;
};
  
/******************************************************************************/
/*                       x r d _ B u f f M a n a g e r                        */
/******************************************************************************/

#define XRD_BUCKETS 12
#define XRD_BUSHIFT 10

// There should be only one instance of this class per buffer pool.
//
  
class XrdBuffManager
{
public:

void        Init();

XrdBuffer  *Obtain(int bsz);

int         Recalc(int bsz);

void        Release(XrdBuffer *bp);

int         MaxSize() {return maxsz;}

void        Reshape();

void        Set(int maxmem=-1, int minw=-1);

int         Stats(char *buff, int blen, int do_sync=0);

            XrdBuffManager(int minrst=20*60);

           ~XrdBuffManager() {} // The buffmanager is never deleted

private:

const int  slots;
const int  shift;
const int  pagsz;
const int  maxsz;

struct {XrdBuffer *bnext;
        int         numbuf;
        int         numreq;
       } bucket[XRD_BUCKETS];          // 1K to 1<<(szshift+slots-1)M buffers

int       totreq;
int       totbuf;
long long totalo;
long long maxalo;
int       minrsw;
int       rsinprog;
int       totadj;

XrdSysCondVar      Reshaper;
static const char *TraceID;
};
#endif
