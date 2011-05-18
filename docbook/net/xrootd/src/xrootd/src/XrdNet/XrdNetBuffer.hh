#ifndef __NET_BUFF__
#define __NET_BUFF__
/******************************************************************************/
/*                                                                            */
/*                       X r d N e t B u f f e r . h h                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include <stdlib.h>

#include "XrdOuc/XrdOucChain.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                         X r d N e t B u f f e r Q                          */
/******************************************************************************/

class XrdNetBuffer;
  
class XrdNetBufferQ
{
public:

       XrdNetBuffer  *Alloc();

inline int            BuffSize(void) {return size;}

       void           Recycle(XrdNetBuffer *bp);

       void           Set(int maxb);

       XrdNetBufferQ(int bsz, int maxb=16);
      ~XrdNetBufferQ();

       int                       alignit;
       XrdSysMutex               BuffList;
       XrdOucStack<XrdNetBuffer> BuffStack;
       int                       maxbuff;
       int                       numbuff;
       int                       size;
};

/******************************************************************************/
/*                          X r d N e t B u f f e r                           */
/******************************************************************************/

class XrdNetBuffer
{
friend class XrdNetBufferQ;

public:
       char         *data;
       int           dlen;

inline int           BuffSize(void) {return BuffQ->BuffSize();}

       void          Recycle(void)  {BuffQ->Recycle(this);}

      XrdNetBuffer(XrdNetBufferQ *bq);
     ~XrdNetBuffer() {if (data) free(data);}

private:

      XrdOucQSItem<XrdNetBuffer> BuffLink;
      XrdNetBufferQ             *BuffQ;
};
#endif
