#ifndef __XRDOUCCRC_HH__
#define __XRDOUCCRC_HH__
/******************************************************************************/
/*                                                                            */
/*                          X r d O u c C R C . h h                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//        $Id$

class XrdOucCRC
{
public:

static unsigned int CRC32(const unsigned char *rec, int reclen);

                    XrdOucCRC() {}
                   ~XrdOucCRC() {}

private:

static unsigned int crctable[256];
};
#endif
