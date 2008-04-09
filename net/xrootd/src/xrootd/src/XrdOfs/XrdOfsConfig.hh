#ifndef _OFS_CONFIG_H
#define _OFS_CONFIG_H
/******************************************************************************/
/*                                                                            */
/*                       X r d O f s C o n f i g . h h                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//         $Id$

#include <errno.h>

/******************************************************************************/
/*              c o m m o n   c o n f i g   p a r a m e t e r s               */
/******************************************************************************/

// Set the following three parameters as follows:
//
// XrdOfsFDMAXOPEN   Number of open files before we start FD idle scan
// XrdOfsFDMINIDLE   Minimum number of seconds between idle scans
// XrdOfsFDMAXIDLE   Maximum number of seconds before file is closed
// XrdOfsFDMAXUSER   Maximum number of users per file descriptor (0 -> no limit)
//
#define XrdOfsFDOPENMAX       9
#define XrdOfsFDMINIDLE     120
#define XrdOfsFDMAXIDLE    1200
#define XrdOfsFDMAXUSER       0
#define XrdOfsLOCKTRIES       3
#define XrdOfsLOCKWAIT      333

/******************************************************************************/
/*                     E x e c u t i o n   O p t i o n s                      */
/******************************************************************************/

// The following flags are set in the Options file system variable
//
#define XrdOfsAUTHORIZE    0x0001
#define XrdOfsFDNOSHARE    0x0002

#define XrdOfsREDIREER     0x0050
#define XrdOfsREDIROXY     0x0020
#define XrdOfsREDIRRMT     0x0040
#define XrdOfsREDIRTRG     0x0080
#define XrdOfsREDIRVER     0x00C0
#define XrdOfsREDIRECT     0x00F0

#define XrdOfsFWD          0x0100

/******************************************************************************/
/*                         M i s c e l l a n e o u s                          */
/******************************************************************************/
  
#define XrdOfsENOTOPEN EBADF
#endif
