#ifndef _XRDOSS_CONFIG_H
#define _XRDOSS_CONFIG_H
/******************************************************************************/
/*                                                                            */
/*                       X r d O s s C o n f i g . h h                        */
/*                                                                            */
/* (C) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC02-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$
  
#define  XRDOSS_VERSION "2.0.0"

/* Constant to indicate all went well.
*/
#ifndef XrdOssOK
#define XrdOssOK 0
#endif

// Flags set in OptFlags
//
#define XrdOss_USRPRTY   0x00000001
#define XrdOss_CacheFS   0x00000002

// Small structure to hold dual paths
//
struct  OssDPath
       {OssDPath *Next;
        char     *Path1;
        char     *Path2;
        OssDPath(OssDPath *dP,char *p1,char *p2) : Next(dP),Path1(p1),Path2(p2) {}
       };
#endif
