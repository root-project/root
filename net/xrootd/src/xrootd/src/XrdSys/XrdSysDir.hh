#ifndef __SYS_DIR_H__
#define __SYS_DIR_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d S y s D i r . h h                                */
/*                                                                            */
/* (c) 2006 G. Ganis (CERN)                                                   */
/*     All Rights Reserved. See XrdInfo.cc for complete License Terms         */
/******************************************************************************/
// $Id$

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdSysDir                                                            //
//                                                                      //
// Author: G. Ganis, CERN, 2006                                         //
//                                                                      //
// API for handling directories                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(WINDOWS)
#  include <sys/types.h>
#else
#  define uid_t unsigned int
#  define gid_t unsigned int
#endif

class XrdSysDir
{
 public:
   XrdSysDir(const char *path);
   virtual ~XrdSysDir();

   bool  isValid() { return (dhandle ? 1 : 0); }
   int   lastError() { return lasterr; }
   char *nextEntry();

 private:
   void  *dhandle;  // Directory handle
   int    lasterr;  // Error occured at last operation
};
#endif
