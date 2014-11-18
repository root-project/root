/******************************************************************************/
/* XrdFfsFsinfo.hh filesystem/xrootd oss space usage info cache               */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2010)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#include <sys/statvfs.h>

#ifdef __cplusplus
  extern "C" {
#endif

int XrdFfsFsinfo_cache_search(int (*func)(const char*, const char*, struct statvfs*, uid_t), const char* rdrurl, const char* path, struct statvfs *stbuf, uid_t user_uid);

#ifdef __cplusplus
  }
#endif
