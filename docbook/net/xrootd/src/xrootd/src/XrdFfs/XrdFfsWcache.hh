/******************************************************************************/
/* XrdFfsWcache.hh simple write cache that captures consecutive small writes  */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#ifdef __cplusplus
  extern "C" {
#endif

void    XrdFfsWcache_init();
int     XrdFfsWcache_create(int fd);
void    XrdFfsWcache_destroy(int fd);
ssize_t  XrdFfsWcache_flush(int fd);
ssize_t  XrdFfsWcache_pwrite(int fd, char *buf, size_t len, off_t offset);

#ifdef __cplusplus
  }
#endif
