/******************************************************************************/
/* XrdFfsWcache.cc simple write cache that captures consecutive small writes  */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

/* 
   When direct_io is not used, kernel will break large write to 4Kbyte  
   writes. This significantly reduces the writting performance. This 
   simple cache mechanism is to improve the performace on small writes. 

   Note that fuse 2.8.0 pre2 or above and kernel 2.6.27 or above provide
   a big_writes option to allow > 4KByte writing. It will make this 
   smiple write caching obsolete. 
*/
#define XrdFfsWcacheBufsize 131072

#if defined(__linux__)
/* For pread()/pwrite() */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#endif

#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>

#include <pthread.h>

#include "XrdFfs/XrdFfsWcache.hh"
#ifndef NOXRD
    #include "XrdFfs/XrdFfsPosix.hh"
#endif

#ifdef __cplusplus
  extern "C" {
#endif

struct XrdFfsWcacheFilebuf {
    off_t offset;
    size_t len;
    char *buf;
    pthread_mutex_t *mlock;
};

struct XrdFfsWcacheFilebuf *XrdFfsWcacheFbufs;

/* #include "xrdposix.h" */

int XrdFfsWcacheNFILES;
void XrdFfsWcache_init()
{
    int fd;
    struct rlimit rlp;

    getrlimit(RLIMIT_NOFILE, &rlp);
    XrdFfsWcacheNFILES = rlp.rlim_cur;
    XrdFfsWcacheNFILES = (XrdFfsWcacheNFILES == (int)RLIM_INFINITY? 4096 : XrdFfsWcacheNFILES);
    
/*    printf("%d %d\n", XrdFfsWcacheNFILES, sizeof(struct XrdFfsWcacheFilebuf)); */
    XrdFfsWcacheFbufs = (struct XrdFfsWcacheFilebuf*)malloc(sizeof(struct XrdFfsWcacheFilebuf) * XrdFfsWcacheNFILES);
    for (fd = 0; fd < XrdFfsWcacheNFILES; fd++)
    {
        XrdFfsWcacheFbufs[fd].offset = 0;
        XrdFfsWcacheFbufs[fd].len = 0;
        XrdFfsWcacheFbufs[fd].buf = NULL;
        XrdFfsWcacheFbufs[fd].mlock = NULL;
    }
}

int XrdFfsWcache_create(int fd) 
{
    XrdFfsWcache_destroy(fd);

    XrdFfsWcacheFbufs[fd].offset = 0;
    XrdFfsWcacheFbufs[fd].len = 0;
    XrdFfsWcacheFbufs[fd].buf = (char*)malloc(XrdFfsWcacheBufsize);
    if (XrdFfsWcacheFbufs[fd].buf == NULL)
        return 0;
    XrdFfsWcacheFbufs[fd].mlock = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (XrdFfsWcacheFbufs[fd].mlock == NULL)
        return 0;
    else
        pthread_mutex_init(XrdFfsWcacheFbufs[fd].mlock, NULL);
    return 1;
}

void XrdFfsWcache_destroy(int fd)
{
/*  XrdFfsWcache_flush(fd); */
    
    XrdFfsWcacheFbufs[fd].offset = 0;
    XrdFfsWcacheFbufs[fd].len = 0;
    if (XrdFfsWcacheFbufs[fd].buf != NULL) 
        free(XrdFfsWcacheFbufs[fd].buf);
    XrdFfsWcacheFbufs[fd].buf = NULL;
    if (XrdFfsWcacheFbufs[fd].mlock != NULL)
    {
        pthread_mutex_destroy(XrdFfsWcacheFbufs[fd].mlock);
        free(XrdFfsWcacheFbufs[fd].mlock);
    }
    XrdFfsWcacheFbufs[fd].mlock = NULL;
}

ssize_t XrdFfsWcache_flush(int fd)
{
    ssize_t rc;

    if (XrdFfsWcacheFbufs[fd].len == 0 || XrdFfsWcacheFbufs[fd].buf == NULL )
        return 0;

#ifndef NOXRD
    rc = XrdFfsPosix_pwrite(fd, XrdFfsWcacheFbufs[fd].buf, XrdFfsWcacheFbufs[fd].len, XrdFfsWcacheFbufs[fd].offset);
#else
    rc =     pwrite(fd, XrdFfsWcacheFbufs[fd].buf, XrdFfsWcacheFbufs[fd].len, XrdFfsWcacheFbufs[fd].offset);
#endif
    if (rc > 0)
    {
        XrdFfsWcacheFbufs[fd].offset = 0;
        XrdFfsWcacheFbufs[fd].len = 0;
    }
    return rc;
}

ssize_t XrdFfsWcache_pwrite(int fd, char *buf, size_t len, off_t offset)
{
    ssize_t rc;
    char *bufptr;

/* do not use caching under these cases */
    if (len > XrdFfsWcacheBufsize/2 || fd >= XrdFfsWcacheNFILES)
    {
#ifndef NOXRD
        rc = XrdFfsPosix_pwrite(fd, buf, len, offset);
#else 
        rc = pwrite(fd, buf, len, offset);
#endif
        return rc;
    }

    pthread_mutex_lock(XrdFfsWcacheFbufs[fd].mlock);
    rc = XrdFfsWcacheFbufs[fd].len;
/* 
   in the following two cases, a XrdFfsWcache_flush is required:
   1. current offset isnn't pointing to the tail of data in buffer
   2. adding new data will exceed the current buffer 
*/ 
    if (offset != (off_t)(XrdFfsWcacheFbufs[fd].offset + XrdFfsWcacheFbufs[fd].len) ||
        (off_t)(offset + len) > (XrdFfsWcacheFbufs[fd].offset + XrdFfsWcacheBufsize))
        rc = XrdFfsWcache_flush(fd);

    errno = 0;
    if (rc < 0) 
    {
        errno = ENOSPC;
        pthread_mutex_unlock(XrdFfsWcacheFbufs[fd].mlock);
        return -1;
    }

    bufptr = &XrdFfsWcacheFbufs[fd].buf[XrdFfsWcacheFbufs[fd].len];
    memcpy(bufptr, buf, len);
    if (XrdFfsWcacheFbufs[fd].len == 0)
        XrdFfsWcacheFbufs[fd].offset = offset;
    XrdFfsWcacheFbufs[fd].len += len;

    pthread_mutex_unlock(XrdFfsWcacheFbufs[fd].mlock);
    return (ssize_t)len;
}

#ifdef __cplusplus
  }
#endif
