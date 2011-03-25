/******************************************************************************/
/* xrootdfs.cc  FUSE based file system interface to Xrootd Storage Cluster    */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#define FUSE_USE_VERSION 26

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>

#if defined(__linux__)
/* For pread()/pwrite() */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#endif

#ifdef HAVE_FUSE
#include <fuse.h>
#include <fuse/fuse_opt.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/time.h>
#include <pthread.h>
#include <pwd.h>
#include <libgen.h>
#include <syslog.h>
#include <signal.h>
#if !defined(__solaris__)
#include <sys/xattr.h>
#endif

#include "XrdFfs/XrdFfsPosix.hh"
#include "XrdFfs/XrdFfsMisc.hh"
#include "XrdFfs/XrdFfsWcache.hh"
#include "XrdFfs/XrdFfsQueue.hh"
//#include "XrdFfs/XrdFfsDent.hh"
#include "XrdFfs/XrdFfsFsinfo.hh"
#include "XrdPosix/XrdPosixXrootd.hh"

struct XROOTDFS {
    char *rdr;
    char *cns;
    char *fastls;
    char *daemon_user;
    bool ofsfwd;
};

struct XROOTDFS xrootdfs;
static struct fuse_opt xrootdfs_opts[10];

enum { OPT_KEY_HELP, OPT_KEY_SECSSS, };

/*
char *rdr, *cns, *fastls="", *daemon_user;
//enum Boolean {false, true} ofsfwd;
bool ofsfwd;
*/

static void* xrootdfs_init(struct fuse_conn_info *conn)
{
    struct passwd *pw;

    if (xrootdfs.daemon_user != NULL)
    {
        pw = getpwnam(xrootdfs.daemon_user);
        setgid((gid_t)pw->pw_gid);
        setuid((uid_t)pw->pw_uid);
    }

/*
   From FAQ:
      Miscellaneous threads should be started from the init() method.
   Threads started before fuse_main() will exit when the process goes
   into the background.
*/

#ifndef NOUSE_QUEUE
    if (getenv("XROOTDFS_NWORKERS") != NULL)
        XrdFfsQueue_create_workers(atoi(getenv("XROOTDFS_NWORKERS")));
    else
        XrdFfsQueue_create_workers(4);

    syslog(LOG_INFO, "INFO: Starting %d workers", XrdFfsQueue_count_workers());
#else
    syslog(LOG_INFO, "INFO: Not compiled to use task queue");
#endif
    return NULL;
}

static int xrootdfs_getattr(const char *path, struct stat *stbuf)
{
//  int res, fd;
    int res;
    char rootpath[1024];
    uid_t user_uid, uid;
    gid_t user_gid, gid;

    user_uid = fuse_get_context()->uid;
    uid = getuid();

    user_gid = fuse_get_context()->gid;
    gid = getgid();

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);

    rootpath[0]='\0';
/*
    if (xrootdfs.cns != NULL && xrootdfs.fastls != NULL)
        strcat(rootpath,xrootdfs.cns);
    else
        strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);

//    setegid(fuse_get_context()->gid);
//    seteuid(fuse_get_context()->uid);

    res = XrdFfsPosix_stat(rootpath, stbuf);
*/

    if (xrootdfs.cns != NULL && xrootdfs.fastls != NULL)
    {
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_stat(rootpath, stbuf);
    }
    else
        res = XrdFfsPosix_statall(xrootdfs.rdr, path, stbuf, fuse_get_context()->uid);

//    seteuid(getuid());
//    setegid(getgid());

//    stbuf->st_uid = user_uid;
//    stbuf->st_gid = user_gid;

    if (res == 0)
    {
        if (S_ISREG(stbuf->st_mode))
        {
/*
   By adding the following 'if' block, 'xrootdfs.fastls = RDR' will force XrootdFS to check
   with redirector for file status info (not directory). 

   Some applicatios such as SRM may do extensive file or directory existence checking.
   These applications can't tolerant slow responding on file or directory info (if
   don't exist). They also expect correct file size. For this type of application, we
   can set 'xrootdfs.fastls = RDR'.

   Allowing multi-thread may solve this problem. However, XrootdFS crashs under some 
   situation, and we have to add -s (single thread) option when runing XrootdFS.
 */
            if (xrootdfs.cns != NULL && xrootdfs.fastls != NULL && strcmp(xrootdfs.fastls,"RDR") == 0)
            {
                rootpath[0]='\0';
                strcat(rootpath,xrootdfs.rdr);
                strcat(rootpath,path);
                XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
                XrdFfsPosix_stat(rootpath, stbuf);
//                stbuf->st_uid = user_uid;
//                stbuf->st_gid = user_gid;
            }
            stbuf->st_mode |= 0666;
            stbuf->st_mode &= 0772777;  /* remove sticky bit and suid bit */
            stbuf->st_blksize = 32768;  /* unfortunately, it is ignored, see include/fuse.h */
            return 0;
        }
        else if (S_ISDIR(stbuf->st_mode))
        {
            stbuf->st_mode |= 0777;
            stbuf->st_mode &= 0772777;  /* remove sticky bit and suid bit */
            return 0;
        }
        else
            return -EIO;
    }
    else if (res == -1 && xrootdfs.cns != NULL && xrootdfs.fastls != NULL)
        return -errno;
    else if (xrootdfs.cns == NULL)
        return -errno;
    else
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_stat(rootpath, stbuf);
//        stbuf->st_uid = user_uid;
//        stbuf->st_gid = user_gid;
        if (res == -1)
            return -errno;
        else 
        {
            if (S_ISREG(stbuf->st_mode))
                return -ENOENT;
            else if (S_ISDIR(stbuf->st_mode))
            {
                stbuf->st_mode |= 0777;
                stbuf->st_mode &= 0772777;
                return 0;
            }
            else
                return -EIO;
        }
    }
}

static int xrootdfs_access(const char *path, int mask)
{
/*
    int res;
    res = access(path, mask);
    if (res == -1)
        return -errno;
*/
    return 0;
}

static int xrootdfs_readlink(const char *path, char *buf, size_t size)
{
/*
    int res;

    res = readlink(path, buf, size - 1);
    if (res == -1)
        return -errno;

    buf[res] = '\0';
*/
    return 0;
}

static int xrootdfs_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                       off_t offset, struct fuse_file_info *fi)
{
    DIR *dp;
    struct dirent *de;

    (void) offset;
    (void) fi;

    char rootpath[1024];

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
/* 
   if CNS server is not defined, there is no way to list files in a directory
   because we don't know the data nodes
*/
    if (xrootdfs.cns != NULL)
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        dp = XrdFfsPosix_opendir(rootpath);
        if (dp == NULL)
            return -errno;
                                                                                                                                               
        while ((de = XrdFfsPosix_readdir(dp)) != NULL)
        {
/*
            struct stat st;
            memset(&st, 0, sizeof(st));
            st.st_ino = de->d_ino;
            st.st_mode = de->d_type << 12;
 */
            if (filler(buf, de->d_name, NULL, 0))
                break;
        }
        XrdFfsPosix_closedir(dp);
        return 0;
    }
    else  /* if there is no CNS, try collect dirents from all known data servers. */
    {
         int i, n;
         char **dnarray;

         n = XrdFfsPosix_readdirall(xrootdfs.rdr, path, &dnarray, fuse_get_context()->uid);

         for (i = 0; i < n; i++)
             if (filler(buf, dnarray[i], NULL, 0)) break;

/* 
  this loop should not be merged with the above loop because all members of 
  dnarray[] should be freed, or there will be memory leak.
 */
         for (i = 0; i < n; i++) 
             free(dnarray[i]);
         free(dnarray); 

         return -errno;
    }
}

static int xrootdfs_mknod(const char *path, mode_t mode, dev_t rdev)
{
    int res;

    /* On Linux this could just be 'mknod(path, mode, rdev)' but this
       is more portable */
    char rootpath[1024];

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (S_ISREG(mode))
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.rdr);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
/* 
   Around May 2008, the O_EXCL was added to the _open(). No reason was given. It is removed again 
   due to the following reason (the situation that redirector thinks a file exist while it doesn't):

   1. FUSE will use _getattr to determine file status. _mknod() will be called only if _getattr() 
      determined that the file does not exist.
   2. In the case that rootd security is enabled, if a user create a file at an unauthorized path 
      (and fail), redirector thinks the files exist but it actually does't exist (enabling security 
      on redirector doesn't seems to help. An authorized user won't be able to create the same file
      until the redirector forgets about it.

        res = XrdFfsPosix_open(rootpath, O_CREAT | O_EXCL | O_WRONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH); 
*/
        res = XrdFfsPosix_open(rootpath, O_CREAT | O_WRONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH); 
        if (res == -1)
            return -errno;
        XrdFfsPosix_close(res);
/* We have to make sure CNS file is created as well, otherwise, xrootdfs_getattr()
   may or may not find this file (due to multi-threads) */
        if (xrootdfs.cns == NULL)
            return 0;

        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_open(rootpath, O_CREAT | O_EXCL, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH); 
        XrdFfsPosix_close(res);
/* We actually don't need to care about error by this XrdFfsPosix_open() */
    }
    return 0;
}

static int xrootdfs_mkdir(const char *path, mode_t mode)
{
    int res;
    char rootpath[1024];
/*  
    Posix Mkdir() fails on the current version of Xrootd, 20071101-0808p1 
    So we avoid doing that. This is fixed in CVS head version.
*/
/*
    if CNS is defined, only mkdir() on CNS. Otherwise, mkdir() on redirector
 */
    rootpath[0]='\0';

    if (xrootdfs.cns != NULL)
        strcat(rootpath,xrootdfs.cns);
    else
        strcat(rootpath,xrootdfs.rdr);

    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);

    res = XrdFfsPosix_mkdir(rootpath, mode);
    if (res == 0) return 0;
/* 
   now we are here either because there is either a race to create the directory, or the redirector 
   incorrectly cached a non-existing one (see _mknod() for more explaitation)

   the following code try to clear the redirector cache. In the case of two racing mkdir(), it doesn't 
   care which one will success/fail.
*/
    XrdFfsPosix_clear_from_rdr_cache(rootpath);

    res = XrdFfsPosix_mkdir(rootpath, mode);
    return ((res == -1)? -errno : 0);
}

static int xrootdfs_unlink(const char *path)
{
    int res;
    char rootpath[1024];

    rootpath[0]='\0';
    strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (xrootdfs.ofsfwd == true)
    {
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_unlink(rootpath);
    }
    else
        res = XrdFfsPosix_unlinkall(xrootdfs.rdr, path, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (xrootdfs.cns != NULL && xrootdfs.ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_unlink(rootpath);
        if (res == -1)
            return -errno;
    }
    return 0;
}

static int xrootdfs_rmdir(const char *path)
{
    int res;
//  struct stat stbuf;
    char rootpath[1024];

    rootpath[0]='\0';
    strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (xrootdfs.ofsfwd == true)
    { 
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_rmdir(rootpath);
    }
    else
        res = XrdFfsPosix_rmdirall(xrootdfs.rdr, path, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (xrootdfs.cns != NULL && xrootdfs.ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_rmdir(rootpath);
        if (res == -1)
            return -errno;
    }
    /* 
      clear cache in redirector. otherwise, an immediate mkdir(path) will fail
    if (xrootdfs.ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.rdr);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        XrdFfsPosix_clear_from_xrootdfs.rdr_cache(rootpath);  // no needed. _mkdir() is doing this.
    }
     */
    return 0;
}

static int xrootdfs_symlink(const char *from, const char *to)
{
/*
    int res;

    res = symlink(from, to);
    if (res == -1)
        return -errno;
*/
    return -EIO;
}

static int xrootdfs_rename(const char *from, const char *to)
{
    int res;
    char from_path[1024], to_path[1024];
    struct stat stbuf;

    from_path[0]='\0';
    strcat(from_path, xrootdfs.rdr);
    strcat(from_path, from);

    to_path[0]='\0';
    strcat(to_path, xrootdfs.rdr);
    strcat(to_path, to);
/*
  1. do actual renaming on data servers if if is a file in order to speed up
     renaming
  2. return -EXDEV for renaming of directory so that files in the directory
     are renamed individually (in order for the .pfn pointing back correctly).
 */

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    XrdFfsMisc_xrd_secsss_editurl(from_path, fuse_get_context()->uid);

    XrdFfsPosix_stat(from_path, &stbuf);
    if (S_ISDIR(stbuf.st_mode)) /* && xrootdfs.cns == NULL && xrootdfs.ofsfwd == false) */
        return -EXDEV;

    if (xrootdfs.ofsfwd == true)
        res = XrdFfsPosix_rename(from_path, to_path);
    else
        res = XrdFfsPosix_renameall(xrootdfs.rdr, from, to, fuse_get_context()->uid);

    if (res == -1)
        return -errno;
    
/* data servers may not notify redirector about the renaming. So we notify redirector */
    XrdFfsPosix_clear_from_rdr_cache(from_path);

    if (xrootdfs.cns != NULL && xrootdfs.ofsfwd == false)
    {
        from_path[0]='\0';
        strcat(from_path, xrootdfs.cns);
        strcat(from_path, from);

        to_path[0]='\0';
        strcat(to_path, xrootdfs.cns);
        strcat(to_path, to);

        res = XrdFfsPosix_rename(from_path, to_path);
        if (res == -1)
            return -errno;
    }
    return 0;

/*  return -EXDEV */
}

static int xrootdfs_link(const char *from, const char *to)
{
/*
    int res;

    res = link(from, to);
    if (res == -1)
        return -errno;
*/
    return -EMLINK;
}

static int xrootdfs_chmod(const char *path, mode_t mode)
{
/*
    int res;

    res = chmod(path, mode);
    if (res == -1)
        return -errno;
*/
    return 0;
}

static int xrootdfs_chown(const char *path, uid_t uid, gid_t gid)
{
/*
    int res;

    res = lchown(path, uid, gid);
    if (res == -1)
        return -errno;
*/
    return 0;
}

/* _ftruncate() will only work with kernel >= 2.6.15. See FUSE ChangeLog */
static int xrootdfs_ftruncate(const char *path, off_t size,
                              struct fuse_file_info *fi)
{
    int fd, res;
//  char rootpath[1024];
                                                                                                                                           
    fd = (int) fi->fh;
    XrdFfsWcache_flush(fd);
    res = XrdFfsPosix_ftruncate(fd, size);
    if (res == -1)
        return -errno;
                                                                                                                              
/* 
   There is no need to update the size of the CNS shadow file now. That
   should be updated when the file is closed 

    if (xrootdfs.cns != NULL) 
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);

        res = XrdFfsPosix_truncate(rootpath, size);
        if (res == -1)
            return -errno;
    }
*/
    return 0;
}

static int xrootdfs_truncate(const char *path, off_t size)
{
    int res;
    char rootpath[1024];

    rootpath[0]='\0';
    strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (xrootdfs.ofsfwd == true)
    {
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_truncate(rootpath, size);
    }
    else
        res = XrdFfsPosix_truncateall(xrootdfs.rdr, path, size, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (xrootdfs.cns != NULL && xrootdfs.ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,xrootdfs.cns);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_truncate(rootpath, size);
        if (res == -1)
            return -errno;
    }
    return 0;
}

static int xrootdfs_utimens(const char *path, const struct timespec ts[2])
{
/*
    int res;
    struct timeval tv[2];

    tv[0].tv_sec = ts[0].tv_sec;
    tv[0].tv_usec = ts[0].tv_nsec / 1000;
    tv[1].tv_sec = ts[1].tv_sec;
    tv[1].tv_usec = ts[1].tv_nsec / 1000;

    res = utimes(path, tv);
    if (res == -1)
        return -errno;
*/
    return 0;
}

static int xrootdfs_open(const char *path, struct fuse_file_info *fi)
{
    int res;
    char rootpath[1024]="";
    strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
    res = XrdFfsPosix_open(rootpath, fi->flags, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
    if (res == -1)
        return -errno;

    fi->fh = res;
    XrdFfsWcache_create(fi->fh);
    return 0;
}

static int xrootdfs_read(const char *path, char *buf, size_t size, off_t offset,
                    struct fuse_file_info *fi)
{
    int fd;
    int res;

    fd = (int) fi->fh;
    XrdFfsWcache_flush(fd);  /* in case is the file is reading/writing */
    res = XrdFfsPosix_pread(fd, buf, size, offset);
    if (res == -1)
        res = -errno;

    return res;
}

static int xrootdfs_write(const char *path, const char *buf, size_t size,
                     off_t offset, struct fuse_file_info *fi)
{
    int fd;
    int res;

/* 
   File already existed. FUSE uses xrootdfs_open() and xrootdfs_truncate() to open and
   truncate a file before calling xrootdfs_write() 
*/
    fd = (int) fi->fh;
//    res = XrdFfsPosix_pwrite(fd, buf, size, offset);
    res = XrdFfsWcache_pwrite(fd, (char *)buf, size, offset);
    if (res == -1)
        res = -errno;

    return res;
}

static int xrootdfs_statfs(const char *path, struct statvfs *stbuf)
{
    int res;
//  char rootpath[1024], xattr[256];
//  char *token, *key, *value;
//  char *lasts_xattr[256], *lasts_tokens[128];
//  long long size;

//    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
#ifndef __macos__
    stbuf->f_bsize = 1024;
#else
    stbuf->f_bsize = 1024 * 128; // work around 32 bit fsblkcnt_t in struct statvfs on Mac OSX
    stbuf->f_frsize = stbuf->f_bsize; // seems there are other limitations, 1024*128 is a max we set
#endif

//    res = XrdFfsPosix_statvfsall(xrootdfs.rdr, path, stbuf, fuse_get_context()->uid);
    res = XrdFfsFsinfo_cache_search(&XrdFfsPosix_statvfsall, xrootdfs.rdr, path, stbuf, fuse_get_context()->uid);

/*
    stbuf->f_blocks /= stbuf->f_bsize;
    stbuf->f_bavail /= stbuf->f_bsize;
    stbuf->f_bfree /= stbuf->f_bsize;
*/
    return res;
/*
    stbuf->f_bsize = 16384;
    stbuf->f_blocks = 1048576;
    stbuf->f_bfree = stbuf->f_blocks;
    stbuf->f_bavail = stbuf->f_blocks;

    if (xrootdfs.cns == NULL) return 0; 

    rootpath[0]='\0';
    strcat(rootpath,xrootdfs.cns);
    strcat(rootpath,path);

    res = XrdFfsPosix_getxattr(rootpath, "xroot.space", xattr, 256);
    if (res == -1)
        return 0;
    else 
    {
        token = strtok_r(xattr, "&", lasts_xattr);
        while (token != NULL)
        {
             token = strtok_r(NULL, "&", lasts_xattr); 
             if (token == NULL) break;
             key = strtok_r(token, "=", lasts_tokens);
             value = strtok_r(NULL, "=", lasts_tokens);
             if (!strcmp(key,"oss.used"))
             {
                  sscanf((const char*)value, "%lld", &size);
                  stbuf->f_bavail = size / stbuf->f_bsize;
             }
             else if (!strcmp(key,"oss.quota"))
             {
                  sscanf((const char*)value, "%lld", &size);
                  stbuf->f_blocks = size / stbuf->f_bsize;
             }
        }
        stbuf->f_bavail = stbuf->f_blocks - stbuf->f_bavail;
        stbuf->f_bfree = stbuf->f_bavail;
    }
    return 0;
 */
}

static int xrootdfs_release(const char *path, struct fuse_file_info *fi)
{
    /* Just a stub.  This method is optional and can safely be left
       unimplemented */

    int fd, oflag;
    struct stat xrdfile, cnsfile;
    char rootpath[1024];

    fd = (int) fi->fh;
    XrdFfsWcache_flush(fd);
    XrdFfsWcache_destroy(fd);
    XrdFfsPosix_close(fd);
    fi->fh = 0;
/* 
   Return at here because the current version of Cluster Name Space daemon 
   doesn't implement the 'truncate' functon we originally planned.

    return 0;
*/
    if (xrootdfs.cns == NULL || (fi->flags & 0100001) == (0100000 | O_RDONLY))
        return 0;

    int res;
    char xattr[256], xrdtoken[256];
    char *token, *key, *value;
    char *lasts_xattr[256], *lasts_tokens[128];

    rootpath[0]='\0';
    strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);
/*
 * Get xrootd token info from data nodes. And set the token info on CNS
 */
    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
    xrdtoken[0]='\0';
    res = XrdFfsPosix_getxattr(rootpath, "xroot.xattr", xattr, 256);
    if (res != -1)
    {
        token = strtok_r(xattr, "&", lasts_xattr);
        while (token != NULL)
        {
             key = strtok_r(token, "=", lasts_tokens);
             value = strtok_r(NULL, "=", lasts_tokens);
             if (!strcmp(key,"oss.cgroup")) 
                 strcpy(xrdtoken, value);

             if (!strcmp(key,"oss.used"))
                {long long llVal;
                 sscanf((const char*)value, "%lld", &llVal);
                 xrdfile.st_size = llVal;
                }
             token = strtok_r(NULL, "&", lasts_xattr);
        }
    }
    else
    {
        XrdFfsPosix_stat(rootpath,&xrdfile);
    }

    rootpath[0]='\0';
    strcat(rootpath,xrootdfs.cns);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
    if (xrdtoken[0] != '\0' && strstr(path,"?oss.cgroup=") == NULL)
    {
        strcat(rootpath,"?oss.cgroup=");
        strcat(rootpath,xrdtoken);
    }

    if (XrdFfsPosix_stat(rootpath,&cnsfile) == -1)
        oflag = O_CREAT|O_WRONLY;
    else
        oflag = O_TRUNC|O_WRONLY;

/* 
   This creates a file on CNS with the right size. But it is actually an empty file.
   It doesn't use disk space, only inodes.
*/
    if (cnsfile.st_size != xrdfile.st_size)
    {
        fd = XrdFfsPosix_open(rootpath,oflag,S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
        if (fd >= 0) 
        {
            XrdFfsPosix_lseek(fd,(off_t)xrdfile.st_size-1,SEEK_SET);
            XrdFfsPosix_write(fd,"",1);
            XrdFfsPosix_close(fd);
        }
    }

    return 0;
}

static int xrootdfs_fsync(const char *path, int isdatasync,
                     struct fuse_file_info *fi)
{
    /* Just a stub.  This method is optional and can safely be left
       unimplemented */

    (void) path;
    (void) isdatasync;
    (void) fi;
    return 0;
}

/* xattr operations are optional and can safely be left unimplemented */
static int xrootdfs_setxattr(const char *path, const char *name, const char *value,
                        size_t size, int flags)
{
    if (fuse_get_context()->uid != 0 && fuse_get_context()->uid != getuid())
        return -EPERM;

    if (!strcmp(name,"xrootdfs.fs.dataserverlist"))
    {
        XrdFfsMisc_refresh_url_cache(xrootdfs.rdr);
        XrdFfsMisc_logging_url_cache(NULL);
    }
    else if (!strcmp(name,"xrootdfs.fs.nworkers"))
    {
        int i, j;
        char *tmp_value;
        tmp_value=strdup(value);
        if (size > 0) tmp_value[size] = '\0';

        i = XrdFfsQueue_count_workers();
        j = atoi(tmp_value);
        free(tmp_value);
        if (j > i) 
            XrdFfsQueue_create_workers( j-i );
        if (j < i)
            XrdFfsQueue_remove_workers( i-j ); // XrdFfsQueue_remove_workers() will wait until workers are removed.
        j = XrdFfsQueue_count_workers();
#ifndef NOUSE_QUEUE
        syslog(LOG_INFO, "INFO: Adjust the number of workers from %d to %d", i, j);
#endif
    }
    return 0;
}

static int xrootdfs_getxattr(const char *path, const char *name, char *value,
                    size_t size)
{
    int xattrlen;
    char rootpath[1024]="";
    char rooturl[1024]="";

    if (!strcmp(name,"xroot.url"))
    {
        errno = 0;
        strcat(rootpath,xrootdfs.rdr);
        strcat(rootpath,path);

//        XrdFfsMisc_get_current_url(rootpath, rooturl);
        strcat(rooturl, rootpath);

        if (size == 0)
            return strlen(rooturl);
        else if (size >= strlen(rooturl)) 
        {
            size = strlen(rooturl);
            if (size != 0) 
            {
                value[0] = '\0';
                strcat(value, rooturl);
            }
            return size;
        }
        else
        {
            errno = ERANGE;
            return -1;
        }
    }
    else if (!strcmp(name, "xrootdfs.fs.dataserverlist"))
    {
        char *hostlist;

        hostlist = (char*) malloc(sizeof(char) * XrdFfs_MAX_NUM_NODES * 256);
        XrdFfsMisc_get_list_of_data_servers(hostlist);

        if (size == 0)
        {
            xattrlen = strlen(hostlist);
            free(hostlist);
            return xattrlen;
        }
        else if (size >= strlen(hostlist))
        {
            size = strlen(hostlist);
            if (size != 0)
            {
                value[0] = '\0';
                strcat(value, hostlist);
            }
            free(hostlist);
            return size;
        }
        else
        {
            errno = ERANGE;
            free(hostlist);
            return -1;
        }
    }
    else if (!strcmp(name, "xrootdfs.fs.nworkers"))
    {
        char nworkers[7];
        int n;
        n = XrdFfsQueue_count_workers();
        sprintf(nworkers, "%d", n);

        if (size == 0)
            return strlen(nworkers);
        else if (size >= strlen(nworkers))
        {
             size = strlen(nworkers);
             if (size != 0)
             {
                  value[0] = '\0';
                  strcat(value, nworkers);
             }
             return size;
        }
        else
        {
            errno = ERANGE;
            return -1;
        }
    }

    if (xrootdfs.cns != NULL)
        strcat(rootpath,xrootdfs.cns);
    else
        strcat(rootpath,xrootdfs.rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
    xattrlen = XrdFfsPosix_getxattr(rootpath, name, value, size);
    if (xattrlen == -1)
        return -errno;
    else
        return xattrlen;
}

static int xrootdfs_listxattr(const char *path, char *list, size_t size)
{
/*
    int res = llistxattr(path, list, size);
    if (res == -1)
        return -errno;
    return res;
*/
    return 0;
}

static int xrootdfs_removexattr(const char *path, const char *name)
{
/*
    int res = lremovexattr(path, name);
    if (res == -1)
        return -errno;
*/
    return 0;
}

void xrootdfs_sigusr1_handler(int sig) 
{
/* Do this in a new thread because XrdFfsMisc_refresh_url_cache() contents mutex. */
    pthread_t *thread;
    pthread_attr_t attr;
    size_t stacksize = 2*1024*1024;

    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, stacksize);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    thread = (pthread_t*) malloc(sizeof(pthread_t));
    pthread_create(thread, &attr, (void* (*)(void*))XrdFfsMisc_logging_url_cache, xrootdfs.rdr);
    pthread_detach(*thread);
    free(thread);

    pthread_attr_destroy(&attr);
}

static struct fuse_operations xrootdfs_oper; 

static void xrootdfs_usage(const char *progname)
{
    fprintf(stderr,
"usage: %s mountpoint options\n"
"\n"
"XrootdFS options:\n"
"    -h -help --help        print help\n"
"\n"
"[Required]\n"
"    -rdr=redirector_url    root URL of the Xrootd redirector\n"
"\n"
"[Optional]\n"
"    -cns=cns_server_url    root URL of the CNS server\n"
"    -R=username            cause XrootdFS to switch effective uid to that of username if possible\n"
"    -fastls=RDR            set to RDR when CNS is presented will cause stat() to go to redirector\n"
"    -sss                   use Xrootd seciruty module \"sss\"\n"
"\n", progname);
}

static int xrootdfs_opt_proc(void* data, const char* arg, int key, struct fuse_args* outargs) 
{
    (void) data;
    (void) outargs;

//    printf("hellow key %d arg %s\n", key, arg);
    switch (key) {
      case FUSE_OPT_KEY_OPT:
        return 1;
      case FUSE_OPT_KEY_NONOPT:
        return 1;
      case OPT_KEY_SECSSS:  
/* this specify using "sss" security module. the actually location of the key is 
   determined by shell enviornment variable XrdSecsssKT (or default locations). 
   The location of the key should not appear in command line. */
        setenv("XROOTDFS_SECMOD", "sss", 1);
        return 0;
      case OPT_KEY_HELP:
        xrootdfs_usage(outargs->argv[0]);
        fuse_opt_add_arg(outargs, "-ho");
        fuse_main(outargs->argc, outargs->argv, &xrootdfs_oper, NULL);
        exit(1);
      default:
        return(-1); 
        ;
    }
}

int main(int argc, char *argv[])
{
    static XrdPosixXrootd abc; // Do one time init for posix interface

    xrootdfs_oper.init		= xrootdfs_init;
    xrootdfs_oper.getattr	= xrootdfs_getattr;
    xrootdfs_oper.access	= xrootdfs_access;
    xrootdfs_oper.readlink	= xrootdfs_readlink;
    xrootdfs_oper.readdir	= xrootdfs_readdir;
    xrootdfs_oper.mknod		= xrootdfs_mknod;
    xrootdfs_oper.mkdir		= xrootdfs_mkdir;
    xrootdfs_oper.symlink	= xrootdfs_symlink;
    xrootdfs_oper.unlink	= xrootdfs_unlink;
    xrootdfs_oper.rmdir		= xrootdfs_rmdir;
    xrootdfs_oper.rename	= xrootdfs_rename;
    xrootdfs_oper.link		= xrootdfs_link;
    xrootdfs_oper.chmod		= xrootdfs_chmod;
    xrootdfs_oper.chown		= xrootdfs_chown;
    xrootdfs_oper.ftruncate	= xrootdfs_ftruncate;
    xrootdfs_oper.truncate	= xrootdfs_truncate;
    xrootdfs_oper.utimens	= xrootdfs_utimens;
    xrootdfs_oper.open		= xrootdfs_open;
    xrootdfs_oper.read		= xrootdfs_read;
    xrootdfs_oper.write		= xrootdfs_write;
    xrootdfs_oper.statfs	= xrootdfs_statfs;
    xrootdfs_oper.release	= xrootdfs_release;
    xrootdfs_oper.fsync		= xrootdfs_fsync;
    xrootdfs_oper.setxattr	= xrootdfs_setxattr;
    xrootdfs_oper.getxattr	= xrootdfs_getxattr;
    xrootdfs_oper.listxattr	= xrootdfs_listxattr;
    xrootdfs_oper.removexattr	= xrootdfs_removexattr;

/* Define XrootdFS options */
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);

    xrootdfs_opts[0].templ = "-rdr=%s";
    xrootdfs_opts[0].offset = offsetof(struct XROOTDFS, rdr);
    xrootdfs_opts[0].value = 0;

    xrootdfs_opts[1].templ = "-cns=%s";
    xrootdfs_opts[1].offset = offsetof(struct XROOTDFS, cns);
    xrootdfs_opts[1].value = 0;

    xrootdfs_opts[2].templ = "-fastls=%s";
    xrootdfs_opts[2].offset = offsetof(struct XROOTDFS, fastls);
    xrootdfs_opts[2].value = 0;

    xrootdfs_opts[3].templ = "-R=%s";
    xrootdfs_opts[3].offset = offsetof(struct XROOTDFS, daemon_user);
    xrootdfs_opts[3].value = 0;

    xrootdfs_opts[4].templ = "-ofsfwd=%s";
    xrootdfs_opts[4].offset = offsetof(struct XROOTDFS, ofsfwd);
    xrootdfs_opts[4].value = 0;

    xrootdfs_opts[5].templ = "-sss";
    xrootdfs_opts[5].offset = -1U;
    xrootdfs_opts[5].value = OPT_KEY_SECSSS;

    xrootdfs_opts[6].templ = "-h";
    xrootdfs_opts[6].offset = -1U;
    xrootdfs_opts[6].value = OPT_KEY_HELP;

    xrootdfs_opts[7].templ = "-help";
    xrootdfs_opts[7].offset = -1U;
    xrootdfs_opts[7].value = OPT_KEY_HELP;

    xrootdfs_opts[8].templ = "--help";
    xrootdfs_opts[8].offset = -1U;
    xrootdfs_opts[8].value = OPT_KEY_HELP;

    xrootdfs_opts[9].templ = NULL;

/* initialize struct xrootdfs */
//    memset(&xrootdfs, 0, sizeof(xrootdfs));
    xrootdfs.rdr = NULL;
    xrootdfs.cns = NULL;
    xrootdfs.fastls = NULL;
    xrootdfs.daemon_user = NULL;
    xrootdfs.ofsfwd = false;

/* Get options from environment variables first */
    xrootdfs.rdr = getenv("XROOTDFS_RDRURL");
    xrootdfs.cns = getenv("XROOTDFS_CNSURL");
    xrootdfs.fastls = getenv("XROOTDFS_FASTLS");
// If this is defined, XrootdFS will setuid/setgid to this user at xrootdfs_init().
    xrootdfs.daemon_user = getenv("XROOTDFS_USER");
    if (getenv("XROOTDFS_OFSFWD") != NULL && ! strcmp(getenv("XROOTDFS_OFSFWD"),"1")) xrootdfs.ofsfwd = true;

/* Parse XrootdFS options, will overwrite those defined in environment variables */
    fuse_opt_parse(&args, &xrootdfs, xrootdfs_opts, xrootdfs_opt_proc);

/* make sure xrootdfs.rdr is specified */
    if (xrootdfs.rdr == NULL)
    {
        argc = 2;
        argv[1] = strdup("-h");
        struct fuse_args xargs = FUSE_ARGS_INIT(argc, argv);
        fuse_opt_parse(&xargs, &xrootdfs, xrootdfs_opts, xrootdfs_opt_proc);
    }

/* convert xroot://... to root://... */
    if (xrootdfs.rdr != NULL && xrootdfs.rdr[0] == 'x') xrootdfs.rdr += 1;
    if (xrootdfs.cns != NULL && xrootdfs.cns[0] == 'x') xrootdfs.cns += 1;
/*
    XROOTDFS_OFSFWD (ofs.fwd (and ofs.fwd 3way) on rm, rmdir, mv, truncate.
    if this is not defined, XrootdFS will go to CNS and each data server
    and do the work.

    If CNS is not defined, we have to set ofsfwd to false, or we will not be able to 
    get a return status of rm, rmdir, mv and truncate.
 */
    if (xrootdfs.cns == NULL) xrootdfs.ofsfwd = false; 

    XrdFfsMisc_xrd_init(xrootdfs.rdr,0);
    XrdFfsWcache_init();
    signal(SIGUSR1,xrootdfs_sigusr1_handler);

    umask(0);
    return fuse_main(args.argc, args.argv, &xrootdfs_oper, NULL);
}
#else

int main(int argc, char *argv[])
{
    printf("This platform does not have FUSE; xrootdfs cannot be started!");
    exit(13);
}
#endif
