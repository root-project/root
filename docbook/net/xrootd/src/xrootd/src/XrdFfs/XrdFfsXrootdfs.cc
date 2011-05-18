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

#if defined(__linux__)
/* For pread()/pwrite() */
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif
#endif

#ifdef HAVE_FUSE
#include <fuse.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/time.h>
#include <pthread.h>
#include <pwd.h>
#include <libgen.h>
#include <syslog.h>
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

char *rdr, *cns, *fastls="", *daemon_user;
//enum Boolean {false, true} ofsfwd;
bool ofsfwd;

static void* xrootdfs_init(struct fuse_conn_info *conn)
{
    struct passwd *pw;

    if (daemon_user != NULL)
    {
        pw = getpwnam(daemon_user);
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
    if (cns != NULL && fastls != NULL)
        strcat(rootpath,cns);
    else
        strcat(rootpath,rdr);
    strcat(rootpath,path);

//    setegid(fuse_get_context()->gid);
//    seteuid(fuse_get_context()->uid);

    res = XrdFfsPosix_stat(rootpath, stbuf);
*/

    if (cns != NULL && fastls != NULL)
    {
        strcat(rootpath,cns);
        strcat(rootpath,path);
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_stat(rootpath, stbuf);
    }
    else
        res = XrdFfsPosix_statall(rdr, path, stbuf, fuse_get_context()->uid);

//    seteuid(getuid());
//    setegid(getgid());

//    stbuf->st_uid = user_uid;
//    stbuf->st_gid = user_gid;

    if (res == 0)
    {
        if (S_ISREG(stbuf->st_mode))
        {
/*
   By adding the following 'if' block, 'fastls = RDR' will force XrootdFS to check
   with redirector for file status info (not directory). 

   Some applicatios such as SRM may do extensive file or directory existence checking.
   These applications can't tolerant slow responding on file or directory info (if
   don't exist). They also expect correct file size. For this type of application, we
   can set 'fastls = RDR'.

   Allowing multi-thread may solve this problem. However, XrootdFS crashs under some 
   situation, and we have to add -s (single thread) option when runing XrootdFS.
 */
            if (cns != NULL && fastls != NULL && strcmp(fastls,"RDR") == 0)
            {
                rootpath[0]='\0';
                strcat(rootpath,rdr);
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
    else if (res == -1 && cns != NULL && fastls != NULL)
        return -errno;
    else if (cns == NULL)
        return -errno;
    else
    {
        rootpath[0]='\0';
        strcat(rootpath,cns);
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
    if (cns != NULL)
    {
        rootpath[0]='\0';
        strcat(rootpath,cns);
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

         n = XrdFfsPosix_readdirall(rdr, path, &dnarray, fuse_get_context()->uid);

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
        strcat(rootpath,rdr);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_open(rootpath, O_CREAT | O_EXCL | O_WRONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH); 
//        res = XrdFfsPosix_open(rootpath, O_CREAT | O_WRONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH); 
        if (res == -1)
            return -errno;
        XrdFfsPosix_close(res);
/* We have to make sure CNS file is created as well, otherwise, xrootdfs_getattr()
   may or may not find this file (due to multi-threads) */
        if (cns == NULL)
            return 0;

        rootpath[0]='\0';
        strcat(rootpath,cns);
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

    if (cns != NULL)
        strcat(rootpath,cns);
    else
        strcat(rootpath,rdr);

    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
    res = XrdFfsPosix_mkdir(rootpath, mode);
    return ((res == -1)? -errno : 0);
}

static int xrootdfs_unlink(const char *path)
{
    int res;
    char rootpath[1024];

    rootpath[0]='\0';
    strcat(rootpath,rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (ofsfwd == true)
    {
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_unlink(rootpath);
    }
    else
        res = XrdFfsPosix_unlinkall(rdr, path, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (cns != NULL && ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,cns);
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
    strcat(rootpath,rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (ofsfwd == true)
    { 
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_rmdir(rootpath);
    }
    else
        res = XrdFfsPosix_rmdirall(rdr, path, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (cns != NULL && ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,cns);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_rmdir(rootpath);
        if (res == -1)
            return -errno;
    }
    /* 
      clear cache in redirector. otherwise, an immediate mkdir(path) will fail
     */
    if (ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,rdr);
        strcat(rootpath,path);

        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        XrdFfsPosix_rmdir(rootpath);
    }
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
    strcat(from_path, rdr);
    strcat(from_path, from);

    to_path[0]='\0';
    strcat(to_path, rdr);
    strcat(to_path, to);
/*
  As of 2009-11-19(20), the CVS head has a bug in XrdFfsPosix_rename() fixed so that 
  the removal of old file and creation of new file is notified to redirector
  (to update its cache). Until this is bundled in main stream xrootd releases,
  we will just retuen a -EXDEV for now.

  After the main stream xrootd includes this fix, the ideal way is:
  1. do actual renaming on data servers if if is a file in order to speed up
     renaming
  2. return -EXDEV for renaming of directory so that files in the directory
     are renamed individually (in order for the .pfn pointing back correctly).
 */
    return -EXDEV;

    XrdFfsPosix_stat(from_path, &stbuf);
    if (S_ISDIR(stbuf.st_mode)) /* && cns == NULL && ofsfwd == false) */
        return -EXDEV;

    if (ofsfwd == true)
        res = XrdFfsPosix_rename(from_path, to_path);
    else
        res = XrdFfsPosix_renameall(rdr, from, to, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (cns != NULL && ofsfwd == false)
    {
        from_path[0]='\0';
        strcat(from_path, cns);
        strcat(from_path, from);

        to_path[0]='\0';
        strcat(to_path, cns);
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

    if (cns != NULL) 
    {
        rootpath[0]='\0';
        strcat(rootpath,cns);
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
    strcat(rootpath,rdr);
    strcat(rootpath,path);

    XrdFfsMisc_xrd_secsss_register(fuse_get_context()->uid, fuse_get_context()->gid);
    if (ofsfwd == true)
    {
        XrdFfsMisc_xrd_secsss_editurl(rootpath, fuse_get_context()->uid);
        res = XrdFfsPosix_truncate(rootpath, size);
    }
    else
        res = XrdFfsPosix_truncateall(rdr, path, size, fuse_get_context()->uid);

    if (res == -1)
        return -errno;

    if (cns != NULL && ofsfwd == false)
    {
        rootpath[0]='\0';
        strcat(rootpath,cns);
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
    strcat(rootpath,rdr);
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

//    res = XrdFfsPosix_statvfsall(rdr, path, stbuf, fuse_get_context()->uid);
    res = XrdFfsFsinfo_cache_search(&XrdFfsPosix_statvfsall, rdr, path, stbuf, fuse_get_context()->uid);

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

    if (cns == NULL) return 0; 

    rootpath[0]='\0';
    strcat(rootpath,cns);
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
    if (cns == NULL || (fi->flags & 0100001) == (0100000 | O_RDONLY))
        return 0;

    int res;
    char xattr[256], xrdtoken[256];
    char *token, *key, *value;
    char *lasts_xattr[256], *lasts_tokens[128];

    rootpath[0]='\0';
    strcat(rootpath,rdr);
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
    strcat(rootpath,cns);
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
        int i;
        char *hostlist, *p1, *p2;

        XrdFfsMisc_refresh_url_cache(rdr);

        hostlist = (char*) malloc(sizeof(char) * XrdFfs_MAX_NUM_NODES * 1024);
        i = XrdFfsMisc_get_list_of_data_servers(hostlist);

        syslog(LOG_INFO, "INFO: Will use the following %d data servers", i);
        p1 = hostlist;
        p2 = strchr(p1, '\n');
        while (p2 != NULL)
        {
            p2[0] = '\0';
            syslog(LOG_INFO, "   %s", p1);
            p1 = p2 +1;
            p2 = strchr(p1, '\n');
        }
        free(hostlist);
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
        strcat(rootpath,rdr);
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
        char hostlist[4096 * 1024];
        XrdFfsMisc_get_list_of_data_servers(hostlist);

        if (size == 0)
            return strlen(hostlist);
        else if (size >= strlen(hostlist))
        {
             size = strlen(hostlist);
             if (size != 0)
             {
                  value[0] = '\0';
                  strcat(value, hostlist);
             }
             return size;
        }
        else
        {
            errno = ERANGE;
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

    if (cns != NULL)
        strcat(rootpath,cns);
    else
        strcat(rootpath,rdr);
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

static struct fuse_operations xrootdfs_oper; 

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

    rdr = getenv("XROOTDFS_RDRURL");
    cns = getenv("XROOTDFS_CNSURL");
/* convert xroot://... to root://... */
    if (rdr[0] == 'x') rdr = rdr+1;
    if (cns != NULL && cns[0] == 'x') cns = cns+1;

    fastls = getenv("XROOTDFS_FASTLS");
/*
    If this is defined, XrootdFS will setuid/setgid to this user at xrootdfs_init().
 */
    daemon_user = getenv("XROOTDFS_USER");
/*
    XROOTDFS_OFSFWD (ofs.fwd (and ofs.fwd 3way) on rm, rmdir, mv, truncate.
    if this is not defined, XrootdFS will go to CNS and each data server
    and do the work.

    If CNS is not defined, we have to set ofsfwd to false, or we will not be able to 
    get a return status of rm, rmdir, mv and truncate.
 */
    if (cns != NULL && getenv("XROOTDFS_OFSFWD") != NULL && ! strcmp(getenv("XROOTDFS_OFSFWD"),"1"))
        ofsfwd = true;
    else
        ofsfwd = false;

    XrdFfsMisc_xrd_init(rdr,0);
    XrdFfsWcache_init();

    umask(0);
    return fuse_main(argc, argv, &xrootdfs_oper, NULL);
}
#else

int main(int argc, char *argv[])
{
    printf("This platform does not have FUSE; xrootdfs cannot be started!");
    exit(13);
}
#endif
