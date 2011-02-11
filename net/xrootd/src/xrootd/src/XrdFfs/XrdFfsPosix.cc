/******************************************************************************/
/* XrdFfsPosix.cc C wrapper to some of the Xrootd Posix library functions     */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#if !defined(__solaris__)
#include <sys/xattr.h> 
#endif

#ifndef ENOATTR 
  #define ENOATTR ENODATA 
#endif

#include <iostream>
#include <libgen.h>
#include <unistd.h>
#include <stdlib.h>
#include <syslog.h>
#include "XrdFfs/XrdFfsPosix.hh"
#include "XrdPosix/XrdPosixXrootd.hh"
#include "XrdFfs/XrdFfsMisc.hh"
#include "XrdFfs/XrdFfsDent.hh"
#include "XrdFfs/XrdFfsQueue.hh"

#ifdef __cplusplus
  extern "C" {
#endif

int XrdFfsPosix_stat(const char *path, struct stat *buf)
{
    int rc; 
    errno = 0;
    rc = XrdPosixXrootd::Stat(path, buf);
    if (rc == 0 && S_ISBLK(buf->st_mode)) /* If 'buf' come from HPSS, xrootd will return it as a block device! */
    {                                     /* So we re-mark it to a regular file */
        buf->st_mode &= 0007777;
        if ( buf->st_mode & S_IXUSR )
            buf->st_mode |= 0040000;   /* a directory */
        else
            buf->st_mode |= 0100000;   /* a file */
    }
    return rc;
}

DIR *XrdFfsPosix_opendir(const char *path)
{
    return XrdPosixXrootd::Opendir(path);
}

struct dirent *XrdFfsPosix_readdir(DIR *dirp)
{
    return XrdPosixXrootd::Readdir(dirp);
}

int XrdFfsPosix_closedir(DIR *dirp)
{
    return XrdPosixXrootd::Closedir(dirp);
}

int XrdFfsPosix_mkdir(const char *path, mode_t mode)
{
    return XrdPosixXrootd::Mkdir(path, mode);
}

int XrdFfsPosix_rmdir(const char *path)
{
/* Note: Xrootd returns ENOSYS rather than ENOTEMPTY when a directory is not empty */
    return XrdPosixXrootd::Rmdir(path);
}

int XrdFfsPosix_open(const char *path, int oflags, mode_t mode)
{
    return XrdPosixXrootd::Open(path, oflags, mode);
}

int XrdFfsPosix_close(int fildes)
{
    return XrdPosixXrootd::Close(fildes);
}

off_t XrdFfsPosix_lseek(int fildes, off_t offset, int whence)
{
    return XrdPosixXrootd::Lseek(fildes, (long long)offset, whence);
}

ssize_t XrdFfsPosix_read(int fildes, void *buf, size_t nbyte)
{
    return XrdPosixXrootd::Read(fildes, buf, nbyte);
}

ssize_t XrdFfsPosix_pread(int fildes, void *buf, size_t nbyte, off_t offset)
{
    return XrdPosixXrootd::Pread(fildes, buf, nbyte, (long long)offset);
}

ssize_t XrdFfsPosix_write(int fildes, const void *buf, size_t nbyte)
{
    return XrdPosixXrootd::Write(fildes, buf, nbyte);
}

ssize_t XrdFfsPosix_pwrite(int fildes, const void *buf, size_t nbyte, off_t offset)
{
    return XrdPosixXrootd::Pwrite(fildes, buf, nbyte, (long long) offset);
}

int XrdFfsPosix_fsync(int fildes)
{
    return XrdPosixXrootd::Fsync(fildes);
}

int XrdFfsPosix_unlink(const char *path)
{
    return XrdPosixXrootd::Unlink(path);
}

int XrdFfsPosix_rename(const char *oldpath, const char *newpath)
{
    return XrdPosixXrootd::Rename(oldpath, newpath);
}

int XrdFfsPosix_ftruncate(int fildes, off_t offset)
{
    return XrdPosixXrootd::Ftruncate(fildes, offset);
}
int XrdFfsPosix_truncate(const char *path, off_t Size)
{
    return XrdPosixXrootd::Truncate(path, Size);
}

long long XrdFfsPosix_getxattr(const char *path, const char *name, void *value, unsigned long long size)
{
    int bufsize;
    char xattrbuf[1024], nameclass[128], *namesubclass;
    char *token, *key, *val;
    char *lasts_xattr[256], *lasts_tokens[128];

/*
    Xrootd only support two names: xroot.space and xroot.xattr. We add support of xroot.space.*
    such as xroot.space.oss.cgroup etc.
 */
    strncpy(nameclass, name, 11);
    nameclass[11] = '\0';

    if (strcmp(nameclass, "xroot.space") != 0 && strcmp(nameclass, "xroot.xattr") != 0)
    {
        errno = ENOATTR;
        return -1;
    }

    bufsize = XrdPosixXrootd::Getxattr(path, nameclass, xattrbuf, size);
    if (bufsize == -1) return -1;

    if (strlen(name) > 11) 
    {
        strcpy(nameclass, name);
        namesubclass = &nameclass[12];
    }
    else  /* xroot.space or xroot.xattr is provided. */ 
    {
        strcpy((char*)value, xattrbuf);
        return bufsize;
    }

    token = strtok_r(xattrbuf, "&", lasts_xattr);
    while ( token != NULL )
    {
         key = strtok_r(token, "=", lasts_tokens);
         val = strtok_r(NULL, "=", lasts_tokens);
         if (! strcmp(key, namesubclass))
         {
              strcpy((char*)value, val);
              return strlen(val);
         }
         token = strtok_r(NULL, "&", lasts_xattr);
    }    
    errno = ENOATTR; 
    return -1;
}

/* Posix IO functions to operation on all data servers */

struct XrdFfsPosixX_deleteall_args {
    char *url;
    int *res;
    int *err;
    mode_t st_mode;
};

void* XrdFfsPosix_x_deleteall(void *x)
{
    struct XrdFfsPosixX_deleteall_args* args = (struct XrdFfsPosixX_deleteall_args*) x;

    if (S_ISREG(args->st_mode))
        *(args->res) = XrdFfsPosix_unlink(args->url);
    else if (S_ISDIR(args->st_mode))
        *(args->res) = XrdFfsPosix_rmdir(args->url);

    *(args->err) = errno;
    return NULL;
}
        
int XrdFfsPosix_deleteall(const char *rdrurl, const char *path, uid_t user_uid, mode_t st_mode)
{
    int i, nurls, res, rval;
    char *newurls[XrdFfs_MAX_NUM_NODES];
    int res_i[XrdFfs_MAX_NUM_NODES];
    int errno_i[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsPosixX_deleteall_args args[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsQueueTasks *jobs[XrdFfs_MAX_NUM_NODES];

    nurls = XrdFfsMisc_get_all_urls(rdrurl, newurls, XrdFfs_MAX_NUM_NODES);
    if (nurls < 0) rval = -1;

    for (i = 0; i < nurls; i++)
    {
        errno_i[i] = 0;
        strcat(newurls[i],path);
        XrdFfsMisc_xrd_secsss_editurl(newurls[i], user_uid);
        args[i].url = newurls[i];
        args[i].err = &errno_i[i];
        args[i].res = &res_i[i];
        args[i].st_mode = st_mode;
#ifdef NOUSE_QUEUE
        XrdFfsPosix_x_deleteall((void*) &args[i]);
    }
#else
        jobs[i] = XrdFfsQueue_create_task(XrdFfsPosix_x_deleteall, (void**)(&args[i]), 0);
    }
    for (i = 0; i < nurls; i++)
    {
        XrdFfsQueue_wait_task(jobs[i]);
        XrdFfsQueue_free_task(jobs[i]);
    }
#endif
    res = -1;
    errno = ENOENT;
    for (i = 0; i < nurls; i++)
        if (res_i[i] == 0)
        {
            res = 0;
            errno = 0;
        }
        else if (res_i[i] != 0 && errno_i[i] == 125) // host is down
        {
            res = -1;
            errno = ETIMEDOUT;
            syslog(LOG_WARNING, "WARNING: unlink/rmdir(%s) failed (connection timeout)", newurls[i]);
            break;
        }
        else if (res_i[i] != 0 && errno_i[i] != ENOENT)
        {
            res = -1;
            errno = errno_i[i];
            syslog(LOG_WARNING, "WARNING: unlink/rmdir(%s) failed (errno = %d)", newurls[i], errno);
            break;
        }

    for (i = 0; i < nurls; i++)
        free(newurls[i]);

    return res; 
}

int XrdFfsPosix_unlinkall(const char *rdrurl, const char *path, uid_t user_uid)
{
    return XrdFfsPosix_deleteall(rdrurl, path, user_uid, S_IFREG);
}

int XrdFfsPosix_rmdirall(const char *rdrurl, const char *path, uid_t user_uid)
{
    return XrdFfsPosix_deleteall(rdrurl, path, user_uid, S_IFDIR);
}

int XrdFfsPosix_renameall(const char *rdrurl, const char *from, const char *to, uid_t user_uid)
{
    int i, nurls, res, rval = 0;
    struct stat stbuf;
    char fromurl[1024], tourl[1024], *newurls[XrdFfs_MAX_NUM_NODES];

    nurls = XrdFfsMisc_get_all_urls(rdrurl, newurls, XrdFfs_MAX_NUM_NODES);
    if (nurls < 0) rval = -1;

    for (i = 0; i < nurls; i++)
    {
        errno = 0;

        fromurl[0]='\0';
        strcat(fromurl, newurls[i]);
        strcat(fromurl, from);
        tourl[0]='\0';
        strcat(tourl, newurls[i]);
        strcat(tourl, to);

        XrdFfsMisc_xrd_secsss_editurl(fromurl, user_uid);
        XrdFfsMisc_xrd_secsss_editurl(tourl, user_uid);
        res = (XrdFfsPosix_stat(fromurl, &stbuf));
        if (res == 0)
        {
/* XrdFfsPosix_rename doesn't need this protection
            newdir = strdup(tourl);
            newdir = dirname(newdir);
            if (XrdFfsPosix_stat(newdir, &stbuf) == -1)
                XrdFfsPosix_mkdir(newdir, 0777);

            free(newdir);
*/
            rval = XrdFfsPosix_rename(fromurl, tourl);
            if (rval == -1) 
            {
                syslog(LOG_WARNING, "WARNING: rename(%s, %s) failed (errno = %d)", fromurl, tourl, errno);
                break;
            }
/* well, it will be messy if a successful rename is followed by a failed one */
        } 
    }

    for (i = 0; i < nurls; i++)
        free(newurls[i]);

    if (rval != 0 && errno == 0) errno = EIO;
    return rval;
}

int XrdFfsPosix_truncateall(const char *rdrurl, const char *path, off_t size, uid_t user_uid)
{
    int i, nurls, res, rval = 0;
    struct stat stbuf;
    char *newurls[XrdFfs_MAX_NUM_NODES];

    nurls = XrdFfsMisc_get_all_urls(rdrurl, newurls, XrdFfs_MAX_NUM_NODES);
    if (nurls < 0) rval = -1;

    for (i = 0; i < nurls; i++)
    {
        errno = 0;
        strcat(newurls[i],path);
        XrdFfsMisc_xrd_secsss_editurl(newurls[i], user_uid);
        res = (XrdFfsPosix_stat(newurls[i], &stbuf));
        if (res == 0)
        {
            if (S_ISREG(stbuf.st_mode))
                rval = XrdFfsPosix_truncate(newurls[i], size);
            else 
                rval = -1;
            if (rval == -1) 
            {
                syslog(LOG_WARNING, "WARNING: (f)truncate(%s) failed (errno = %d)", newurls[i], errno);
                break;
            }
/* again, it will be messy if a successful truncate is followed by a failed one */
        }
        else if (errno != ENOENT)
            rval = -1;
    }

    for (i = 0; i < nurls; i++)
        free(newurls[i]);

    if (rval != 0 && errno == 0) errno = EIO;
    return rval;
}

struct XrdFfsPosixX_readdirall_args {
    char *url;
    int *res;
    int *err;
    struct XrdFfsDentnames **dents;
};
 
/*
   It seems xrootd posix return dp[i] != NULL even if the dir
   doesn't exist on a data server. XrdFfsPosix_readdir() returns
   NULL in this case.

   Do we need some protection here? We are not in trouble so far
   because FUSE's _getattr will test the existance of the dir
   so we know that at least one data server has the directory.
 */
void* XrdFfsPosix_x_readdirall(void* x)
{
    struct XrdFfsPosixX_readdirall_args *args = (struct XrdFfsPosixX_readdirall_args*) x;
    DIR *dp;
    struct dirent *de;

/*
   Xrootd's Opendir will not return NULL even under some error. For instance,
   when it is supposed to return ENOENT or ENOTDIR, it actually returns 
   EINPROGRESS (115), and DIR *dp will not be NULL.
 */
    dp = XrdFfsPosix_opendir(args->url);
    if ( dp == NULL && errno != 0)
    {
        *(args->err) = errno;
        *(args->res) = -1;
        if (dp != NULL)
            XrdFfsPosix_closedir(dp);
    }
    else
    {
        *(args->res) = 0;
        while ((de = XrdFfsPosix_readdir(dp)) != NULL)
            XrdFfsDent_names_add(args->dents, de->d_name);
        XrdFfsPosix_closedir(dp);
    }
    return NULL;
}

int XrdFfsPosix_readdirall(const char *rdrurl, const char *path, char*** direntarray, uid_t user_uid)
{
    int i, j, n, nents, nurls; 
    bool hasDirLock = false;

    char *newurls[XrdFfs_MAX_NUM_NODES];
    int res_i[XrdFfs_MAX_NUM_NODES];
    int errno_i[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsDentnames *dir_i[XrdFfs_MAX_NUM_NODES] = {0};
    struct XrdFfsPosixX_readdirall_args args[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsQueueTasks *jobs[XrdFfs_MAX_NUM_NODES];

//    for (i = 0; i < XrdFfs_MAX_NUM_NODES; i++)
//        dir_i[i] = NULL;

    nurls = XrdFfsMisc_get_all_urls(rdrurl, newurls, XrdFfs_MAX_NUM_NODES);
    if (nurls < 0) 
    {
        errno = EACCES;
        return -1;
    }

    for (i = 0; i < nurls; i++)
    {
        errno_i[i] = 0;
        strcat(newurls[i], path);
        XrdFfsMisc_xrd_secsss_editurl(newurls[i], user_uid);
        args[i].url = newurls[i];
        args[i].err = &errno_i[i];
        args[i].res = &res_i[i];
        args[i].dents = &dir_i[i];
#ifdef NOUSE_QUEUE
        XrdFfsPosix_x_readdirall((void*) &args[i]);
    }   
#else
        jobs[i] = XrdFfsQueue_create_task(XrdFfsPosix_x_readdirall, (void**)(&args[i]), 0);
    }
    for (i = 0; i < nurls; i++)
    {
        XrdFfsQueue_wait_task(jobs[i]);
        XrdFfsQueue_free_task(jobs[i]);
    }
#endif

    errno = 0;
    for (i = 0; i < nurls; i++)
        if (res_i[i] != 0 && errno_i[i] == 125) // when host i is down
        {
            errno = ETIMEDOUT;
            syslog(LOG_WARNING, "WARNING: opendir(%s) failed (connection timeout)", newurls[i]);
            break;
        }

    for (i = 0; i < nurls; i++)
        free(newurls[i]);
    for (i = 1; i < nurls; i++)
        XrdFfsDent_names_join(&dir_i[i], &dir_i[i-1]);

    char *last = NULL, **dnarraytmp;

    n = XrdFfsDent_names_extract(&dir_i[nurls-1], &dnarraytmp);
    *direntarray = (char **) malloc(sizeof(char*) * n);

// note that dnarraytmp[] may contain redundant entries

    nents = 0;
    for (i = 0; i < n; i++)
    {
        // put DIR_LOCK to the last one to allow rm -rf to work...
        //
        if (! strcmp(dnarraytmp[i], "DIR_LOCK")) 
        {
            hasDirLock = true;
            continue;  
        }

        if (i != 0)   // can be used to filter out .lock .fail, etc. 
        {
            char *tmp, *tmp_dot;
            tmp = strdup(dnarraytmp[i]);
            tmp_dot = tmp + strlen(tmp) - 5;

            if (! strcmp(tmp_dot, ".lock") || ! strcmp(tmp_dot, ".fail"))   // filter out .lock/.fail files
            {
                for (j = nents - 1; j >= 0; j--)
                {
                    tmp_dot[0] = '\0';
                    if (! strcmp(tmp, (*direntarray)[j]))
                    {
                        tmp_dot[0] = '.';
                        free(tmp);
                        break;
                    }
                }
                if (j >= 0) continue;  // found the file cooresponding to the .lock/.fail
            }
            free(tmp);
        }

        if (last == NULL || strcmp(last, dnarraytmp[i]) != 0)
        {
            last = dnarraytmp[i];
            (*direntarray)[nents++] = strdup(dnarraytmp[i]);
        }
    }
    
    for (i = 0; i < n; i++) free(dnarraytmp[i]);  // do not mergo with the above because the above loop has 'break'.
    free(dnarraytmp);

/* inject this list into dent cache */

    char *p;
    p = strdup(path);
    XrdFfsDent_cache_fill(p, direntarray, nents);
    free(p);

    if (hasDirLock) (*direntarray)[nents++] = strdup("DIR_LOCK");

    return nents;
}

/* 
   struct XrdFfsPosixX_statvfsall_args, void XrdFfsPosix_x_statvfsall(), int XrdFfsPosiXrdFfsPosix_x_statvfsall() are 
   organized in such a way to allow using pthread if needed
 */

struct XrdFfsPosixX_statvfsall_args {
    char *url;
    int *res;
    int *err;
    struct statvfs *stbuf;
    short osscgroup;
};

void* XrdFfsPosix_x_statvfsall(void *x)
{
    struct XrdFfsPosixX_statvfsall_args *args = (struct XrdFfsPosixX_statvfsall_args *)x;
    char xattr[256];
    off_t oss_size;
    long long llVal;

    *(args->res) = XrdFfsPosix_getxattr(args->url, "xroot.space.oss.space", xattr, 256);
    *(args->err) = errno;
    sscanf((const char*)xattr, "%lld", &llVal);
    oss_size = static_cast<off_t>(llVal);
    args->stbuf->f_blocks = (fsblkcnt_t) (oss_size / args->stbuf->f_bsize);
//    sscanf((const char*)xattr, "%lld", &(args->stbuf->f_blocks));
    if (*(args->res) == -1)
    {
        args->stbuf->f_blocks = 0;
        args->stbuf->f_bavail = 0;
        args->stbuf->f_bfree = 0;
        return NULL;
    }
    *(args->res) = XrdFfsPosix_getxattr(args->url, "xroot.space.oss.free", xattr, 256);
    *(args->err) = errno;
    sscanf((const char*)xattr, "%lld", &llVal);
    oss_size = static_cast<off_t>(llVal);
    args->stbuf->f_bavail = (fsblkcnt_t) (oss_size / args->stbuf->f_bsize);
//    sscanf((const char*)xattr, "%lld", &(args->stbuf->f_bavail));
    if (*(args->res) == -1)
    {
        args->stbuf->f_blocks = 0;
        args->stbuf->f_bavail = 0;
        args->stbuf->f_bfree = 0;
        return NULL;
    }

/*
   The relation of the output of df and stbuf->f_blocks, f_bfree and f_bavail is
   Filesystem            Size        Used                  Avail            Use% Mounted on
                         f_blocks    f_blocks - f_bfree    f_bavail

   In the case of querying without oss.cgroup, f_bfree = f_bavail (e.g. Used is used space by all oss.space)
   In the case of querying with oss.cgroup, Used is used space by the specified oss.space (oss_size/f_bsize) and
   therefore f_bfree = f_blocks - oss_size / f_bsize (e.g. Used is oss_size / f_bsize)
 */

    if (args->osscgroup != 1)
        args->stbuf->f_bfree = args->stbuf->f_bavail;
    else
    {
        *(args->res) = XrdFfsPosix_getxattr(args->url, "xroot.space.oss.used", xattr, 256);
        *(args->err) = errno;
        sscanf((const char*)xattr, "%lld", &llVal);
        oss_size = static_cast<off_t>(llVal);
        args->stbuf->f_bfree = args->stbuf->f_blocks - (fsblkcnt_t) (oss_size / args->stbuf->f_bsize);
//        args->stbuf->f_bfree = args->stbuf->f_blocks - oss_size;
    }
    return NULL;
}

int XrdFfsPosix_statvfsall(const char *rdrurl, const char *path, struct statvfs *stbuf, uid_t user_uid)
{
    int i, nurls;
    short osscgroup;

    char *newurls[XrdFfs_MAX_NUM_NODES];
    int res_i[XrdFfs_MAX_NUM_NODES];
    int errno_i[XrdFfs_MAX_NUM_NODES];
    struct statvfs stbuf_i[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsPosixX_statvfsall_args args[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsQueueTasks *jobs[XrdFfs_MAX_NUM_NODES];

    nurls = XrdFfsMisc_get_all_urls(rdrurl, newurls, XrdFfs_MAX_NUM_NODES);
    if (nurls < 0)
    {
        errno = EACCES;
        return -1;
    }

    if (strstr(path, "oss.cgroup") != NULL)
        osscgroup = 1;
    else 
        osscgroup = 0;
    for (i = 0; i < nurls; i++)
    {
        strcat(newurls[i], path);
//        XrdFfsMisc_xrd_secsss_editurl(newurls[i], user_uid);
        args[i].url = newurls[i];
        args[i].res = &res_i[i];
        args[i].err = &errno_i[i];
        stbuf_i[i].f_bsize = stbuf->f_bsize;
        args[i].stbuf = &(stbuf_i[i]);
        args[i].osscgroup = osscgroup;
#ifdef NOUSE_QUEUE
        XrdFfsPosix_x_statvfsall((void*) &args[i]);
    }
#else
        jobs[i] = XrdFfsQueue_create_task(XrdFfsPosix_x_statvfsall, (void**)(&args[i]), 0);
    }
    for (i = 0; i < nurls; i++)
    {
        XrdFfsQueue_wait_task(jobs[i]);
        XrdFfsQueue_free_task(jobs[i]);
    }
#endif
 /*
   for statfs call, we don't care about return code and errno 
  */
    stbuf->f_blocks = 0;
    stbuf->f_bfree = 0;
    stbuf->f_bavail = 0;
    for (i = 0; i < nurls; i++)
    {
        stbuf->f_blocks += args[i].stbuf->f_blocks;
        stbuf->f_bavail += args[i].stbuf->f_bavail;
        stbuf->f_bfree += args[i].stbuf->f_bfree;
    }

    for (i = 0; i < nurls; i++)
        free(newurls[i]);

    return 0;
} 

/* XrdFfsPosiXrdFfsPosix_x_statall() */

struct XrdFfsPosixX_statall_args {
    char *url;
    int *res;
    int *err;
    struct stat *stbuf;
};

void* XrdFfsPosix_x_statall(void *x)
{
    struct XrdFfsPosixX_statall_args *args = (struct XrdFfsPosixX_statall_args *)x;

    *(args->res) = XrdFfsPosix_stat(args->url, args->stbuf);
    *(args->err) = errno;
    return (void *)0;
}

int XrdFfsPosix_statall(const char *rdrurl, const char *path, struct stat *stbuf, uid_t user_uid)
{
    int i, res, nurls;

    char *newurls[XrdFfs_MAX_NUM_NODES];
    int res_i[XrdFfs_MAX_NUM_NODES];
    int errno_i[XrdFfs_MAX_NUM_NODES];
    struct stat stbuf_i[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsPosixX_statall_args args[XrdFfs_MAX_NUM_NODES];
    struct XrdFfsQueueTasks *jobs[XrdFfs_MAX_NUM_NODES];

    char *p1, *p2, *dir, *file, rootpath[1024];

    rootpath[0] = '\0';
    strcat(rootpath,rdrurl);
    strcat(rootpath,path);
    p1 = strdup(path);
    p2 = strdup(path);
    dir = dirname(p1);
    file = basename(p2);

    if (XrdFfsDent_cache_search(dir, file))
    {
         XrdFfsMisc_xrd_secsss_editurl(rootpath, user_uid);
         res = XrdFfsPosix_stat(rootpath, stbuf);
// maybe a data server is down since the last _readdir()? in that case, continue 
// we also saw a case where redirectors report the file exist but meta redirector report
// that the file doesn't exist, and we need to continue at here
         if (res == 0) 
         {
             free(p1);
             free(p2);
             return 0;
         }
    }
    free(p1);
    free(p2);    

    nurls = XrdFfsMisc_get_all_urls(rdrurl, newurls, XrdFfs_MAX_NUM_NODES);

    for (i = 0; i < nurls; i++)
    {
        strcat(newurls[i], path);
        XrdFfsMisc_xrd_secsss_editurl(newurls[i], user_uid);
        args[i].url = newurls[i];
        args[i].res = &res_i[i];
        args[i].err = &errno_i[i];
        args[i].stbuf = &(stbuf_i[i]);
#ifdef NOUSE_QUEUE
        XrdFfsPosix_x_statall((void*) &args[i]);
    }
#else
        jobs[i] = XrdFfsQueue_create_task(XrdFfsPosix_x_statall, (void**)(&args[i]), 0);
    }
    for (i = 0; i < nurls; i++)
    {
        XrdFfsQueue_wait_task(jobs[i]);
        XrdFfsQueue_free_task(jobs[i]);
    }
#endif
    res = -1;
    errno = ENOENT;
    for (i = 0; i < nurls; i++)
        if (res_i[i] == 0) 
        {
            res = 0;
            errno = 0;
            memcpy((void*)stbuf, (void*)(&stbuf_i[i]), sizeof(struct stat));
            break;
        }
        else if (res_i[i] != 0 && errno_i[i] == 125) // when host i is down
        {
            res = -1;
            errno = ETIMEDOUT;
            syslog(LOG_WARNING, "WARNING: stat(%s) failed (connection timeout)", newurls[i]);
        }

    for (i = 0; i < nurls; i++)
        free(newurls[i]);

    return res;
}

#ifdef __cplusplus
  }
#endif
