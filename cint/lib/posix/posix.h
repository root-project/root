/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file lib/posix/posix.h
 ************************************************************************
 * Description:
 *  Create POSIX related function interface
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for non-commercial purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

/* Please read README file in this directory */

#ifndef G__POSIX_H
#define G__POSIX_H

#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#ifdef __MAKECINT__
/********************************************************************
 * types necessary for unistd.h
 ********************************************************************/
typedef mode_t umode_t;
typedef struct __dirstream DIR;

#define NAME_MAX 128
struct dirent {
  long d_ino;                /* inode number */
  /* off_t d_off; */         /* offset to this dirent */
  unsigned short d_reclen;   /* length of record */
  /* char d_namelen; */      /* length of d_name */
  char d_name[NAME_MAX+1];   /* file name */
};

struct stat {
  dev_t         st_dev;      /* device */
  ino_t         st_ino;      /* inode */
  umode_t       st_mode;     /* protection */
  nlink_t       st_nlink;    /* number of hard links */
  uid_t         st_uid;      /* user ID of owner */
  gid_t         st_gid;      /* group ID of owner */
  dev_t         st_rdev;     /* device type (if inode device) */
  off_t         st_size;     /* total size, in bytes */
  unsigned long st_blksize;  /* blocksize for filesystem I/O */
  unsigned long st_blocks;   /* number of blocks allocated */
  time_t        st_atime;    /* time of last access */
  time_t        st_mtime;    /* time of last modification */
  time_t        st_ctime;    /* time of last change */
};

#define SYS_NMLN 65
struct utsname {
  char sysname[SYS_NMLN];
  char nodename[SYS_NMLN];
  char release[SYS_NMLN];
  char version[SYS_NMLN];
  char machine[SYS_NMLN];
  /* char domainname[SYS_NMLN]; */
};

#else /* __MAKECINT__ */

#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sys/stat.h>
#include <dirent.h>
#include <sys/utsname.h>
#include <sys/types.h>

#endif /* __MAKECINT__ */

#ifndef __hpux
/********************************************************************
 * fcntl.h
 ********************************************************************/
#ifdef __MAKECINT__
extern int open(char *pathname,int flags);
extern int fcntl(int fd,int cmd,long arg);
extern int umask(int mask);
extern DIR* opendir(char *name);
extern int telldir(DIR* dir);
#endif /* __MAKECINT__ */

extern struct dirent *readdir(DIR *dir);
extern void seekdir(DIR* dir,long loc);
#if !defined(G__SUN) && !defined(__sun)
extern void rewinddir(DIR *dir);
#endif
extern int closedir(DIR *dirp);

/********************************************************************
 * sys/stat.h , unistd.h
 ********************************************************************/
extern int stat(const char *filename,struct stat *buf);
#ifdef __MAKECINT__
/* int S_ISLNK(mode_t m); */
int S_ISREG(mode_t m);
int S_ISDIR(mode_t m);
int S_ISCHR(mode_t m);
int S_ISBLK(mode_t m);
int S_ISFIFO(mode_t m);
/* int S_ISSOCK(mode_t m); */
/*
#define S_IFMT     00170000
#define S_IFSOCK    0140000
#define S_IFDIR     0040000
*/
#endif


/********************************************************************
 * To be supported ??
 ********************************************************************/
/* extern int mknod(const char *pathname,mode_t mode,dev_t dev); */
/* mount */
/* socket */
/* off_t lseek(int fildes,off_t offset,int whence); */
/* select */
/* readlink */
/* ioctl */
/* fread */

/********************************************************************
 * sys/utsname.h
 ********************************************************************/
int uname(struct utsname *buf);

/********************************************************************
 * unistd.h
 ********************************************************************/
extern int close(int fd);
extern ssize_t read(int fd,void *buf, size_t nbytes);
extern ssize_t write(int fd,const void *buf, size_t n);

extern int dup(int oldfd);
extern int dup2(int oldfd,int newfd);

extern int pipe(int filedes[2]);
extern unsigned int alarm(unsigned int seconds);
extern unsigned int sleep(unsigned int seconds);
#if defined(G__LINUX)
extern void usleep(unsigned long usec); /* BSD */
#endif
extern int pause(void);

extern int chown(const char *path,uid_t owner,gid_t group);
extern int chdir(const char *path);

extern char *getcwd(char *buf,size_t size);

extern long int sysconf(int name);

#if defined(G__GLIBC) && defined(G__GLIBC_MINOR)
#define G__GLIBC_ (G__GLIBC*100+G__GLIBC_MINOR)
#elif defined(__GLIBC__) && defined(__GLIBC_MINOR__)
#define G__GLIBC_ (__GLIBC__*100+__GLIBC_MINOR__)
#endif

#if defined(__SUNPRO_C) || defined(G__SUNPRO_C) 
extern int putenv(char *string);
#elif defined(G__GLIBC_) && (G__GLIBC_<201)
extern int putenv(char *string);
#else
extern int putenv(const char *string);
#endif

extern pid_t getpid(void);
extern pid_t getppid(void);
extern int setpgid(pid_t pid,pid_t pgid);
extern pid_t getpgrp(void);

extern uid_t getuid(void);
extern uid_t geteuid(void);
extern gid_t getgid(void);
extern gid_t getegid(void);
extern int setuid(uid_t uid);

extern char *cuserid(char *string);
extern char *getlogin(void);
extern char *ctermid(char *s);
extern char *ttyname(int desc);

extern int link(const char *oldpath,const char *newpath);
extern int unlink(const char *pathname);
extern int rmdir(const char *path);
extern int mkdir(const char *pathname,mode_t mode);
extern pid_t fork(void);

extern time_t time(time_t *t);

#ifndef G__HPUX
#ifdef __MAKECINT__
int S_ISLNK(mode_t m);
int S_ISSOCK(mode_t m); 
#endif
extern int fchown(int fd,uid_t owner,gid_t group);
extern int fchdir(int fd);
#ifndef G__SUN
extern char *get_current_dir_name(void);
extern pid_t getpgid(pid_t pid);
#endif
extern char *getwd(char *buf);
#if !defined(G__SUN) && !defined(__sun)
extern int setpgrp(void);
#else
extern long setpgrp(void);
#endif
extern int symlink(const char *oldpath,const char *newpath);
extern pid_t vfork(void);
#endif /* G__HPUX */


#endif /* __hpux */

#endif /* G__POSIX_H */
