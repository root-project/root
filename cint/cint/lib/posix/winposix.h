/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
* winposix.h
*  POSIX emulation function on Windows
****************************************************************/

#ifndef G__POSIX_H
#define G__POSIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __CINT__
typedef unsigned long DWORD;
typedef void* HANDLE;
#else
#include <windows.h>
#endif

/****************************************************************
* struct definition
****************************************************************/
typedef unsigned long off_t;
typedef DWORD pid_t;
typedef DWORD uid_t;
typedef DWORD gid_t;
typedef DWORD dev_t;
typedef DWORD ino_t;
typedef DWORD umode_t;
typedef DWORD mode_t;
typedef DWORD nlink_t;

#define S_IRUSR 0x0400
#define S_IWUSR 0x0200
#define S_IXUSR 0x0100
#define S_IRGRP 0x0040
#define S_IWGRP 0x0020
#define S_IXGRP 0x0010
#define S_IROTH 0x0004
#define S_IWOTH 0x0002
#define S_IXOTH 0x0001

#define NAME_MAX 256

typedef struct dirent {
  long d_ino;                /* inode number */
  off_t d_off;               /* offset to this dirent */
  unsigned short d_reclen;   /* length of d_name */
  char d_name[NAME_MAX+1];   /* file name */
#ifndef __CINT__
  LPWIN32_FIND_DATA pwindir; /* Orignial extention */
#endif
} dirent;

typedef struct DIR {
  char dirname[NAME_MAX];
  HANDLE h;
#ifndef __CINT__
  WIN32_FIND_DATA windir;
#endif
  dirent posixdir;
} DIR;

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

/****************************************************************
* POSIX system call emulation
****************************************************************/

DIR* opendir(const char *name) ; 
dirent* readdir(DIR* dir);
void rewinddir(DIR *dir) ;
int closedir(DIR *dirp) ;

char *getcws(char *buf,size_t size);
int chdir(char *path);
int rmdir(char *path);
int mkdir(char *path,mode_t mode);

pid_t getpid(void);
uid_t getuid(void);
gid_t getgid(void);
char *getlogin(void);

int uname(struct utsname* buf);
int stat(char *filename,struct stat *buf);

#if defined(G__SYMANTEC) || defined(__SC__)
void sleep(long seconds);
#else
unsigned int sleep(unsigned int seconds);
#endif

int putenv(const char *string);


/****************************************************************
* Original extension
****************************************************************/

int isDirectory(struct dirent* pd) ;

#endif
