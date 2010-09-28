//         $Id$
//
// You received this file as part of MCA2
// Modular Controller Architecture Version 2
//
//Copyright (C) Forschungszentrum Informatik Karlsruhe
//
//This program is free software; you can redistribute it and/or
//modify it under the terms of the GNU General Public License
//as published by the Free Software Foundation; either version 2
//of the License, or (at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program; if not, write to the Free Software
//Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// this is a -*- C++ -*- file
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#ifndef __XrdWin32_h__
#define __XrdWin32_h__

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <time.h>
#include <direct.h>
#include <sys/types.h>
#include <Winsock2.h>

#ifndef POLLIN
#define POLLIN          0x0001    /* There is data to read */
#define POLLPRI         0x0002    /* There is urgent data to read */
#define POLLOUT         0x0004    /* Writing now will not block */
#define POLLERR         0x0008    /* Error condition */
#define POLLHUP         0x0010    /* Hung up */
#define POLLNVAL        0x0020    /* Invalid request: fd not open */
#define POLLRDNORM      0x0001
#define POLLWRNORM      0x0002
#define POLLRDBAND      0x0000

struct pollfd {
   unsigned int fd;
   short events;
   short revents;
};
#endif

#ifndef EMSGSIZE
#define EMSGSIZE        WSAEMSGSIZE 
#endif
#ifndef EAFNOSUPPORT
#define EAFNOSUPPORT    WSAEAFNOSUPPORT 
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK     WSAEWOULDBLOCK 
#endif
#ifndef ECONNRESET
#define ECONNRESET      WSAECONNRESET 
#endif
#ifndef EINPROGRESS
#define EINPROGRESS     WSAEINPROGRESS 
#endif
#ifndef ENOBUFS
#define ENOBUFS         WSAENOBUFS 
#endif
#ifndef EPROTONOSUPPORT
#define EPROTONOSUPPORT WSAEPROTONOSUPPORT 
#endif
#ifndef ECONNREFUSED
#define ECONNREFUSED    WSAECONNREFUSED 
#endif
#ifndef EBADFD
#define EBADFD          WSAENOTSOCK 
#endif
#ifndef EOPNOTSUPP
#define EOPNOTSUPP      WSAEOPNOTSUPP 
#endif
#ifndef ENETUNREACH
#define ENETUNREACH     WSAENETUNREACH
#endif
#ifndef EHOSTUNREACH
#define EHOSTUNREACH    WSAEHOSTUNREACH
#endif
#ifndef EHOSTDOWN
#define EHOSTDOWN       WSAEHOSTDOWN
#endif
#ifndef EISCONN
#define EISCONN         WSAEISCONN
#endif
#ifndef ECONNRESET
#define ECONNRESET      WSAECONNRESET
#endif
#ifndef ECONNABORTED
#define ECONNABORTED    WSAECONNABORTED
#endif
#ifndef ESHUTDOWN
#define ESHUTDOWN       WSAESHUTDOWN
#endif
#ifndef ETIMEDOUT
#define ETIMEDOUT       WSAETIMEDOUT 
#endif
#ifndef ETXTBSY
#define ETXTBSY         26 
#endif

#define WEXITSTATUS(w) (((w) >> 8) & 0xff) 
#define WIFEXITED(w)   (((w) & 0xff) == 0) 
#define WTERMSIG(w)     ((w) & 0x7f)
#define WIFSIGNALED(w) (((w) & 0x7f) > 0 && (((w) & 0x7f) < 0x7f)) 

#ifndef S_ISDIR
#define S_ISDIR(m) (((m)&(S_IFMT)) == (S_IFDIR))
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m)&(S_IFMT)) == (S_IFREG))
#endif
#ifndef S_IXUSR
#define S_IXUSR 00100
#endif
#ifndef S_IRGRP
#define S_IRGRP 00040
#endif
#ifndef S_IXGRP
#define S_IXGRP 00010
#endif
#ifndef S_IROTH
#define S_IROTH 00004
#endif
#ifndef S_IXOTH
#define S_IXOTH 00001
#endif
#ifndef S_IRUSR
#define S_IRUSR S_IREAD
#endif
#ifndef S_IWUSR
#define S_IWUSR S_IWRITE
#endif
#ifndef S_IWGRP
#define S_IWGRP 000020
#endif
#ifndef S_IWOTH
#define S_IWOTH 000002
#endif
#ifndef S_IRWXU
#define S_IRWXU 0000700
#endif
#ifndef S_IRWXG
#define S_IRWXG 0000070
#endif

#ifndef S_ISFIFO
# ifndef S_IFIFO
#  define S_IFIFO 0010000
# endif
# define S_ISFIFO(m)	((m & S_IFMT) == S_IFIFO)
#endif

#ifndef S_IFSOCK
#define S_IFSOCK   0140000
#endif

#define _SC_PAGESIZE 1

#define F_GETFL      1
#define F_SETFL      2
#define F_GETFD      4
#define F_SETFD      8

#define FD_CLOEXEC   1
#define O_NDELAY     2
#define O_NONBLOCK   4

#define X_OK 1
#define W_OK 2
#define R_OK 4

#define RTLD_NOW 0x0001

#ifndef STDIN_FILENO
#define STDIN_FILENO    0
#define STDOUT_FILENO   1
#define STDERR_FILENO   2
#endif

#ifndef fsync
#define fsync(a) _commit(a)
#endif

#ifndef socklen_t
#define socklen_t int
#endif

#ifndef SOCKLEN_t
#define SOCKLEN_t int
#endif

#ifndef snprintf
#define snprintf _snprintf
#endif

#ifndef caddr_t 
typedef char* caddr_t;
#endif

#ifndef pid_t 
typedef int pid_t;
#endif

#ifndef mode_t
typedef unsigned int mode_t;
#endif

#ifndef uint16_t
typedef unsigned short uint16_t;
#endif

struct timezone {
   int tz_minuteswest; /* minutes west of Greenwich */
   int tz_dsttime;     /* type of dst correction */
};

inline int poll(struct pollfd *fds, unsigned int nfds, int timeout)
{
   unsigned int max_fd = 0;
   unsigned int i;

   fd_set *open_fds, *read_fds, *write_fds, *except_fds;
   struct timeval tv = { timeout / 1000, (timeout % 1000) * 1000 };

   for (i = 0; i < nfds; ++i) {
      if (fds[i].fd > max_fd) {
         max_fd = fds[i].fd;
      }
   }

   size_t fds_size = (max_fd + 1) * sizeof (fd_set);

   open_fds = (fd_set *) malloc (fds_size);
   read_fds = (fd_set *) malloc (fds_size);
   write_fds = (fd_set *) malloc (fds_size);
   except_fds = (fd_set *) malloc (fds_size);

   if (!open_fds || !read_fds || !write_fds || !except_fds) {
      return -1;
   }

   FD_ZERO(open_fds) ;
   FD_ZERO(read_fds) ;
   FD_ZERO(write_fds) ;
   FD_ZERO(except_fds) ;

   for ( i = 0; i < nfds; ++i) {
      FD_SET (fds[i].fd, open_fds);
      if (fds[i].events & POLLIN)
         FD_SET (fds[i].fd, read_fds);
      if (fds[i].events & POLLOUT)
         FD_SET (fds[i].fd, write_fds);
      if (fds[i].events & POLLPRI)
         FD_SET (fds[i].fd, except_fds);
   }

//  Sleep(1);
   int ret = select(max_fd + 1, read_fds, write_fds, except_fds, timeout < 0 ? NULL : &tv);

   for (i = 0; i < nfds; ++i) {
      if (!FD_ISSET (fds[i].fd, open_fds))
         fds[i].revents = POLLNVAL;
      else if (ret < 0)
         fds[i].revents = POLLERR;
      else {
         fds[i].revents = 0;
         if (FD_ISSET (fds[i].fd, read_fds))
            fds[i].revents |= POLLIN;
         if (FD_ISSET (fds[i].fd, write_fds))
            fds[i].revents |= POLLOUT;
         if (FD_ISSET (fds[i].fd, except_fds))
            fds[i].revents |= POLLPRI;
      }
   }

   free(open_fds);
   free(read_fds);
   free(write_fds);
   free(except_fds);

   return ret;
}

struct iovec {
   u_long    iov_len;
   char FAR *iov_base;
};

inline int lrint(double n) { return (int)n; };

extern void gethostbyname_r(const char *inetName, struct hostent *hent, char *buff, 
                            int buffsize, struct hostent **hp, int *rc);
extern void gethostbyaddr_r(char *addr, size_t len, int type, struct hostent *hent, char *buff,
                            size_t buffsize, struct hostent **hp, int *rc);
extern int getservbyname_r(const char *servname, const char *servtype, struct servent *sent,
                           char *buff, size_t buffsize, struct servent **sp);
extern int gettimeofday(struct timeval * tp, struct timezone * tzp);
extern void *dlopen(const char *libPath, int opt);
extern BOOL dlclose(void *lib);
extern void *dlsym(void *libHandle, const char *pname);
extern char *dlerror();
extern pid_t fork();
extern const char *inet_ntop(int af, const void *src, char *dst, size_t size);
extern int sysconf(int what);
extern int fcntl(int fd, int cmd, long arg);
extern int close(int fd);
extern int writev(int sock, const struct iovec iov[], int nvecs);
extern int posix_memalign (void **memptr, size_t alignment, size_t size);
extern char *index(const char *str, int c);
extern char *cuserid(char * s);

#ifndef localtime_r
#define localtime_r( _clock, _result ) \
        ( *(_result) = *localtime( (_clock) ), \
          (_result) )
#endif

#define pipe(a) _pipe(a, 256, O_BINARY)
#define rindex strrchr
#define sleep(s) Sleep(s*1000)

#define strtoll(a, b, c) _strtoi64(a, b, c)
#define ntohll(x) (((_int64)(ntohl((int)((x << 32) >> 32))) << 32) | (unsigned int)ntohl(((int)(x >> 32))))
#define htonll(x) ntohll(x)
#define random() rand()

#define usleep(x) Sleep(x / 1000)
#define lstat(a, b) stat(a, b)
#define memalign(a, b) _aligned_malloc(b, a)
struct sockaddr_un { unsigned short sun_family; char sun_path[128]; };
#define setpgid(x,y)
#define fsync(a) _commit(a)
#define ssize_t SSIZE_T

#endif // __XrdWin32_h__

