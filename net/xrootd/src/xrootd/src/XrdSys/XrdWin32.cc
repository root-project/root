//          $Id$

//const char *XrdWin32CVSID = "$Id$";

#include "XrdSys/XrdWin32.hh"
#include <Windows.h>
#include <errno.h>
#include <malloc.h>

int sysconf(int what)
{
   SYSTEM_INFO info;
   GetSystemInfo(&info);
   return (int)(info.dwPageSize);
}

#if 0 // defined(_MSC_VER) && (_MSC_VER < 1400)
//
// Though working fine with MSVC++7.1, this definition gives problems
// with MSVC++9.0; we comment it out and wait for better times
int fcntl(int fd, int cmd, long arg)
{
   u_long argp = 1;
   if ((cmd == F_SETFL) && (arg & O_NONBLOCK))
      return ioctlsocket((SOCKET)fd, FIONBIO, &argp);
   return 0;
}
#else
int fcntl(int, int, long)
{
   // Dummy version
   return 0;
}
#endif

void gethostbyname_r(const char *inetName, struct hostent *hent, char *buff,
                     int buffsize, struct hostent **hp, int *rc)
{
   struct hostent *hentry;
   hentry = gethostbyname(inetName);
   if (hentry == 0) {
      int err = WSAGetLastError();
      switch (err) {
         case WSAHOST_NOT_FOUND:
            *rc = 1; //HOST_NOT_FOUND
            break;
         case WSATRY_AGAIN:
            *rc = 2; //TRY_AGAIN
            break;
         case WSANO_RECOVERY:
            *rc = 3; //NO_RECOVERY
            break;
         case WSANO_DATA:
            *rc = 4; //NO_DATA;
            break;
         default:
            *rc = 1;
            break;
      }
      return;
   }
   hent->h_addr_list = hentry->h_addr_list;
   hent->h_addrtype  = hentry->h_addrtype;
   hent->h_aliases   = hentry->h_aliases;
   hent->h_length    = hentry->h_length;
   hent->h_name      = hentry->h_name;
}

void gethostbyaddr_r(char *addr, size_t len, int type, struct hostent *hent, char *buff,
                     size_t buffsize, struct hostent **hp, int *rc)
{
   struct hostent *hentry;
   hentry = gethostbyaddr(addr, (int)len, type);
   if (hentry == 0) {
      int err = WSAGetLastError();
      switch (err) {
         case WSAHOST_NOT_FOUND:
            *rc = 1; //HOST_NOT_FOUND
            break;
         case WSATRY_AGAIN:
            *rc = 2; //TRY_AGAIN
            break;
         case WSANO_RECOVERY:
            *rc = 3; //NO_RECOVERY
            break;
         case WSANO_DATA:
            *rc = 4; //NO_DATA;
            break;
         default:
            *rc = 1;
            break;
      }
      return;
   }
   hent->h_addr_list = hentry->h_addr_list;
   hent->h_addrtype  = hentry->h_addrtype;
   hent->h_aliases   = hentry->h_aliases;
   hent->h_length    = hentry->h_length;
   hent->h_name      = hentry->h_name;
}

int getservbyname_r(const char *servname, const char *servtype, struct servent *sent,
                    char *buff, size_t buffsize, struct servent **sp)
{
   sent = getservbyname(servname, servtype);
   if (sent != NULL)
      return 0;
   return -1;
}

/* FILETIME of Jan 1 1970 00:00:00. */
static const unsigned __int64 epoch = 116444736000000000L;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
   FILETIME       file_time;
   SYSTEMTIME     system_time;
   ULARGE_INTEGER ularge;

   GetSystemTime(&system_time);
   SystemTimeToFileTime(&system_time, &file_time);
   ularge.LowPart = file_time.dwLowDateTime;
   ularge.HighPart = file_time.dwHighDateTime;

   tp->tv_sec = (long) ((ularge.QuadPart - epoch) / 10000000L);
   tp->tv_usec = (long) (system_time.wMilliseconds * 1000);

   return 0;
}

void *dlopen(const char *libPath, int opt)
{
   return (void *)LoadLibrary(libPath);
}

BOOL dlclose(void *lib)
{
   return FreeLibrary((HMODULE)lib);
}

void *dlsym(void *libHandle, const char *pname)
{
   return (void *)GetProcAddress((HMODULE)libHandle, (LPCSTR)pname);
}

char *dlerror()
{
   LPVOID lpMsgBuf;
   FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, GetLastError(),
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &lpMsgBuf, 0, NULL );
   return (char *)lpMsgBuf;
}

pid_t fork()
{
   char *cmdline;
   STARTUPINFO startInfo;
   PROCESS_INFORMATION procInfo;

   cmdline = GetCommandLine();

   ZeroMemory(&startInfo, sizeof(startInfo));
   startInfo.cb = sizeof(startInfo);
   BOOL retval = CreateProcess(NULL, cmdline, NULL, NULL, TRUE,
                               DETACHED_PROCESS, NULL, NULL,
                               &startInfo, &procInfo);
   delete [] cmdline;
   return (pid_t)procInfo.hProcess;
}

LARGE_INTEGER LargeIntegerSubtract(LARGE_INTEGER a, LARGE_INTEGER b)
{
   LARGE_INTEGER ret;
   ret.QuadPart = a.QuadPart - b.QuadPart;
   return ret;
}

/*
 * Copyright (c) 1999 Kungliga Tekniska Hgskolan
 * (Royal Institute of Technology, Stockholm, Sweden).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Kungliga Tekniska
 *      Hgskolan and its contributors.
 *
 * 4. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#define INET_ADDRSTRLEN 16
#define EAFNOSUPPORT WSAEAFNOSUPPORT

#ifndef IN6ADDRSZ
#define IN6ADDRSZ   16   /* IPv6 T_AAAA */
#endif

#ifndef INT16SZ
#define INT16SZ     2    /* word size */
#endif

static const char *
inet_ntop_v4 (const void *src, char *dst, size_t size)
{
   const char digits[] = "0123456789";
   int i;
   struct in_addr *addr = (struct in_addr *)src;
   u_long a = ntohl(addr->s_addr);
   const char *orig_dst = dst;

   if (size < INET_ADDRSTRLEN) {
      errno = ENOSPC;
      return NULL;
   }
   for (i = 0; i < 4; ++i) {
      int n = (a >> (24 - i * 8)) & 0xFF;
      int non_zerop = 0;

      if (non_zerop || n / 100 > 0) {
         *dst++ = digits[n / 100];
         n %= 100;
         non_zerop = 1;
      }
      if (non_zerop || n / 10 > 0) {
         *dst++ = digits[n / 10];
         n %= 10;
         non_zerop = 1;
      }
      *dst++ = digits[n];
      if (i != 3)
         *dst++ = '.';
   }
   *dst++ = '\0';
   return orig_dst;
}

#ifdef INET6
/*
 * Convert IPv6 binary address into presentation (printable) format.
 */
static const char *
inet_ntop_v6 (const u_char *src, char *dst, size_t size)
{
   /*
   * Note that int32_t and int16_t need only be "at least" large enough
   * to contain a value of the specified size.  On some systems, like
   * Crays, there is no such thing as an integer variable with 16 bits.
   * Keep this in mind if you think this function should have been coded
   * to use pointer overlays.  All the world's not a VAX.
   */
   char  tmp [INET6_ADDRSTRLEN+1];
   char *tp;
   struct {
      long base;
      long len;
   } best, cur;
   u_long words [IN6ADDRSZ / INT16SZ];
   int    i;

   /* Preprocess:
   *  Copy the input (bytewise) array into a wordwise array.
   *  Find the longest run of 0x00's in src[] for :: shorthanding.
   */
   memset (words, 0, sizeof(words));
   for (i = 0; i < IN6ADDRSZ; i++)
      words[i/2] |= (src[i] << ((1 - (i % 2)) << 3));

   best.base = -1;
   cur.base  = -1;
   for (i = 0; i < (IN6ADDRSZ / INT16SZ); i++) {
      if (words[i] == 0) {
         if (cur.base == -1)
            cur.base = i, cur.len = 1;
         else cur.len++;
      }
      else if (cur.base != -1) {
         if (best.base == -1 || cur.len > best.len)
            best = cur;
         cur.base = -1;
      }
   }
   if ((cur.base != -1) && (best.base == -1 || cur.len > best.len))
      best = cur;
   if (best.base != -1 && best.len < 2)
      best.base = -1;

   /* Format the result.
   */
   tp = tmp;
   for (i = 0; i < (IN6ADDRSZ / INT16SZ); i++) {
      /* Are we inside the best run of 0x00's?
       */
      if (best.base != -1 && i >= best.base && i < (best.base + best.len)) {
         if (i == best.base)
            *tp++ = ':';
         continue;
      }

      /* Are we following an initial run of 0x00s or any real hex?
       */
      if (i != 0)
         *tp++ = ':';

      /* Is this address an encapsulated IPv4?
       */
      if (i == 6 && best.base == 0 && (best.len == 6 ||
         (best.len == 5 && words[5] == 0xffff))) {
         if (!inet_ntop_v4(src+12, tp, sizeof(tmp) - (tp - tmp))) {
            errno = ENOSPC;
            return (NULL);
         }
         tp += strlen(tp);
         break;
      }
      tp += sprintf (tp, "%lX", words[i]);
   }

   /* Was it a trailing run of 0x00's?
    */
   if (best.base != -1 && (best.base + best.len) == (IN6ADDRSZ / INT16SZ))
      *tp++ = ':';
   *tp++ = '\0';

   /* Check for overflow, copy, and we're done.
    */
   if ((size_t)(tp - tmp) > size) {
      errno = ENOSPC;
      return (NULL);
   }
   return strcpy (dst, tmp);
   return (NULL);
}
#endif   /* INET6 */

const char *inet_ntop(int af, const void *src, char *dst, size_t size)
{
   switch (af) {
      case AF_INET :
         return inet_ntop_v4 (src, dst, size);
#ifdef INET6
      case AF_INET6:
         return inet_ntop_v6 ((const u_char*)src, dst, size);
#endif
      default :
         errno = EAFNOSUPPORT;
         return NULL;
   }
}

static void myerrcode(int err)
{
   if (err == EIO)
      err = EBADF; // sometimes we get EIO when Unix would be EBADF
   if (err == WSAENOTSOCK)
      err = EBADF; // if it's not a socket, it's also not a fd
   SetLastError(err);
   errno = err;
}

// This is the simple and dirty method to check if fd is a socket
#define IS_SOCKET(fd) ((fd)>=64)

// And this is a more clever one
static bool is_socket(SOCKET fd)
{
   char sockbuf[80];
   int optlen;
   int retval;

   optlen = sizeof(sockbuf);
   retval = getsockopt(fd, SOL_SOCKET, SO_TYPE, sockbuf, &optlen);
   if (retval == SOCKET_ERROR) {
      int iRet;
      iRet = WSAGetLastError();
      if (iRet == WSAENOTSOCK || iRet == WSANOTINITIALISED)
         return FALSE;
   }
   //
   // If we get here, then fd is actually a socket.
   //
   return TRUE;
}

int close(int fd)
{
   int ret;
   if (is_socket(fd)) {
      ret = closesocket(fd);
      myerrcode(GetLastError());
   }
   else {
      ret = _close(fd);
      myerrcode(errno);
   }
   return ret;
}

int writev(int fd, const struct iovec iov[], int nvecs)
{
   DWORD ret;
   char *buffer, *bp;
   size_t i, bytes = 0;
   if (is_socket(fd)) {
      if (WSASend(fd, (LPWSABUF)iov, nvecs, &ret, 0, NULL, NULL) == 0) {
         return ret;
      }
   }
   else {

      /* Find the total number of bytes to write */
      for (i = 0; i < (size_t)nvecs; i++)
         bytes += iov[i].iov_len;

      if (bytes == 0)   /* not an error */
         return (0);

      /* Allocate a temporary buffer to hold the data */
      buffer = bp = (char*) alloca (bytes);
      if (!buffer) {
         errno = ENOMEM;
         return (-1);
      }

      /* Copy the data into buffer. */
      for (i = 0; i < (size_t)nvecs; ++i) {
         memcpy (bp, iov[i].iov_base, iov[i].iov_len);
         bp += iov[i].iov_len;
      }
      return (int)_write(fd, buffer, bytes);
   }
   return -1;
}

char *index(const char *str, int c)
{
   return strchr((char *)str, c);
}

char *getlogin()
{
   static char user_name[256];
   DWORD  length = sizeof(user_name);
   if (GetUserName(user_name, &length))
      return user_name;
   return NULL;
}

char *cuserid(char * s)
{
   char * name = getlogin();
   if (s)
      return strcpy(s, name ? name : "");
   return name;
}

int posix_memalign(void **memptr, size_t alignment, size_t size)
{
   void *mem = malloc(size);
   if (mem != NULL) {
      *memptr = mem;
      return 0;
   }
   return ENOMEM;
}

