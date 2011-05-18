/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************
* cintsock.h
*  TCP/IP connection library 
****************************************************************/

#ifndef G__CINTSOCK_H
#define G__CINTSOCK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**********************************************************************
 * For compiler to read
 **********************************************************************/
#if defined(_WIN32) && !defined(__CINT__)

#include <windows.h>
#include <winsock.h>
#include <process.h>

#elif !defined(__CINT__)

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

typedef struct sockaddr SOCKADDR, *LPSOCKADDR;
typedef struct sockaddr_in SOCKADDR_IN, *LPSOCKADDR_IN;
typedef struct servent SERVENT, *LPSERVENT;
typedef int SOCKET;
typedef unsigned long IN_ADDR;
typedef struct WSAData { int dmy;} WSADATA, *LPWSADATA;
typedef int WORD;

/* Dummy */
int WSACleanup();
int WSAStartup(WORD wVersionRequested,LPWSADATA lpWSAData);
int MAKEWORD(int a,int b);
int closesocket(SOCKET s);


/**********************************************************************
 * For makecint to read
 **********************************************************************/
#else /* __CINT__ */

#ifdef G__WIN32

#include <windows.h>

typedef unsigned int SOCKET;

#define WSDESCRIPTION_LEN 128
#define WSASYS_STATUS_LEN 128
typedef struct WSAData {
  WORD wVersion;
  WORD wHighVersion;
  char szDescription[WSDESCRIPTION_LEN+1];
  char szSystemStatus[WSASYS_STATUS_LEN+1];
  unsigned short iMaxSockets;
  unsigned short iMaxUdpDg;
  char* lpVendorInfo;
} WSADATA, *LPWSADATA;

#else /* UNIX */

typedef int SOCKET;

typedef int WORD;
typedef struct WSAData { int dmy; } WSADATA, *LPWSADATA;

#endif


typedef unsigned int u_int;
typedef unsigned long u_long;
typedef unsigned short u_short;
typedef unsigned char u_char;

struct sockaddr {
  unsigned short sa_family;
  char sa_data[14];
};

struct in_addr {
  u_long s_addr;
#if 0
  union {
    struct { u_char s_b1,s_b2,s_b3,s_b4;} S_un_b;
    struct { u_short s_w1,s_w2;} S_un_w;
    u_long S_addr;
  } S_un;
#endif
};

struct sockaddr_in {
  short sin_family;
  unsigned short sin_port;
  struct in_addr sin_addr;
  char sin_zero[8];
};

struct hostent {
  char* h_name;
  char** h_aliases;
  short h_addrtype;
  short h_length;
  char** h_addr_list;
};
// #define h_addr h_addr_list[0];

struct servent {
};
typedef unsigned long IN_ADDR;

#if 0
struct iovec {
};

struct msghdr {
  caddr_t msg_name;
  struct iovec *msg_iov;
  u_int msg_iovlen;
  caddr_t msg_control;
  u_int msg_controllen;
  int msg_flags;
};

struct cmsghdr {
  u_int cmsg_len;
  int cmsg_level;
  int cmsg_type;
  /* u_char cmsg_data[]; */
};
#endif

#pragma link off class sockaddr;
#pragma link off class servent;
//#pragma link off class hostent;



/* typedefs */
typedef struct sockaddr SOCKADDR, *LPSOCKADDR;
typedef struct sockaddr_in SOCKADDR_IN, *LPSOCKADDR_IN;
typedef struct servent SERVENT, *LPSERVENT;

/***************************************************************
 * WIN32 only function
 ***************************************************************/
int WSACleanup();
int WSAStartup(WORD wVersionRequested,LPWSADATA lpWSAData);
int MAKEWORD(int a,int b);

/***************************************************************
 * common function
 ***************************************************************/
unsigned long htonl(unsigned long hostlong);
unsigned short htons(unsigned short hostshort);
unsigned long ntohl(unsigned long netlong);
unsigned short ntohs(unsigned short netshort);

/***************************************************************
 * TCP/IP function
 ***************************************************************/
SOCKET accept(SOCKET s,SOCKADDR* addr,int *addrlen);
int bind(SOCKET s,SOCKADDR* name,int namelen);
int closesocket(SOCKET s);
#ifdef G__SUN
int connect(SOCKET s,SOCKADDR* name,int namelen);
#else
int connect(SOCKET s,const SOCKADDR* name,int namelen);
#endif
struct hostent* gethostbyname(const char* hostname);
SERVENT* getservbyname(const char* name,const char* proto);
int getsockopt(SOCKET s,int level,int optname,char *optval,int *optlen);
IN_ADDR inet_addr(const char *iphost);
int listen(SOCKET s,int backlog);
int send(SOCKET s,const char* buf,int len,int flags);
int setsockopt(SOCKET s,int level,int optname,const char *optval,int optlen);
int shutdown(SOCKET s,int how);
int socket(int domain,int type,int protocol);
int recv(SOCKET s,char* buf,int len,int flags);

#endif /* __CINT__ */


#endif
