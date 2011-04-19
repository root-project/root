// @(#)root/rpdutils:$Id$
// Author: Gerardo Ganis, March 2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_rpdconn
#define ROOT_rpdconn

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rpdconn                                                              //
//                                                                      //
// This header file contains the definition of some utility classes     //
// used for process communication between xproofd, rootd, proofexecv.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <pthread.h>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>

//
// Basic mutex helper class
//
 
class rpdmtxhelper {
private:
   pthread_mutex_t *mtx;
   bool             ok;

   void rawlock(const pthread_mutex_t *m) { if (m) {
                                         if (!pthread_mutex_lock((pthread_mutex_t *)m)) {
                                            mtx = (pthread_mutex_t *)m; ok = 1; } } }
   void rawunlock(const pthread_mutex_t *m) { if (m) { pthread_mutex_unlock((pthread_mutex_t *)m); } mtx = 0; ok = 0; }
   
public:
   rpdmtxhelper(const pthread_mutex_t *m = 0) { mtx = 0; ok = 0; rawlock(m); }
   rpdmtxhelper(const pthread_mutex_t &m) { mtx = 0; ok = 0; rawlock(&m); }
   virtual ~rpdmtxhelper() { rawunlock(mtx); }

   inline void lock() { rawlock(mtx); };
   inline void unlock() { rawunlock(mtx); }
   inline bool isok() { return ok; }
};

//
// Class describing a basic message
//
class rpdconn;
class rpdmsg {
friend class rpdconn;
protected:
   int          type;   // Message type (positive; -1 for undef)
   std::string  buf;    // Internal buffer
   int          cur;    // Offset while streaming out

public:
   rpdmsg(int t = -1) : type(t), cur(0) { }
   rpdmsg(int t, const char *b) : type(t), buf(b), cur(0) { }

   bool empty() { return ((cur < 0 ||
                          (cur >= 0 && (int) buf.length() <= cur)) ? 1 : 0); }
   void reset(int t = -1) { if (t >= 0) type = t; buf = ""; cur = 0; }
   void rewind() { cur = 0; }
   void settype(int t) { type = t; }

   // Access content
   int what() const { return type; }
   const char *content() const { return buf.c_str(); }

   // in/out functions
   void r_int(int &i);
   void r_double(double &d);
   void r_string(std::string &s);
   void w_int(int i);
   void w_double(double d);
   void w_string(const std::string &s);
};
// rpdmsg external operators
inline rpdmsg &operator>>(rpdmsg &m, int &i) { m.r_int(i); return m; }
inline rpdmsg &operator>>(rpdmsg &m, double &d) { m.r_double(d); return m; }
inline rpdmsg &operator>>(rpdmsg &m, std::string &s)  { m.r_string(s); return m; }
inline rpdmsg &operator<<(rpdmsg &m, int i) { m.w_int(i); return m; }
inline rpdmsg &operator<<(rpdmsg &m, double d) { m.w_double(d); return m; }
inline rpdmsg &operator<<(rpdmsg &m, const std::string &s) { m.w_string(s); return m; }

//
// Class describing a basic connection
//
class rpdconn {
protected:
   pthread_mutex_t rdmtx;   // Mutex for read operations
   pthread_mutex_t wrmtx;   // Mutex for write operations
   int             rdfd;    // Descriptor for read operations
   int             wrfd;    // Descriptor for write operations
public:
   rpdconn(int r = -1, int w = -1);
   virtual ~rpdconn() { }

   virtual void close() = 0;
   virtual bool isvalid(bool rd) const {
      int rc = 0;
      if (rd) {
         rpdmtxhelper mh(&rdmtx);
         rc = (rdfd > 0) ? 1 : 0;
      } else {
         rpdmtxhelper mh(&wrmtx);
         rc = (wrfd > 0) ? 1 : 0;
      }
      return rc;
   }
   void setdescriptors(int r = -1, int w = -1) {
      { rpdmtxhelper mh(&rdmtx); rdfd = r; }
      { rpdmtxhelper mh(&wrmtx); wrfd = w; }
   }

   int pollrd(int to = -1);

   int send(int i);
   int send(int type, const char *msg);
   int send(const rpdmsg &msg);
   int recv(int &i);
   int recv(int &type, std::string &msg);
   int recv(rpdmsg &msg);
   
   int senddesc(int desc);
   int recvdesc(int &desc);
};

//
// Class describing a TCP connection
//
class rpdtcp : public rpdconn {
private:
   std::string     host;    // Host name
   int             port;    // Port
protected:
   int             fd;      // Socket descriptor
   struct sockaddr addr;    // Structure describing the peer address
public:
   rpdtcp(int d = -1) : rpdconn(), port(0), fd(d) { setdescriptors(d,d); } // Used by rpdtcpsrv
   rpdtcp(const char *h, int p);
   virtual ~rpdtcp() { close(); }

   void close() { if (fd > 0) ::close(fd); fd = -1; setdescriptors(); }
   int  exportfd() { int d = fd; fd = -1; setdescriptors(); return d; }
   const struct sockaddr *address() const { return (const struct sockaddr *)&addr; }
};

//
// Class describing a server TCP connection
//
class rpdtcpsrv : public rpdtcp {
public:
   rpdtcpsrv(int p, int backlog = 10);
   virtual ~rpdtcpsrv() { rpdtcp::close(); }

   rpdtcp *accept(int to = -1, int *err = 0);
};

//
// Class describing a UNIX connection
//
class rpdunix : public rpdtcp {
private:
   std::string     sockpath;  // Socket path
public:
   rpdunix(int d = -1) : rpdtcp(d) { }
   rpdunix(const char *path);
   virtual ~rpdunix() { rpdtcp::close(); }
};

//
// Class describing a server UNIX connection
//
class rpdunixsrv : public rpdunix {
protected:
   int             fd;    // Socket descriptor
public:
   rpdunixsrv(const char *path, int backlog = 10);
   virtual ~rpdunixsrv() { rpdtcp::close(); }

   rpdunix *accept(int to = -1, int *err = 0);
};

#endif
