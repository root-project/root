#ifndef __OUC_ERRINFO_H__
#define __OUC_ERRINFO_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d O u c E r r I n f o . h h                       */
/*                                                                            */
/* (c) 2043 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/*                                                                            */
/******************************************************************************/

#include <string.h>      // For strlcpy()
#include <sys/types.h>

#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                              X r d O u c E I                               */
/******************************************************************************/

struct XrdOucEI      // Err information structure
{ 
 static const size_t Max_Error_Len = 2048;
 static const int    Path_Offset   = 1024;

const      char *user;
           int   code;
           char  message[Max_Error_Len];

           void clear(const char *usr=0) 
                     {code=0; message[0]='\0'; user = (usr ? usr : "?");}

           XrdOucEI &operator =(const XrdOucEI &rhs)
               {code = rhs.code;
                user = rhs.user;
                strcpy(message, rhs.message); 
                return *this;
               }
           XrdOucEI(const char *usr) {clear(usr);}
};

/******************************************************************************/
/*                         X r d O u c E r r I n f o                          */
/******************************************************************************/

class XrdOucEICB;
class XrdOucEnv;
class XrdSysSemaphore;
  
class XrdOucErrInfo
{
public:
       void  clear() {ErrInfo.clear();}

inline void  setErrArg(unsigned long long cbarg=0) {ErrCBarg = cbarg;}
inline void  setErrCB(XrdOucEICB *cb, unsigned long long cbarg=0)
                     {ErrCB = cb; ErrCBarg = cbarg;}
inline int   setErrCode(int code) {return ErrInfo.code = code;}
inline int   setErrInfo(int code, const char *message)
                {strlcpy(ErrInfo.message, message, sizeof(ErrInfo.message));
                 return ErrInfo.code = code;
                }
inline int   setErrInfo(int code, const char *txtlist[], int n)
                {int i, j = 0, k = sizeof(ErrInfo.message), l;
                 for (i = 0; i < n && k > 1; i++)
                     {l = strlcpy(&ErrInfo.message[j], txtlist[i], k);
                      j += l; k -= l;
                     }
                 return ErrInfo.code = code;
                }
inline void  setErrUser(const char *user) {ErrInfo.user = (user ? user : "?");}

inline unsigned long long  getErrArg() {return ErrCBarg;}

inline char               *getMsgBuff(int &mblen)
                                   {mblen = sizeof(ErrInfo.message);
                                    return ErrInfo.message;
                                   }
inline XrdOucEICB         *getErrCB() {return ErrCB;}
inline XrdOucEICB         *getErrCB(unsigned long long &ap) 
                                   {ap = ErrCBarg; return ErrCB;}
inline int                 getErrInfo() {return ErrInfo.code;}
inline int                 getErrInfo(XrdOucEI &errorParm)
                                   {errorParm = ErrInfo; return ErrInfo.code;}
inline const char         *getErrText()
                                   {return (const char *)ErrInfo.message;}
inline const char         *getErrText(int &ecode)
                                   {ecode = ErrInfo.code; 
                                    return (const char *)ErrInfo.message;}
inline const char         *getErrUser() {return ErrInfo.user;}

inline XrdOucEnv          *getEnv() {return (ErrCB ? 0 : ErrEnv);}

inline XrdOucEnv          *setEnv(XrdOucEnv *newEnv)
                                 {XrdOucEnv *oldEnv = (ErrCB ? 0 : ErrEnv);
                                  ErrEnv = newEnv;
                                  ErrCB  = 0;
                                  return oldEnv;
                                 }

inline const char         *getErrData()
                                 {return (dOff < 0 ? 0 : ErrInfo.message+dOff);}

inline void                setErrData(const char *Data, int Offs=0)
                                 {if (!Data) dOff = -1;
                                     else {strlcpy(ErrInfo.message+Offs, Data,
                                                   sizeof(ErrInfo.message)-Offs);
                                           dOff = Offs;
                                          }
                                 }

inline int                 getErrMid() {return mID;}

inline void                setErrMid(int  mid) {mID = mid;}

         XrdOucErrInfo &operator =(const XrdOucErrInfo &rhs)
                        {ErrInfo = rhs.ErrInfo;
                         ErrCB   = rhs.ErrCB;
                         ErrCBarg= rhs.ErrCBarg;
                         mID     = rhs.mID;
                         dOff    = -1;
                         return *this;
                        }

         XrdOucErrInfo(const char *user=0,XrdOucEICB *cb=0,
                       unsigned long long ca=0, int mid=0)
                    : ErrInfo(user), ErrCB(cb), ErrCBarg(ca), mID(mid), 
                      dOff(-1), Reserved0(0), Reserved1(0) {}

         XrdOucErrInfo(const char *user,XrdOucEnv *envp)
                    : ErrInfo(user), ErrCB(0), ErrEnv(envp), mID(0),
                      dOff(-1), Reserved0(0), Reserved1(0) {}

         XrdOucErrInfo(const char *user, int MonID)
                    : ErrInfo(user), ErrCB(0), ErrCBarg(0), mID(MonID),
                      dOff(-1), Reserved0(0), Reserved1(0) {}

virtual ~XrdOucErrInfo() {}

protected:

XrdOucEI            ErrInfo;
XrdOucEICB         *ErrCB;
union {
unsigned long long  ErrCBarg;
XrdOucEnv          *ErrEnv;
      };
int                 mID;
short               dOff;
short               Reserved0;
void               *Reserved1;
};

/******************************************************************************/
/*                            X r d O u c E I C B                             */
/******************************************************************************/

class XrdOucEICB
{
public:

// Done() is invoked when the requested operation completes. Arguments are:
//        Result - the original function's result (may be changed).
//        eInfo  - Associated error information. The eInfo object may not be
//                 modified until it's own callback Done() method is called, if
//                 supplied. If the callback function in eInfo is zero, then the
//                 eInfo object is deleted by the invoked callback. Otherwise,
//                 that method must be invoked by this callback function after
//                 the actual callback message is sent. This allows the callback
//                 requestor to do post-processing and be asynchronous.
//        Path   - Optionally, the path related to thid request. It is used
//                 for tracing and detailed monitoring purposes.
//
//
virtual void        Done(int           &Result,   //I/O: Function result
                         XrdOucErrInfo *eInfo,    // In: Error Info
                         const char    *Path=0)=0;// In: Relevant path

// Same() is invoked to determine if two arguments refer to the same user.
//        True is returned if so, false, otherwise.
//
virtual int         Same(unsigned long long arg1, unsigned long long arg2)=0;

                    XrdOucEICB() {}
virtual            ~XrdOucEICB() {}
};
#endif
