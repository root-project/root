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

//        $Id$

#include <string.h>      // For strlcpy()
#include <sys/types.h>

#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                              X r d O u c E I                               */
/******************************************************************************/

struct XrdOucEI      // Err information structure
{ 
 static const size_t Max_Error_Len = 1280;

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
class XrdSysSemaphore;
  
class XrdOucErrInfo
{
public:
      void  clear() {ErrInfo.clear();}

      void  setErrCB(XrdOucEICB *cb, unsigned long long cbarg=0)
                    {ErrCB = cb; ErrCBarg = cbarg;}
      int   setErrCode(int code)
               {return ErrInfo.code = code;}
      int   setErrInfo(int code, const char *message)
               {strlcpy(ErrInfo.message, message, sizeof(ErrInfo.message));
                return ErrInfo.code = code;
               }
      int   setErrInfo(int code, const char *txtlist[], int n)
               {int i, j = 0, k = sizeof(ErrInfo.message), l;
                for (i = 0; i < n && k > 1; i++)
                    {l = strlcpy(&ErrInfo.message[j], txtlist[i], k);
                     j += l; k -= l;
                    }
                return ErrInfo.code = code;
               }
      void  setErrUser(const char *user) {ErrInfo.user = (user ? user : "?");}

XrdOucEICB *getErrCB() {return ErrCB;}
XrdOucEICB *getErrCB(unsigned long long &ap) {ap = ErrCBarg; return ErrCB;}
      int   getErrInfo() {return ErrInfo.code;}
      int   getErrInfo(XrdOucEI &errorParm)
               {errorParm = ErrInfo; return ErrInfo.code;}
const char *getErrText() 
               {return (const char *)ErrInfo.message;}
const char *getErrText(int &ecode)
               {ecode = ErrInfo.code; return (const char *)ErrInfo.message;}
const char *getErrUser()
               {return ErrInfo.user;}

      XrdOucErrInfo &operator =(const XrdOucErrInfo &rhs)
               {ErrInfo = rhs.ErrInfo; 
                ErrCB   = rhs.ErrCB;
                ErrCBarg= rhs.ErrCBarg;
                return *this;
               }

      XrdOucErrInfo(const char *user=0,XrdOucEICB *cb=0,unsigned long long ca=0)
                   : ErrInfo(user) {ErrCB = cb; ErrCBarg = ca;}

virtual ~XrdOucErrInfo() {}

protected:

XrdOucEI            ErrInfo;
XrdOucEICB         *ErrCB;
unsigned long long  ErrCBarg;
};

/******************************************************************************/
/*                            X r d O u c E I C B                             */
/******************************************************************************/

class XrdOucEICB
{
public:

// Done() is invoked when the requested operation completes. Arguments are:
//        Result - the original function's result (may be changed).
//        eInfo  - Associated error information. The callback function must
//                 manually delete this object when it is through! While icky
//                 this allows callback functions to be asynchronous.
//
virtual void        Done(int           &Result,   //I/O: Function result
                         XrdOucErrInfo *eInfo)=0; // In: Error Info

// Same() is invoked to determine if two argtuments refer to the same user.
//        True is returned if so, false, otherwise.
//
virtual int         Same(unsigned long long arg1, unsigned long long arg2)=0;

                    XrdOucEICB() {}
virtual            ~XrdOucEICB() {}
};
#endif
