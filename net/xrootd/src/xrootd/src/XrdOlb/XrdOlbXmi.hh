#ifndef __XRDOLBXMI_H__
#define __XRDOLBXMI_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d O l b X m i . h h                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <sys/types.h>

#include "XrdOlb/XrdOlbReq.hh"
  
/*
   The XrdOlbXmi class defines the interface the olbd uses to an external
   manager. When the olb.xmilib directive is specified, the olbd loads the
   plugin using XrdOlbgetXmi() to instantiate the plugin objects, as explained
   after the definition of the abstract class. The olbd then uses the Xmi
   defined methods to process certain requests instead of it's own default 
   methods. As the Xmi interface runs synchronously with respect to being called, 
   it should dispatch any long work to another thread to allow the next request 
   to be processed by the plugin. You should use the thread scheduler passed
   via the XrdOlbXmiEnv structure (see below).

   Each method (other that Prep(), see below) is passed an XrdOlbReq object.
   This object must be used to send a reply to the client (only one reply is 
   allowed). A reply is *mandatory* if the function returns TRUE; even if that
   reply is merely an indication that everything succeeded. A reply must *not*
   be sent if the function returns FALSE; as a reply will be sent by the driver.
   Refer to XrdOlbReq.hh on the description of how replies are effected.

   The Prep() method is a background function and the client never expects a
   reply. Therefore, no request object is passed since no reply is possible.
   Instead, the first parameter is a request ID that is used to tag the
   request. This ID may be passed later with XMI_CANCEL set to cancel and
   path passed as a null string. All preparation for files tagged with request
   ID should be stopped, if possible.

   The Xmi methods may be called in one of two modes, DIRECT or INDIRECT.
   The modes are indicated by XeqMode(). In DIRECT mode the Xmi methods are
   called synchronously. In INDIRECT mode, the Xmi methods are given thread as
   well as client isolation by the olbd using an asynchronous encapsulated
   callback mechanism. Normally, INDIRECT mode should be used. The 
   implementation is not particularly relevant as the protocol details are
   handled by the Xmi driver and the request object.

   Each method must return either true (1) or false (0). However, the action
   taken based on the return value depends on the calling mode.

   TRUE  (1)  -> The function was processed and a reply was sent.
         Action: INDIRECT: Normal processing continues, the request was done.
                   DIRECT: Same as above.
   FALSE (0)  -> The function was *not* processed and *no* reply was sent.
         Action: INDIRECT: An error reply is sent and processing continues.
                   DIRECT: Processing continues as if the Xmi was not present.

   See the description of XeqMode() on how to indicate which methods are to
   be called and which mode each method requires.
*/

/******************************************************************************/
/*                          X r d O l b X m i E n v                           */
/******************************************************************************/
  
/* The XrdOlbXmiEnv structure is passed to XrdOlbgetXmi() and contains
   information that may be relevant to the Xmi object. The information is
   static in that it persists during the execution of the program.
*/

class XrdSysError;
class XrdInet;
class XrdScheduler;
class XrdOucName2Name;
class XrdOucTrace;

struct XrdOlbXmiEnv
{
const char      *Role;          // Manager | Peer | Proxy | Supervisor | Server
const char      *ConfigFN;      // -> Config file name
const char      *Parms;         // -> Optional parms from xmilib directive
XrdSysError     *eDest;         // -> Error message handler
XrdInet         *iNet;          // -> Network object
XrdScheduler    *Sched;         // -> Thread scheduler
XrdOucTrace     *Trace;         // -> Trace handler
XrdOucName2Name *Name2Name;     // -> lfn to xxx mapper (may be null)
};

/******************************************************************************/
/*                             X r d O l b X m i                              */
/******************************************************************************/
  
class XrdOlbPrepArgs;

// Flags passed to Prep():   XMI_RW, XMI_CANCEL
// Flags passed to Select(): XMI_RW, XMI_NEW, XMI_TRUNC
//
#define XMI_RW     0x0001
#define XMI_NEW    0x0002
#define XMI_TRUNC  0x0004
#define XMI_CANCEL 0x0008
#define XMI_LOCATE 0x0010

// Flags to be passed back by XeqMode()
//
#define XMI_CHMOD  0x00000001
#define XMI_LOAD   0x00000002
#define XMI_MKDIR  0x00000004
#define XMI_MKPATH 0x00000008
#define XMI_PREP   0x00000010
#define XMI_RENAME 0x00000020
#define XMI_REMDIR 0x00000040
#define XMI_REMOVE 0x00000080
#define XMI_SELECT 0x00000100
#define XMI_SPACE  0x00000200
#define XMI_STAT   0x00000400
#define XMI_ALL    0x000007ff

class XrdOlbXmi
{
public:

// Called when trying to change the mode of a file
//
virtual int  Chmod (      XrdOlbReq      *Request,
                    const char           *path,
                          mode_t          mode) = 0;

// Called when trying to determine the load on this host (not yet implemented)
//
virtual int  Load  (      XrdOlbReq      *Request) {return 0;} // Server only

// Called to make a directory
//
virtual int  Mkdir (      XrdOlbReq      *Request,
                    const char           *path,
                          mode_t          mode) = 0;

// Called to make a directory path
//
virtual int  Mkpath(      XrdOlbReq      *Request,
                    const char           *path,
                          mode_t          mode) = 0;

// Called to prepare future access to a file
//
virtual int  Prep  (const char           *ReqID,
                    const char           *Path,
                          int             Opts) = 0;
// Called to rename a file or directory
//
virtual int  Rename(      XrdOlbReq      *Request,
                    const char           *oldpath,
                    const char           *newpath) = 0;

// Called to remove a directory
//
virtual int  Remdir(      XrdOlbReq      *Request,
                    const char           *path) = 0;

// Called to remove a file
//
virtual int  Remove(      XrdOlbReq      *Request,
                    const char           *path) = 0;

// Called when a client attempts to open a file. The opts indicate how the
// file will used and whether it is to be created.
//
virtual int  Select(      XrdOlbReq      *Request, // See description above
                    const char           *path,
                          int             opts) = 0;

// Called to determine how much space exists in this server (not implemented)
//
virtual int  Space (      XrdOlbReq      *Request) {return 0;}  // Server Only

// Called to get information about a file
//
virtual int  Stat  (      XrdOlbReq      *Request,
                    const char           *path) = 0;

// Called after the plugin is loaded to determine which and how the above
// methods are to be called.
//
virtual void XeqMode(unsigned int &isNormal, 
                     unsigned int &isDirect)
                    {isNormal = XMI_LOAD | XMI_SPACE; isDirect = 0;}

            XrdOlbXmi() {}
virtual    ~XrdOlbXmi() {}
};

/*
   The XrdOlbXmi object is intended to be supplied as a plugin from a shared
   library. This library is identified by the "olb.xmilib" directive. When
   the library is loaded the following extern "C" function is called to obtain
   an instance of the XrdOlbXmi object that will be used for request processing.
   The function is passed the command line arguments (xrd options stripped) and
   a pointer to the XrdOlbXmiEnv structure. If the function returns a null
   pointer, the olbd exits with an error.

   After the object is obtained, XeqMode() is called to determine how each
   method is to operate by default. A value must be set in each provided mask
   for each method, as applicable. Two mask are supplied:

   isNormal           When the XMI_func bit is set in this mask, the corresponding
                      method is executed in the normal way the olbd would
                      have done it had the plugin not existed. Otherwise,
                      you will have to indicate this at run-time for each call
                      which can only be done in direct calling mode.

   isDirect           When the XMI_func bit is set in this mask, the corresponding
                      method is called directly without thread isolation. Use
                      this mode if the processing is immediate (e.g., you will
                      be imediately redirecting the client). By default, the
                      client is told to wait for a defered response and the
                      request is queued for a thread running the Xmi plugin.
                      Three threads are used to drive the Xmi:
                      1) A thread to feed Prep()
                      2) A thread to feed Select()
                      3) A thread to feed everything else.
                      Warning! The three thread model obviously affects how
                               objects can be shared!
*/

extern "C"
{
XrdOlbXmi *XrdOlbgetXmi(int argc, char **argv, XrdOlbXmiEnv *XmiEnv);
}
#endif
