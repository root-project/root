#ifndef __CMS_CLIENT__
#define __CMS_CLIENT__
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s C l i e n t . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

class  XrdOucEnv;
class  XrdOucErrInfo;
class  XrdOucLogger;
class  XrdOucTList;
struct XrdSfsPrep;

/******************************************************************************/
/*                    R e t u r n   C o n v e n t i o n s                     */
/******************************************************************************/
  
/* The following return conventions are use by Forward(), Locate(), & Prepare()
   Return Val   Resp.errcode          Resp.errtext
   ---------    -------------------   --------
   SFS_DATA     Length of data.       Data to be returned to caller.
                Action: Caller is provided data as successful response.

   SFS_ERROR    errno                 Error message text.
                Action: Caller given error response.

   SFS_REDIRECT port (0 for default)  Host name
                Action: Caller is redirected to <host>:<port>

   SFS_STARTED  Expected seconds      n/a
                Action: Caller is told to wait for the "expected seconds" for a
                        callback with the result. A callback must follow.
                        See how to do callbacks below.

   > 0          Wait time (= retval)  Reason for wait
                Action: Caller told to wait retval seconds and retry request.

   < 0          Error number          Error message
                Action: Same as SFS_ERROR. You should *always* use SFS_ERROR.

   = 0          Not applicable        Not applicable (see below)
                Action: Forward() -> Return success; request forwarded.
                        Locate()  -> Redirection does not apply, operation
                                     should be done against local file system.
                        Prepare() -> Return success, request submitted.
*/

/******************************************************************************/
/*                  C a l l b a c k   C o n v e n t i o n s                   */
/******************************************************************************/
  
/* Most operations allow you to return SFS_STARTED to setup a callback.
   Callback information is contained in the XrdOucErrInfo object passed to
   Forward(), Locate() and Prepare(); the only methods that can apply callbacks.
   Use a callback when the operation will take at least several seconds so as
   to not occupy the calling thread for an excessive amount of time.

   The actual mechanics of a callback are rather complicated because callbacks
   are subject to non-causaility if not correctly handled. In order to avoid
   such issues, you should use the XrdOucCallBack object (see XrdOucCallBack.hh)
   to test for applicability, setup, and effect a callback.

   When calling back, you return the same information you would have returned
   had the execution path been synchronous. From that standpoint callbacks are
   relatively easy to understand. All you are doing is defering the return of
   information without occupying a thread while waiting to do so.

   A typical scenario, using Resp and the original ErrInfo object, would be....

   XrdOucCallBack cbObject;  // Must be persistent for the callback duration

   if (XrdOucCallBack::Allowed(Resp))
      {cbObject.Init(Resp);
       <hand off the cbObject to a thread that will perform the work>
       Resp.setErrCode(<seconds end-point should wait>);
       return SFS_STARTED; // Effect callback response!
      }

   Once the thread doing the work has a result, send it via a callback as if
   the work was done in a synchronous fashion.

   cbObject->Reply(retValue, ErrCodeValue, ErrTextValue);
*/

/******************************************************************************/
/*                    C l a s s   X r d C m s C l i e n t                     */
/******************************************************************************/
  
class XrdCmsClient
{
public:

// Added() notifies the cms of a newly added file or a file whose state has
//         changed. It is only used on data server nodes. When Pend is true,
//         the file is scheduled to be present in the future (e.g. copied in).
//
virtual void   Added(const char *path, int Pend=0) {}

// Configue() is called to configure the client. If the client is obtained via
//            a plug-in then Parms are whether parameters were specified after
//            cmslib path. It is zero if no parameters exist.
// Return:    If successful, true must be returned; otherise, false/
//
virtual int    Configure(const char *cfn, char *Parms, XrdOucEnv *EnvInfo) = 0;

// Forward() relays a meta-operation to all nodes in the cluster. It is only
//           used on manager nodes and is enabled by the ofs.forward directive.
//           The 'cmd" specified what command is must be forwarded (see table).
//           If it starts with a '+' then a response (2way) is needed.
//           Otherwise, a best-effort is all that is all that is required and
//           success can always be returned. The "Env" arguments provide
//           associated environmental information. For instance, opaque data
//           can be retrieved by Env->Env(<len>). The following is passed:

//           cmd       arg1    arg2           cmd       arg1    arg2
//           --------  ------  ------         --------  ------  ------
//           [+]chmod  <path>  <mode %o>      [+]rmdir  <path>  0
//           [+]mkdir  <path>  <mode %o>      [+]mv     <oldp>  <newp>
//           [+]mkpath <path>  <mode %o>      [+]trunc  <path>  <size %lld>
//           [+]rm     <path>  0

// Return:   As explained under "return conventions".
//
virtual int    Forward(XrdOucErrInfo &Resp,   const char *cmd,
                       const char    *arg1=0, const char *arg2=0,
                       XrdOucEnv     *Env1=0, XrdOucEnv  *Env2=0) {return 0;}

// isRemote() returns true of this client is configured for a manager node.
//
virtual int    isRemote() {return myPersona == XrdCmsClient::amRemote;}

// Locate() is called to retrieve file location information. It is only used
//          on a manager node. This can be the list of servers that have a
//          file or the single server that the client should be sent to. The
//          "flags" indicate what is required and how to process the request.

//          SFS_O_LOCATE  - return the list of servers that have the file.
//                          Otherwise, redirect to the best server for the file.
//          SFS_O_NOWAIT  - w/ SFS_O_LOCATE return readily available info.
//                          Otherwise, select online files only.
//          SFS_O_CREAT   - file will be created.
//          SFS_O_NOWAIT  - select server if file is online.
//          SFS_O_REPLICA - a replica of the file will be made.
//          SFS_O_STAT    - only stat() information wanted.
//          SFS_O_TRUNC   - file will be truncated.

//          For any the the above, additional flags are passed:
//          SFS_O_META    - data will not change (inode operation only)
//          SFS_O_RESET   - reset cached info and recaculate the location(s).
//          SFS_O_WRONLY  - file will be only written    (o/w RDWR   or RDONLY).
//          SFS_O_RDWR    - file may be read and written (o/w WRONLY or RDONLY).

// Return:  As explained under "return conventions".
//
virtual int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv  *Info=0) = 0;

// Managers() is called to obtain the list of cmsd's being used by a manager
//            node along with their associated index numbers, origin 1.
//            This is used by the monitoring systems to report who redirected.
//            The list is considered permanent and is not deleted.

// Return:    A list of managers or null if none exist.
//
virtual
XrdOucTList   *Managers() {return 0;}

// Prepare() is called to start the preparation of a file for future processing.
//           It is only used on a manager node.

// Return:  As explained under "return conventions".
//
virtual int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs,
                       XrdOucEnv  *Info=0) {return 0;}

// Removed() is called when a file or directory has been deleted. It is only
//           called on a data server node.
//
virtual void   Removed(const char *path) {}

// Resume() and Suspend() server complimentary functions and, by default,
//          persist across server restarts. A temporary suspend/resume may be
//          requested by passing a value of 0. Suspend() informs cluster 
//          managers that data services are suspended. Resume() re-enables
//          data services. The default implementation performs nothing.
//
virtual void   Resume (int Perm=1) {}
virtual void   Suspend(int Perm=1) {}

// The following set of functions can be used to control whether or not clients
// are dispatched to this data server based on a virtual resource. The default
// implementations do nothing.
//
// Resource() should be called first and enables the Reserve() & Release()
//            methods. It's argument a positive integer that specifies the
//            amount of resource units that are available. It may be called
//            at any time (though usually it is not) and returns the previous
//            value. This first call will return 0.
// Reserve()  decreases the amount of resources available by the value passed
//            as the argument (default is 1). When the available resources
//            becomes non-positive, a temporary suspend is activated preventing
//            additional clients from being dispatched to this data server.
//            Reserve() returns the amount of resource left.
// Release()  increases the amount of resource available by the value passed
//            as the argument (default 1). The total amount is capped by the
//            amount specified by Resource(). When a transition is made from
//            a non-positive to a positive amount, resume is activated that
//            allows additional clients to be dispatched to this data server.
//            Release() returns the amount of resource left.
//
virtual int    Resource(int n)   {return 0;}
virtual int    Reserve (int n=1) {return 0;}
virtual int    Release (int n=1) {return 0;}

// Space() is called to obtain the overall space usage of a cluster. It is
//         only called on manager nodes.

// Return: Space information as defined by the response to kYR_statfs. Fo a
//         typical implementation see XrdCmsNode::do_StatFS().
//
virtual int    Space(XrdOucErrInfo &Resp, const char *path,
                     XrdOucEnv  *Info=0) = 0;

        enum   Persona {amLocal, amRemote, amTarget};

               XrdCmsClient(Persona acting) : myPersona(acting) {}
virtual       ~XrdCmsClient() {}

protected:

Persona        myPersona;
};

/******************************************************************************/
/*              I n s t a n t i a t i o n   M o d e   F l a g s               */
/******************************************************************************/
  
// The following instantiation mode flags are passed to the instantiator. They
// may be or'd together, depending on which mode the client should operate.
// They are defined as follows:

namespace XrdCms
{
enum  {IsProxy  = 1, // The role is proxy  <one or more of the below>
       IsRedir  = 2, // The role is manager and will redirect users
       IsTarget = 4, // The role is server  and will be a redirection target
       IsMeta   = 8  // The role is meta   <one or more of the above>
      };
}

/******************************************************************************/
/*               C M S   C l i e n t   I n s t a n t i a t o r                */
/******************************************************************************/

// This function is called to obtain an instance of a configured XrdCmsClient
// Object. This is only used of the client is an actual plug-in as identified
// by the ofs.cmslib directive.

// There are two general types of clients, Redir and Target. The plug-in must
// provide an instance of each whether or not they actually do anything.

// Redir  clients are anything other than a data provider (i.e., data servers).
//        These clients are expected to locate files and redirect a requestor
//        to an actual data server.

// Target clients are typically data providers (i.e., data servers) but may
//        actually do other functions are are allowed to redirect as well.

// The instantiator is passed the operational mode (opMode) as defined by the
// enum above. The returned object must provide suitable functions for the mode.

// If successful, the instantiator must return a pointer to the appropriate
// object. Otherwise, a null pointer should be returned upon which server
// initialization fails.

// As this is a plug-in, the plug-in loader searches the cmslib for the
// following extern symbol which must be a "C" type symbol. Once the object
// is obtained, its Configure() method is called to initialize the object.

/*
extern "C"
{
XrdCmsClient *XrdCmsGetClient(XrdSysLogger *Logger, // Where messages go
                              int           opMode, // Operational mode
                              int           myPort, // Server's port number
                              XrdOss       *theSS); // Storage System I/F
}
*/
#endif
