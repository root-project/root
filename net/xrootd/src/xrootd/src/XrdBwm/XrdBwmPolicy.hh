#ifndef __BWM_POLICY__
#define __BWM_POLICY__
/******************************************************************************/
/*                                                                            */
/*                       X r d B w m P o l i c y . h h                        */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

class XrdBwmPolicy
{
public:

/* General note: Each request is to be identified by an int-sized handle.
                 The value of the handle is unique with respect to all of the
                 requests that are active and queued. Once a request leaves
                 the system (i.e., cancelled or released) the handle may be
                 re-used. Handle signs are immaterial. That is the property
                 "n == abs(-n) == <same request>" always must hold. Note that
                 Schedule() uses negative handles to merely indicate queuing.
*/

/* Dispatch() returns the handle of the next request that may become active
   because the resources are now available or that must be terminated
   because resources are not available. The returned value must have the
   the following property: "Dispatch() == abs(Schedule()) == <same request>".
   Hence, the handle returned by Dispatch() must be one previously returned by
   Schedule() that was negative to indicate that the request was queued. The
   sign of the returned handle indicates success or failure:

   returns <  0: The associated previously scheduled request cannot obtain 
                 the resource. RespBuff, of size RespSize, should contain
                 null terminated text describing the failure. Done() will not
                 called for the returned handle.
   returns >= 0: The associated previously scheduled request can now be
                 dispatched as resources are available. RespBuff, of size
                 RespSize, should contain any visa information, as an
                 ASCII null terminated string to be sent to client. If none,
                 it should contain a null string (i.e., zero byte). Done()
                 will be called for the returned handle when the resource is no
                 longer needed.

   Dispatch() blocks until a request is ready or has failed.
*/

virtual int  Dispatch(char *RespBuff, int RespSize) = 0;

/* Done() indicates that the resources with a previous request associated with
   the handle, as returned by Dispatch() and Schedule(). When Done() is called
   with a handle referring to a queued request, the request should be cancelled
   and removed from the queue. If the handle refers to an active request (i.e.,
   a non-negative one that was returned by Dispatch()), the resources associated
   with the dispatched request are no longer needed and are to be made available
   to another request. The value returned by Done() indicates what happened:

   returns < 0: The queued request was cancelled.
   returns = 0: No request matching the handle was found.
   returns > 0: The resources associated with the dispatched request returned.

   The handle itself may be a positive or negative, as returned by Dispatch()
   and Schedule(). Note that "n == abs(-n) == <same request>", so the sign
   of the handle should be immaterial to Done(). Negative handles returned by
   Dispatch() indicate failure and thus Done() will not be called for such
   handles. Handles returned by Schedule() may be postive or negative.
*/

virtual int  Done(int rHandle) = 0;

/* Schedule() is invoked when the caller wishes to obtain resources controlled
   by the policy. The caller passes a pointer to a response buffer, RespBuff,
   of size contained in RespSize, to hold hold any response. Additionally. a
   reference to the SchedParms struct that contains information about the 
   nature of the request. Schedule() may choose to immediately allow the 
   resourse to be used, fail the request, or to defer the request. 
   This is indicated by the returned int, as follows:

   returns < 0: The request has been queued. The returned value is the handle
                for the request and is to be used as the argument to Done() to
                cancel the queued request.

   returns = 0: The request failed. The RespBuff should contain any error text
                or a null byte if no text is present.

   returns > 0: The request succeeded and the resource can be used. The returned
                value is the handle for the request and is to be used as the
                argument to Done() to release the associated request resource.

                RespBuff should contain any visa information, as an ASCII null
                         terminated string to be sent to client. If none, it
                         must contain a null string (i.e., zero byte).
*/
enum Flow {Incomming = 0, Outgoing};

struct SchedParms
{
const char  *Tident;     // In: -> Client's trace identity
      char  *Lfn;        // In: -> Logical File Name
      char  *LclNode;    // In: -> Local  node involved in the request
      char  *RmtNode;    // In: -> Remote node involved in the request
      Flow   Direction;  // In: -> Data flow relative to Lclpoint (see enum)
};

virtual int  Schedule(char *RespBuff, int RespSize, SchedParms &Parms) = 0;

/* Status() returns the number of requests as three items via parameters:
            numqIn  - Number of incomming data requests queued
            numqOut - Number of outgoing  data requests queued
            numXeq  - Number of requests that are active (in or out).
*/

virtual void Status(int &numqIn, int &numqOut, int &numXeq) = 0;

             XrdBwmPolicy() {}

virtual     ~XrdBwmPolicy() {}
};
  
/******************************************************************************/
/*                    X r d B w m P o l i c y O b j e c t                     */
/******************************************************************************/

class XrdSysLogger;
  
/* XrdBwmPolicyObject() is called to obtain an instance of the policy object
   that will be used for all subsequent policy scheduling requests. If it 
   returns a null pointer; initialization fails and the program exits. 
   The args are:

   lp    -> XrdSysLogger to be tied to an XrdSysError object for messages
   cfn   -> The name of the configuration file
   parm  -> Parameters specified on the policy lib directive. If none it's zero.
*/

extern "C" XrdBwmPolicy *XrdBwmPolicyObject(XrdSysLogger *lp,
                                            const char   *cfn,
                                            const char   *parm);
#endif
