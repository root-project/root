#ifndef __POSIX_CALLBACK_HH__
#define __POSIX_CALLBACK_HH__
/******************************************************************************/
/*                                                                            */
/*                   X r d P o s i x C a l l B a c k . h h                    */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// This abstract class defines the callback interface for file open() calls.
// It is passed when using the XrdPosixXrootd::Open() call. When passed, the
// open request will be done in the background. When a callback object is
// supplied, Open() will *always* return -1. However, if started successfully,
// Open() will return -1 with errno set to EINPROGRESS. Otherwise, errno will
// contain the reason the Open() request immediately failed. Upon completion,
// the callback's Compete() method is invoked.  The Result parameter will either
// be a non-negative file descriptor or -errno indicating that the Open()
// failed. Note that the caller is responsible for deleting the callback object
// after it has been invoked. Note that callbacks will be executed in a
// separate thread unless open() is called with O_SYNC or maxThreads is zero.
// WARNING: If O_SYNC or maxThreads is zero, then the callback must *not*
//          issue any filesystem calls using the supplied file descriptor.
//          Ignoring this will produce undetermined results including possible
//          deadlock. Synchrnous callbacks are only meant to support private
//          thread management.

class XrdPosixCallBack
{
public:

virtual void Complete(int Result) = 0;

             XrdPosixCallBack() {}
virtual     ~XrdPosixCallBack() {}
};
#endif
