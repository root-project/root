#ifndef __SFS_AIO_H__
#define __SFS_AIO_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d S f s A i o . h h                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

#include <signal.h>
#include <sys/types.h>
#ifdef _POSIX_ASYNCHRONOUS_IO
#ifdef __macos__
#include <AvailabilityMacros.h>
#include <sys/aio.h>
#else
#include <aio.h>
#endif
#else
struct aiocb {           // Minimal structure to avoid compiler errors
       int    aio_fildes;
       void  *aio_buf;
       size_t aio_nbytes;
       off_t  aio_offset;
       int    aio_reqprio;
       struct sigevent aio_sigevent;
      };
#endif

// The XrdSfsAIO class is meant to be derived. This object provides the
// basic interface to handle AIO control block queues not processing.
//
class XrdSfsAio
{
public:

struct aiocb sfsAio;

ssize_t      Result; // If >= 0 valid result; else is -errno

const char  *TIdent; // Trace information (optional)

// Method to handle completed reads
//
virtual void doneRead() = 0;

// Method to hand completed writes
//
virtual void doneWrite() = 0;

// Method to recycle free object
//
virtual void Recycle() = 0;

             XrdSfsAio() {
#if defined(__macos__) && (!defined(MAC_OS_X_VERSION_10_4) || \
    MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_4)
                         sfsAio.aio_sigevent.sigev_value.sigval_ptr = (void *)this;
#else
                         sfsAio.aio_sigevent.sigev_value.sival_ptr  = (void *)this;
#endif
                         sfsAio.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
                         sfsAio.aio_reqprio = 0;
                         TIdent = (char *)"";
                        }
virtual     ~XrdSfsAio() {}
};
#endif
