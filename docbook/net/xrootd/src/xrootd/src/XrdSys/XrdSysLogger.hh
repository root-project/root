#ifndef __SYS_LOGGER_H__
#define __SYS_LOGGER_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d S y s L o g g e r . h h                        */
/*                                                                            */
/*(c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University   */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*Produced by Andrew Hanushevsky for Stanford University under contract       */
/*           DE-AC03-76-SFO0515 with the Deprtment of Energy                  */
/******************************************************************************/

//        $Id$

#include <stdlib.h>
#ifndef WIN32
#include <unistd.h>
#include <string.h>
#include <strings.h>
#else
#include <string.h>
#include <io.h>
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdSys/XrdSysPthread.hh"

class XrdSysLogger
{
public:
         XrdSysLogger(int ErrFD=STDERR_FILENO, int xrotate=1);

        ~XrdSysLogger() {if (ePath) free(ePath);}

// Bind allows you to bind standard error to a file with an optional periodic
// closing and opening of the file.
//
int Bind(const char *path, int intsec=0);

// Flush any pending output
//
void Flush() {fsync(eFD);}

// originalFD() returns the base FD that we started with
//
int  originalFD() {return baseFD;}

// Output data and optionally prefix with date/time
//
void Put(int iovcnt, struct iovec *iov);

// Set log file keep value. A negative number means keep abs() files.
//                          A positive number means keep no more than n bytes.
//
void setKeep(long long knum) {eKeep = knum;}

// Set log file rotation on/off. Used by forked processes.
//
void setRotate(int onoff) {doLFR = onoff;}

// Return formated date/time (tbuff must be atleast 24 characters)
//
int Time(char *tbuff);

// traceBeg() obtains  the logger lock and returns a message header.
// traceEnd() releases the logger lock and returns a newline
//
char *traceBeg() {Logger_Mutex.Lock(); Time(TBuff); return TBuff;}
char  traceEnd() {Logger_Mutex.UnLock(); return '\n';}

// xlogFD() returns the FD to be used by external programs as their STDERR.
// A negative one indicates that no special FD is assigned.
//
int   xlogFD();

private:

XrdSysMutex Logger_Mutex;
static int extLFD[4];
long long  eKeep;
char       TBuff[24];        // Trace header buffer
int        eFD;
int        baseFD;
char      *ePath;
char       Filesfx[8];
time_t     eNTC;
int        eInt;
time_t     eNow;
int        doLFR;

void   putEmsg(char *msg, int msz);
int    ReBind(int dorename=1);
void   Trim();
};
#endif
