#ifndef __OOUC_STREAM__
#define __OOUC_STREAM__
/******************************************************************************/
/*                                                                            */
/*                       X r d O u c S t r e a m . h h                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//          $Id$

#include <sys/types.h>
#include <signal.h>
#include <stdlib.h>
#ifdef WIN32
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdSys/XrdSysError.hh"

class XrdOucEnv;

class XrdOucStream
{
public:

// When creating a stream object, you may pass an optional error routing object.
// If you do so, error messages will be writen via the error object. Otherwise,
// errors will be returned quietly.
//
            XrdOucStream(XrdSysError *erobj=0, const char *ifname=0,
                         XrdOucEnv   *anEnv=0, const char *Pfx=0);

           ~XrdOucStream() {Close(); if (myInst) free(myInst);
                                     if (varVal) delete [] varVal;
                                     if (llBuff) free(llBuff);
                           }

// Attach a file descriptor to an existing stream. Any curently associated
// stream is closed and detached. An optional buffer size can be specified.
// Zero is returned upon success, otherwise a -1 (use LastError to get rc).
//
int          Attach(int FileDescriptor, int bsz=2047);
int          AttachIO(int infd, int outfd, int bsz=2047);

// Close the current stream and release the associated buffer.
//
void         Close(int hold=0);

// Detach a file descriptor from a stream. This should be called prior to
// close/delete when you are managing your own descriptors. Return the FD num.
//
int          Detach() {int oldFD = FD; FD = FE = -1; return oldFD;}

// Wait for an Exec() to finish and return the ending status. Use this
// function only when you need to find out the ending status of the command.
//
int          Drain();

// Display last valid line if variable substitution enabled. Fully formed
// input lines are displayed if 'set -v' was encountered (only when using
// the GetxxxWord() methods),
//
void         Echo();

// Execute a command on a stream. Returns 0 upon success or -1 otherwise.
// Use LastError() to get the actual error code. Subsequent Get() calls
// will return the standard output of the executed command. If inrd=1 then
// standardin is redirected so that subqseuent Put() calls write to the
// process via standard in. When inrd=-1 then the current attached FD's are
// used to redirect STDIN and STDOUT of the child process. Standard error
// is handled as determined by the efd argument:
// efd < 0 -> The current stderr file decriptor is unchanged.
// efd = 0 -> The stderr file descriptor is set to the original logging FD
// efd > 0 -> The stderr file descriptor is set to the value of efd.
//
int          Exec(const char *,  int inrd=0, int efd=0);
int          Exec(      char **, int inrd=0, int efd=0);

// Get the file descriptor number associated with a stream
//
int          FDNum() {return FD;}
int          FENum() {return FE;}

// Flush any remaining output queued on an output stream.
//
void         Flush() {fsync(FD); if (FE != FD) fsync(FE);}

// Get the next record from a stream. Return null upon eof or error. Use
// LastError() to determine which condition occurred (an error code of 0
// indicates that end of file has been reached). Upon success, a pointer
// to the next record is returned. The record is terminated by a null char.
//
char        *GetLine();

// Get the next blank-delimited token in the record returned by Getline(). A
// null pointer is returned if no more tokens remain. Each token is terminated
// a null byte. Note that the record buffer is modified during processing. The
// first form returns simply a token pointer. The second form returns a token
// pointer and a pointer to the remainder of the line with no leading blanks.
// The lowcase argument, if 1, converts all letters to lower case in the token.
// RetToken() simply backups the token scanner one token. None of these
// methods perform variable substitution (see GetxxxWord() below).
//
char        *GetToken(int lowcase=0);
char        *GetToken(char **rest, int lowcase=0);
void         RetToken();

// Get the next word, ignoring any blank lines and comment lines (lines whose
// first non-blank is a pound sign). Words are returned until logical end of
// line is encountered at which time, a null is returned. A subsequent call
// will return the next word on the next logical line. A physical line may be
// continued by placing a back slash at it's end (i.e., last non-blank char).
// GetFirstWord() always makes sure that the first word of a logical line is
// returned (useful for start afresh after a mid-sentence error). GetRest()
// places the remining tokens in the supplied buffer; returning 0 if the
// buffer was too small. All of these methods perform variable substitution
// should an XrdOucEnv object be passed to the constructor.
//
char        *GetFirstWord(int lowcase=0);
char        *GetMyFirstWord(int lowcase=0);
int          GetRest(char *theBuf, int Blen, int lowcase=0);
char        *GetWord(int lowcase=0);

// Indicate wether there is an active program attached to the stream
//
#ifndef WIN32
inline int  isAlive() {return (child ? kill(child,0) == 0 : 0);}
#else
inline int  isAlive() {return (child ? 1 : 0);}
#endif

// Return last error code encountered.
//
inline int   LastError() {int n = ecode; ecode = 0; return n;}

// Return the last input line
//
char        *LastLine() {return recp;}

// Suppress echoing the previous line when the next line is fetched.
//
int          noEcho() {llBok = 0; return 0;}

// Write a record to a stream, if a length is not given, then the buffer must
// be null terminated and this defines the length (the null is not written).
//
int          Put(const char *data, const int dlen);
inline int   Put(const char *data) {return Put(data, strlen(data));}

// Write record fragments to a stream. The list of fragment/length pairs ends
// when a null pointer is encountered.
//
int          Put(const char *data[], const int dlen[]);

// Set the Env (returning the old Env). This is useful for suppressing
// substitutions for a while.
//
XrdOucEnv   *SetEnv(XrdOucEnv *newEnv)
                   {XrdOucEnv *oldEnv = myEnv; myEnv = newEnv; return oldEnv;}

// Set error routing
//
void         SetEroute(XrdSysError *eroute) {Eroute = eroute;}

// A 0 indicates that tabs in the stream should be converted to spaces.
// A 1 inducates that tabs should be left alone (the default).
//
void         Tabs(int x=1) {notabs = !x;}

// Wait for inbound data to arrive. The argument is the max number of millisec
// to wait (-1 means wait forever). Returns 0 if data is present. Otherwise,
// -1 indicates that the connection timed out, a positive value indicates an
// error and the value is the errno describing the error.
//
int          Wait4Data(int msMax=-1);

/******************************************************************************/
  
private:
        char *add2llB(char *tok, int reset=0);
        char *doelse();
        char *doif();
        int   isSet(char *var);
        char *vSubs(char *Var);
        int   xMsg(const char *txt1, const char *txt2=0, const char *txt3=0);

static const int maxVLen = 512;
static const int llBsz   = 1024;

        int   FD;
        int   FE;
        int   bsize;
        int   bleft;
        char *buff;
        char *bnext;
        char *recp;
        char *token;
        int   flags;
        pid_t child;
        int   ecode;
        int   notabs;
        int   xcont;
        int   xline;
        char *myInst;
        char *myHost;
        char *myName;
        char *myExec;
 XrdSysError *Eroute;
 XrdOucEnv   *myEnv;
        char *varVal;
 const  char *llPrefix;
        char *llBuff;
        char *llBcur;
        int   llBleft;
        char  Verbose;
        char  sawif;
        char  skpel;
        char  llBok;
};
#endif
