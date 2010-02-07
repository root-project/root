#ifndef __OOUC_PROG__
#define __OOUC_PROG__
/******************************************************************************/
/*                                                                            */
/*                         X r d O u c P r o g . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//          $Id$

#include <sys/types.h>

class XrdSysError;
class XrdOucStream;

class XrdOucProg
{
public:

// When creating an Prog object, you may pass an optional error routing object.
// If you do so, error messages and all command output will be writen via the 
// error object. Otherwise, errors will be returned quietly.
//
            XrdOucProg(XrdSysError *errobj=0)
                      {eDest = errobj; myStream = 0;
                       ArgBuff = Arg[0] = 0; numArgs = 0; theEFD = -1;
                      }

           ~XrdOucProg();

// Feed() send a data to the program started by Start(). Several variations
// exist to accomodate various needs. Note that should the program not be
// running when Feed() is called, it is restarted.
//
int Feed(const char *data[], const int dlen[]);

int Feed(const char *data, int dlen)
        {const char *myData[2] = {data, 0};
         const int   myDlen[2] = {dlen, 0};
         return Feed(myData, myDlen);
        }

int Feed(const char *data) {return Feed(data, (int)strlen(data));}

// getStream() returns the stream created by Start(). Use the object to get
// lines written by the started program.
//
XrdOucStream *getStream() {return myStream;}

// Run executes the command that was passed via Setup(). You may pass
// up to four additional arguments that will be added to the end of any
// existing arguments. The ending status code of the program is returned.
//
int          Run(XrdOucStream *Sp,  const char *arg1=0, const char *arg2=0,
                                    const char *arg3=0, const char *arg4=0);

int          Run(const char *arg1=0, const char *arg2=0,
                 const char *arg3=0, const char *arg4=0);


// Start executes the command that was passed via Setup(). The started
// program is expected to linger so that you can send directives to it
// via its standard in. Use Feed() to do this. If the output of the command
// is wanted, use getStream() to get the stream object and use it to read
// lines the program sends to standard out.
//
int          Start(void);

// Setup takes a command string, checks that the program is executable and
// sets up a parameter list structure.
// Zero is returned upon success, otherwise a -errno is returned,
//
int          Setup(const char *prog, XrdSysError *errP=0);

/******************************************************************************/
  
private:
  int           Restart();
  XrdSysError  *eDest;
  XrdOucStream *myStream;
  char         *ArgBuff;
  char         *Arg[64];
  int           numArgs;
  int           lenArgs;
  int           theEFD;
};
#endif
