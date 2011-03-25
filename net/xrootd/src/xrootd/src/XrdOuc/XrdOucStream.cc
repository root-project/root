/******************************************************************************/
/*                                                                            */
/*                       X r d O u c S t r e a m . c c                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Deprtment of Energy               */
/******************************************************************************/

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef WIN32
#include <poll.h>
#include <unistd.h>
#include <strings.h>
#if !defined(__linux__) && !defined(__CYGWIN__)
#ifdef __FreeBSD__
#include <sys/param.h>
#endif
#include <sys/conf.h>
#endif
#include <sys/stat.h>
#include <sys/termios.h>
#include <sys/types.h>
#include <sys/wait.h>
#else // WIN32
#include "XrdSys/XrdWin32.hh"
#include <process.h>
#endif // WIN32

#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                         l o c a l   d e f i n e s                          */
/******************************************************************************/
  
#define MaxARGC 64
#define XrdOucStream_EOM  0x01
#define XrdOucStream_BUSY 0x02

#define Erq(p, a, b) Err(p, a, b, (char *)0)
#define Err(p, a, b, c) (ecode=(Eroute ? Eroute->Emsg(#p, a, b, c) : a), -1)
#define Erp(p, a, b, c)  ecode=(Eroute ? Eroute->Emsg(#p, a, b, c) : a)

// The following is used by child processes prior to exec() to avoid deadlocks
//
#define Erx(p, a, b) if (Eroute) cerr <<#p <<' ' <<strerror(a) <<' ' <<b <<endl;

/******************************************************************************/
/*               o o u c _ S t r e a m   C o n s t r u c t o r                */
/******************************************************************************/
  
XrdOucStream::XrdOucStream(XrdSysError *erobj, const char *ifname,
                           XrdOucEnv   *anEnv, const char *Pfx)
{
 char *cp;
     if (ifname)
        {myInst = strdup(ifname);
         if (!(cp = index(myInst, ' '))) {cp = myInst; myExec = 0;}
            else {*cp = '\0'; cp++;
                  myExec = (*myInst ? myInst : 0);
                 }
         if ((myHost = index(cp, '@')))
            {*myHost = '\0';
             myHost++;
             myName = (*cp ? cp : 0);
            } else {myHost = cp; myName = 0;}
        } else myInst = myHost = myName = myExec = 0;

     FD     = -1;
     FE     = -1;
     bsize  = 0;
     buff   = 0;
     bnext  = 0;
     bleft  = 0;
     recp   = 0;
     token  = 0;
     flags  = 0;
     child  = 0;
     ecode  = 0;
     notabs = 0;
     xcont  = 1;
     xline  = 0;
     Eroute = erobj;
     myEnv  = anEnv;
     sawif  = 0;
     skpel  = 0;
     if (myEnv && Eroute)
        {llBuff = (char *)malloc(llBsz);
         llBcur = llBuff; llBok = 0; llBleft = llBsz; *llBuff = '\0';
         Verbose= 1;
        } else {
         Verbose= 0;
         llBuff = 0;
         llBcur = 0;
         llBleft= 0;
         llBok  = 0;
        }
     varVal = (myEnv ? new char[maxVLen+1] : 0);
     llPrefix = Pfx;
}

/******************************************************************************/
/*                                A t t a c h                                 */
/******************************************************************************/

int XrdOucStream::AttachIO(int infd, int outfd, int bsz)
{
    if (Attach(infd, bsz)) return -1;
    FE = outfd;
    return 0;
}
  
int XrdOucStream::Attach(int FileDescriptor, int bsz) 
{

    // Close the current stream. Close will handle unopened streams.
    //
    Close();

    // Allocate a new buffer for this stream
    //
    if (!bsz) buff = 0;
       else if (!(buff = (char *)malloc(bsz+1)))
               return Erq(Attach, errno, "allocate stream buffer");

    // Initialize the stream
    //
    FD= FE = FileDescriptor;
    bnext  = buff;
    bsize  = bsz+1;
    bleft  = 0;
    recp   = 0;
    token  = 0;
    flags  = 0;
    ecode  = 0;
    xcont  = 1;
    xline  = 0;
    sawif  = 0;
    skpel  = 0;
    if (llBuff) 
       {llBcur = llBuff; *llBuff = '\0'; llBleft = llBsz; llBok = 0;}
    return  0;
}
  
/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/

void XrdOucStream::Close(int hold)
{

    // Wait for any associated process on this stream
    //
    if (!hold) Drain();
       else child = 0;

    // Close the associated file descriptor if it was open
    //
    if (FD >= 0)             close(FD);
    if (FE >= 0 && FE != FD) close(FE);

    // Release the buffer if it was allocated.
    //
    if (buff) free(buff);

    // Clear all data values by attaching a dummy FD
    //
    FD = FE = -1;
    buff = 0;

    // Check if we should echo the last line
    //
    if (llBuff && Verbose && Eroute)
       {if (*llBuff && llBok > 1) Eroute->Say(llPrefix, llBuff);
        llBok = 0;
       }
}

/******************************************************************************/
/*                                 D r a i n                                  */
/******************************************************************************/
  
int XrdOucStream::Drain() 
{
    int Status = 0;

    // Drain any outstanding processes (i.e., kill the process group)
    //
#ifndef WIN32
    int retc;
    if (child) {kill(-child, 9);
                do {retc = waitpid(child, &Status, 0);}
                    while(retc > 0 || (retc == -1 && errno == EINTR));
                child = 0;
               }
#else
    if (child) {
       TerminateProcess((HANDLE)child, 0);
       child = 0;
    }
#endif
    return Status;
}
  
/******************************************************************************/
/*                                  E c h o                                   */
/******************************************************************************/
  
void XrdOucStream::Echo()
{
   if (llBok && Verbose && *llBuff && Eroute) Eroute->Say(llPrefix, llBuff);
   llBok = 0;
}

/******************************************************************************/
/*                               E   x   e   c                                */
/******************************************************************************/
  
int XrdOucStream::Exec(const char *theCmd, int inrd, int efd)
{
    int j;
    char *cmd, *origcmd, *parm[MaxARGC];

    // Allocate a buffer for the command as we will be modifying it
    //
    origcmd = cmd = (char *)malloc(strlen(theCmd)+1);
    strcpy(cmd, theCmd);
  
    // Construct the argv array based on passed command line.
    //
    for (j = 0; j < MaxARGC-1 && *cmd; j++)
        {while(*cmd == ' ') cmd++;
         if (!(*cmd)) break;
         parm[j] = cmd;
         while(*cmd && *cmd != ' ') cmd++;
         if (*cmd) {*cmd = '\0'; cmd++;}
        }
    parm[j] = (char *)0;

    // Continue with normal processing
    //
    j = Exec(parm, inrd, efd);
    free(origcmd);
    return j;
}

int XrdOucStream::Exec(char **parm, int inrd, int efd)
{
    int fildes[2], Child_in = -1, Child_out = -1, Child_log = -1;

    // Create a pipe. Minimize file descriptor leaks.
    //
    if (inrd >= 0)
       {if (pipe(fildes))
           return Err(Exec, errno, "create input pipe for", parm[0]);
           else {
                 fcntl(fildes[0], F_SETFD, FD_CLOEXEC);
                 Attach(fildes[0]); Child_out = fildes[1];
                }

        if (inrd)
           {if (pipe(fildes))
               return Err(Exec, errno, "create output pipe for", parm[0]);
               else {
                     fcntl(fildes[1], F_SETFD, FD_CLOEXEC);
                     FE = fildes[1]; Child_in  = fildes[0];
                    }
           }
       } else {Child_out = FD; Child_in = FE;}

    // Handle the standard error file descriptor
    //
    if (!efd) Child_log = (Eroute ? dup(Eroute->logger()->originalFD()) : -1);
       else if (efd > 0) Child_log = efd;

    // Fork a process first so we can pick up the next request. We also
    // set the process group in case the chi;d hasn't been able to do so.
    //
    if ((child = fork()))
       {          close(Child_out);
        if (inrd) close(Child_in );
        if (!efd && Child_log >= 0) close(Child_log);
        if (child < 0)
           return Err(Exec, errno, "fork request process for", parm[0]);
        setpgid(child, child);
        return 0;
       }

    /*****************************************************************/
    /*                  C h i l d   P r o c e s s                    */
    /*****************************************************************/

    // Redirect standard in if so requested
    //
    if (Child_in >= 0)
       {if (inrd)
           {if (dup2(Child_in, STDIN_FILENO) < 0)
               {Erx(Exec, errno, "set up standard in for " <<parm[0]);
                exit(255);
               } else if (Child_in != Child_out) close(Child_in);
           }
       }

    // Reassign the stream to be standard out to capture all of the output.
    //
    if (Child_out >= 0)
       {if (dup2(Child_out, STDOUT_FILENO) < 0)
           {Erx(Exec, errno, "set up standard out for " <<parm[0]);
            exit(255);
           } else close(Child_out);
       }

    // Redirect stderr of the stream if we can to avoid keeping the logfile open
    //
    if (Child_log >= 0)
       {if (dup2(Child_log, STDERR_FILENO) < 0)
           {Erx(Exec, errno, "set up standard err for " <<parm[0]);
            exit(255);
           } else close(Child_log);
       }

    // Set our process group (the parent should have done this by now) then
    // invoke the command never to return
    //
    setpgid(0,0);
    execv(parm[0], parm);
    Erx(Exec, errno, "execute " <<parm[0]);
    exit(255);
}

/******************************************************************************/
/*                               G e t L i n e                                */
/******************************************************************************/
  
char *XrdOucStream::GetLine()
{
   int bcnt, retc;
   char *bp;

// Check if end of message has been reached.
//
   if (flags & XrdOucStream_EOM) return (char *)NULL;

// Find the next record in the buffer
//
   if (bleft > 0)
      {recp = bnext; bcnt = bleft;
       for (bp = bnext; bcnt--; bp++)
           if (!*bp || *bp == '\n')
               {if (!*bp) flags |= XrdOucStream_EOM;
                *bp = '\0';
                bnext = ++bp;
                bleft = bcnt;
                token = recp;
                return recp;
               }
               else if (notabs && *bp == '\t') *bp = ' ';
  
   // There is no next record, so move up data in the buffer.
   //
      strncpy(buff, bnext, bleft);
      bnext = buff + bleft;
      }
      else bnext = buff;

// Prepare to read in more data.
//
    bcnt = bsize - (bnext - buff) -1;
    bp = bnext;

// Read up to the maximum number of bytes. Stop reading should we see a
// new-line character or a null byte -- the end of a record.
//
   recp  = token = buff; // This will always be true at this point
   while(bcnt)
        {do { retc = read(FD, (void *)bp, (size_t)bcnt); }
            while (retc < 0 && errno == EINTR);

         if (retc < 0) {Erp(GetLine,errno,"read request",0); return (char *)0;}
         if (!retc)
            {*bp = '\0';
             flags |= XrdOucStream_EOM;
             bnext = ++bp;
             bleft = 0;
             return buff;
            }

         bcnt -= retc;
         while(retc--)
             if (!*bp || *bp == '\n')
                {if (!*bp) flags |= XrdOucStream_EOM;
                    else *bp = '\0';
                 bnext = ++bp;
                 bleft = retc;
                 return buff;
                } else {
                 if (notabs && *bp == '\t') *bp = ' ';
                 bp++;
                }
         }

// All done, force an end of record.
//
   Erp(GetLine, EMSGSIZE, "read full message", 0);
   buff[bsize-1] = '\0';
   return buff;
}

/******************************************************************************/
/*                              G e t T o k e n                               */
/******************************************************************************/
  
char *XrdOucStream::GetToken(int lowcase) {
     char *tpoint;

     // Verify that we have a token to return;
     //
     if (!token) return (char *)NULL;

     // Skip to the first non-blank character.
     //
     while (*token && *token == ' ') token ++;
     if (!*token) {token = 0; return 0;}
     tpoint = token;

     // Find the end of the token.
     //
     if (lowcase) while (*token && *token != ' ')
                        {*token = (char)tolower((int)*token); token++;}
        else      while (*token && *token != ' ') {token++;}
     if (*token) {*token = '\0'; token++;}

     // All done here.
     //
     return tpoint;
}

char *XrdOucStream::GetToken(char **rest, int lowcase)
{
     char *tpoint;

     // Get the next token
     //
     if (!(tpoint = GetToken(lowcase))) return tpoint;

     // Skip to the first non-blank character.
     //
     while (*token && *token == ' ') token ++;
     if (rest) *rest = token;


     // All done.
     //
     return tpoint;
}

/******************************************************************************/
/*                          G e t F i r s t W o r d                           */
/******************************************************************************/

char *XrdOucStream::GetFirstWord(int lowcase)
{
      // If in the middle of a line, flush to the end of the line. Suppress
      // variable substitution when doing this to avoid errors.
      //
      if (xline) 
         {XrdOucEnv *oldEnv = SetEnv(0);
          while(GetWord(lowcase));
          SetEnv(oldEnv);
         }
      return GetWord(lowcase);
}

/******************************************************************************/
/*                        G e t M y F i r s t W o r d                         */
/******************************************************************************/
  
char *XrdOucStream::GetMyFirstWord(int lowcase)
{
   char *var;
   int   skip2fi = 0;


   if (llBok > 1 && Verbose && *llBuff && Eroute) Eroute->Say(llPrefix,llBuff);
   llBok = 0;

   if (!myInst)
      {if (!myEnv) return add2llB(GetFirstWord(lowcase), 1);
          else {while((var = GetFirstWord(lowcase)) && !isSet(var)) {}
                return add2llB(var, 1);
               }
      }

   do {if (!(var = GetFirstWord(lowcase)))
          {if (sawif)
              {ecode = EINVAL;
               if (Eroute) Eroute->Emsg("Stream", "Missing 'fi' for last 'if'.");
              }
           return add2llB(var, 1);
          }

        if (       !strcmp("if",   var)) var = doif();
        if (var && !strcmp("else", var)) var = doelse();
        if (var && !strcmp("fi",   var))
           {if (sawif) sawif = skpel = skip2fi = 0;
               else {if (Eroute)
                        Eroute->Emsg("Stream", "No preceeding 'if' for 'fi'.");
                     ecode = EINVAL;
                    }
            continue;
           }
        if (var && (!myEnv || !isSet(var))) return add2llB(var, 1);
       } while (1);

   return 0;
}

/******************************************************************************/
/*                               G e t W o r d                                */
/******************************************************************************/
  
char *XrdOucStream::GetWord(int lowcase)
{
     char *wp, *ep;

     // If we have a token, return it
     //
     xline = 1;
     while((wp = GetToken(lowcase)))
          {if (!myEnv) return add2llB(wp);
           if ((wp = vSubs(wp)) && *wp) return add2llB(wp);
          }

     // If no continuation allowed, return a null (but only once)
     //
     if (!xcont) {xcont = 1; xline = 0; return (char *)0;}

     // Find the next non-blank non-comment line
     //
     while(GetLine())
        {// Get the first token (none if it is a blank line)
         //
         if (!(wp = GetToken(lowcase))) continue;

         // If token starts with a pound sign, skip the line
         //
         if (*wp == '#') continue;

         // Process continuations (last non-blank character is a back-slash)
         //
         ep = bnext-2;
         while (ep >= buff && *ep == ' ') ep--;
         if (ep < buff) continue;
         if (*ep == '\\') {xcont = 1; *ep = '\0';}
            else xcont = 0;
         return add2llB((myEnv ? vSubs(wp) : wp));
         }
      xline = 0;
      return (char *)0;
}

/******************************************************************************/
/*                               G e t R e s t                                */
/******************************************************************************/
  
int XrdOucStream::GetRest(char *theBuff, int Blen, int lowcase)
{
   char *tp, *myBuff = theBuff;
   int tlen;

// Get remaining tokens
//
   theBuff[0] = '\0';
   while ((tp = GetWord(lowcase)))
         {tlen = strlen(tp);
          if (tlen+1 >= Blen) return 0;
          if (myBuff != theBuff) {*myBuff++ = ' '; Blen--;}
          strcpy(myBuff, tp);
          Blen -= tlen; myBuff += tlen;
         }

// All done
//
   add2llB(0);
   return 1;
}

/******************************************************************************/
/*                              R e t T o k e n                               */
/******************************************************************************/
  
void XrdOucStream::RetToken()
{
     // Check if we can back up
     //
     if (!token || token == recp) return;

     // Find the null byte for the token and remove it, if possible
     //
     while(*token && token != recp) token--;
     if (token != recp) 
        {if (token+1 != bnext) *token = ' ';
         token--;
         while(*token && *token != ' ' && token != recp) token--;
         if (token != recp) token++;
        }

     // If saving line, we must do the same for the saved line
     //
     if (llBuff)
         while(llBcur != llBuff && *llBcur != ' ') {llBcur--; llBleft++;}
}

/******************************************************************************/
/*                                   P u t                                    */
/******************************************************************************/

int XrdOucStream::Put(const char *data, const int dlen) {
    int dcnt = dlen, retc;

    if (flags & XrdOucStream_BUSY) {ecode = ETXTBSY; return -1;}

    while(dcnt)
         {do { retc = write(FE, (const void *)data, (size_t)dlen);}
              while (retc < 0 && errno == EINTR);
          if (retc >= 0) dcnt -= retc;
             else {flags |= XrdOucStream_BUSY;
                   Erp(Put, errno, "write to stream", 0);
                   flags &= ~XrdOucStream_BUSY;
                   return -1;
                  }
         }
    return 0;
}

int XrdOucStream::Put(const char *datavec[], const int dlenvec[]) {
    int i, retc, dlen;
    const char *data;

    if (flags & XrdOucStream_BUSY) {ecode = ETXTBSY; return -1;}

    for (i = 0; datavec[i]; i++)
        {data = datavec[i]; dlen = dlenvec[i];
         while(dlen)
              {do { retc = write(FE, (const void *)data, (size_t)dlen);}
                   while (retc < 0 && errno == EINTR);
               if (retc >= 0) {data += retc; dlen -= retc;}
                  else {flags |= XrdOucStream_BUSY;
                        Erp(Put, errno, "write to stream",0);
                        flags &= ~XrdOucStream_BUSY;
                        return -1;
                       }
              }
        }
    return 0;
}
 
/******************************************************************************/
/*                             W a i t 4 D a t a                              */
/******************************************************************************/

int XrdOucStream::Wait4Data(int msMax)
{
   struct pollfd polltab = {FD, POLLIN|POLLRDNORM, 0};
   int retc;

// Wait until we can actually read something
//
   do {retc = poll(&polltab, 1, msMax);} while(retc < 0 && errno == EINTR);
   if (retc != 1) return (retc ? errno : -1);

// Return correct value
//
   return (polltab.revents & (POLLIN|POLLRDNORM) ? 0 : EIO);
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               a d d 2 l l B                                */
/******************************************************************************/

char *XrdOucStream::add2llB(char *tok, int reset)
{
   int tlen;

// Return if not saving data
//
   if (!llBuff) return tok;

// Check if we should flush the previous line
//
   if (reset)
      {llBok  = 1;
       llBcur = llBuff;
       llBleft= llBsz;
      *llBuff = '\0';
      } else if (!llBok) return tok;
                else {llBok = 2;
                      if (llBleft >= 2)
                         {*llBcur++ = ' '; *llBcur = '\0'; llBleft--;}
                     }

// Add in the new token
//
   if (tok)
      {tlen = strlen(tok);
       if (tlen < llBsz)
          {strcpy(llBcur, tok); llBcur += tlen; llBleft -= tlen;}
      }
   return tok;
}
/******************************************************************************/
/*                                d o e l s e                                 */
/******************************************************************************/

char *XrdOucStream::doelse()
{
   char *var;

// An else must be preceeded by an if and not by a naked else
//
   if (!sawif || sawif == 2)
      {if (Eroute) Eroute->Emsg("Stream", "No preceeding 'if' for 'else'.");
       ecode = EINVAL;
       return 0;
      }

// If skipping all else caluses, skip all lines until we reach a fi
//
   if (skpel)
      {while((var = GetFirstWord()))
            {if (!strcmp("fi", var)) return var;}
       if (Eroute) Eroute->Emsg("Stream", "Missing 'fi' for last 'if'.");
       ecode = EINVAL;
       return 0;
      }

// Elses are still possible then process one of them
//
   do {if (!(var = GetWord())) // A naked else will always succeed
          {sawif = 2;
           return 0;
          }
       if (strcmp("if", var))  // An else may only be followed by an if
          {Eroute->Emsg("Stream","'else",var,"' is invalid.");
           ecode = EINVAL;
           return 0;
          }
       sawif = 0;
       var = doif();
      } while(var && !strcmp("else", var));
   return var;
}
  
/******************************************************************************/
/*                                  d o i f                                   */
/******************************************************************************/

/* Function: doif

   Purpose:  To parse the directive: if [<hlist>] [exec <pgm>] [named <nlist>]
                                     fi

            <hlist> Apply subsequent directives until the 'fi' if this host
                    is one of the hosts in the blank separated list. Each
                    host name may have a single asterisk somewhere in the
                    name to indicate where arbitrry characters lie.

            <pgm>   Apply subsequent directives if this program is named <pgm>.

            <nlist> Apply subsequent directives if this  host instance name
                    is in the list of blank separated names.

   Notes: 1) At least one of hlist, pgm, or nlist must be specified.
          2) The combination of hlist, pgm, nlist must all be true.

   Output: 0 upon success or !0 upon failure.
*/

char *XrdOucStream::doif()
{
    char *var;
    int rc;

// Check if the previous if was properly closed
//
   if (sawif)
      {if (Eroute) Eroute->Emsg("Stream", "Missing 'fi' for last 'if'.");
       ecode = EINVAL;
      }

// Check if we should continue
//
   sawif = 1; skpel = 0;
   if ((rc = XrdOucUtils::doIf(Eroute,*this,"if directive",myHost,myName,myExec)))
      {if (rc < 0) ecode = EINVAL;
          else skpel = 1;
       return 0;
      }

// Skip all lines until we reach a fi or else
//
   while((var = GetFirstWord()))
        {if (!strcmp("fi",   var)) return var;
         if (!strcmp("else", var)) return var;
        }

// Make sure we have a fi
//
   if (!var) 
      {if (Eroute) Eroute->Emsg("Stream", "Missing 'fi' for last 'if'.");
       ecode = EINVAL;
      }
   return 0;
}

/******************************************************************************/
/*                                 i s S e t                                  */
/******************************************************************************/
  
int XrdOucStream::isSet(char *var)
{
   static const char *Mtxt1[2] = {"setenv", "set"};
   static const char *Mtxt2[2] = {"Setenv variable", "Set variable"};
   static const char *Mtxt3[2] = {"Variable", "Environmental variable"};
   char *tp, *vn, *vp, *pv, Vname[64], ec, Nil = 0;
   int sawEQ, Set = 1;

// Process set var = value | set -v | setenv = value
//
   if (!strcmp("setenv", var)) Set = 0;
      else if (strcmp("set", var)) return 0;

// Now get the operand
//
   if (!(tp = GetToken()))
      return xMsg("Missing variable name after '",Mtxt1[Set],"'.");

// Option flags only apply to set not setenv
//
   if (Set)
  {if (!strcmp(tp, "-q")) {if (llBuff) {free(llBuff); llBuff = 0;}; return 1;}
   if (!strcmp(tp, "-v") || !strcmp(tp, "-V"))
      {if (Eroute)
          {if (!llBuff) llBuff = (char *)malloc(llBsz);
           llBcur = llBuff; llBok = 0; llBleft = llBsz; *llBuff = '\0';
           Verbose = (strcmp(tp, "-V") ? 1 : 2);
          }
       return 1;
      }
  }

// Next may be var= | var | var=val
//
   if ((vp = index(tp, '='))) {sawEQ = 1; *vp = '\0'; vp++;}
      else sawEQ = 0;
   if (strlcpy(Vname, tp, sizeof(Vname)) >= sizeof(Vname))
      return xMsg(Mtxt2[Set],tp,"is too long.");
   if (!Set && !strncmp("XRD", Vname, 3))
      return xMsg("Setenv variable",tp,"may not start with 'XRD'.");

// Verify that variable is only an alphanum
//
   tp = Vname;
   while (*tp && (*tp == '_' || isalnum(*tp))) tp++;
   if (*tp) return xMsg(Mtxt2[Set], Vname, "is non-alphanumeric");

// Now look for the value
//
   if (sawEQ) tp = vp;
      else if (!(tp = GetToken()) || *tp != '=')
              return xMsg("Missing '=' after", Mtxt1[Set], Vname);
              else tp++;
   if (!*tp && !(tp = GetToken())) tp = (char *)"";

// The value may be '$var', in which case we need to get it out of the env if
// this is a set or from our environment if this is a setenv
//
   if (*tp != '$') vp = tp;
      else {pv = tp+1;
                 if (*pv == '(') ec = ')';
            else if (*pv == '{') ec = '}';
            else if (*pv == '[') ec = ']';
            else                 ec = 0;
            if (!ec) vn = tp+1;
               else {while(*pv && *pv != ec) pv++;
                     if (*pv) *pv = '\0';
                        else   ec = 0;
                     vn = tp+2;
                    }
            if (!*vn) {*pv = ec; return xMsg("Variable", tp, "is malformed.");}
            if (!(vp = (Set ? getenv(vn) : myEnv->Get(vn))))
               {if (ec != ']')
                   {xMsg(Mtxt3[Set],vn,"is undefined."); *pv = ec; return 1;}
                vp = &Nil;
               }
            *pv = ec;
           }

// Make sure the value is not too long
//
   if ((int)strlen(vp) > maxVLen)
      return xMsg(Mtxt3[Set], Vname, "value is too long.");

// Set the value
//
   if (Verbose == 2 && Eroute)
      if (!(pv = (Set ? myEnv->Get(Vname) : getenv(Vname))) || strcmp(vp, pv))
         {char vbuff[1024];
          strcpy(vbuff, Mtxt1[Set]); strcat(vbuff, " "); strcat(vbuff, Vname);
          Eroute->Say(vbuff, " = ", vp);
         }
   if (Set) myEnv->Put(Vname, vp);
      else if (!(pv = getenv(Vname)) || strcmp(vp,pv))
              XrdOucEnv::Export(Vname, vp);
   return 1;
}

/******************************************************************************/
/*                                 v S u b s                                  */
/******************************************************************************/
  
char *XrdOucStream::vSubs(char *Var)
{
   char *vp, *sp, *dp, *vnp, ec, bkp, valbuff[maxVLen], Nil = 0;
   int n;

// Check for substitution
//
   if (!Var) return Var;
   sp = Var; dp = valbuff; n = maxVLen-1; *varVal = '\0';

   while(*sp && n > 0)
        {if (*sp == '\\') {*dp++ = *(sp+1); sp +=2; n--; continue;}
         if (*sp != '$'
         || (!isalnum(*(sp+1)) && !index("({[", *(sp+1))))
                {*dp++ = *sp++;         n--; continue;}
         sp++; vnp = sp;
              if (*sp == '(') ec = ')';
         else if (*sp == '{') ec = '}';
         else if (*sp == '[') ec = ']';
         else                 ec = 0;
         if (ec) {sp++; vnp++;}
         while(isalnum(*sp)) sp++;
         if (ec && *sp != ec)
            {xMsg("Variable", vnp-2, "is malformed."); return varVal;}
         bkp = *sp; *sp = '\0';
         if (!(vp = myEnv->Get(vnp)))
            {if (ec != ']') xMsg("Variable", vnp, "is undefined.");
             vp = &Nil;
            }
         while(n && *vp) {*dp++ = *vp++; n--;}
         if (*vp) break;
         if (ec) sp++;
            else *sp = bkp;
        }

   if (*sp) xMsg("Substituted text too long using", Var);
      else {*dp = '\0'; strcpy(varVal, valbuff);}
   return varVal;
}

/******************************************************************************/
/*                                  x M s g                                   */
/******************************************************************************/

int XrdOucStream::xMsg(const char *txt1, const char *txt2, const char *txt3)
{
    if (Eroute) Eroute->Emsg("Stream", txt1, txt2, txt3);
    ecode = EINVAL;
    return 1;
}
