// $Id$

const char *XrdSutAuxCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                        X r d S u t A u x . c c                             */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <netinet/in.h>
#include <time.h>
#include <pwd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <XrdSys/XrdSysLogger.hh>
#include <XrdSys/XrdSysError.hh>
#include <XrdOuc/XrdOucString.hh>

#include <XrdSut/XrdSutAux.hh>
#include <XrdSut/XrdSutTrace.hh>

static const char *gXRSBucketTypes[] = {
   "kXRS_none",
   "kXRS_inactive",
   "kXRS_cryptomod",
   "kXRS_main",
   "kXRS_srv_seal",
   "kXRS_clnt_seal",
   "kXRS_puk",
   "kXRS_cipher",
   "kXRS_rtag",
   "kXRS_signed_rtag",
   "kXRS_user",
   "kXRS_host",
   "kXRS_creds",
   "kXRS_message",
   "kXRS_srvID",
   "kXRS_sessionID",
   "kXRS_version",
   "kXRS_status",
   "kXRS_localstatus",
   "kXRS_othercreds",
   "kXRS_cache_idx",
   "kXRS_clnt_opts",
   "kXRS_error_code",
   "kXRS_timestamp",
   "kXRS_x509",
   "kXRS_issuer_hash",
   "kXRS_x509_req",
   "kXRS_cipher_alg",
   "kXRS_md_alg",
   "kXRS_afsinfo",
   "kXRS_reserved"
};

//
// For error logging and tracing
static XrdSysLogger Logger;
static XrdSysError eDest(0,"sut_");
XrdOucTrace *sutTrace = 0;

/******************************************************************************/
/*  X r d S u t S e t T r a c e                                               */
/******************************************************************************/
//______________________________________________________________________________
void XrdSutSetTrace(kXR_int32 trace)
{
   // Set trace flags according to 'trace'

   //
   // Initiate error logging and tracing
   eDest.logger(&Logger);
   if (!sutTrace)
      sutTrace = new XrdOucTrace(&eDest);
   if (sutTrace) {
      // Set debug mask
      sutTrace->What = 0;
      // Low level only
      if ((trace & sutTRACE_Notify))
         sutTrace->What |= sutTRACE_Notify;
      // Medium level
      if ((trace & sutTRACE_Debug))
         sutTrace->What |= (sutTRACE_Notify | sutTRACE_Debug);
      // High level
      if ((trace & sutTRACE_Dump))
         sutTrace->What |= sutTRACE_ALL;
   }
}

/******************************************************************************/
/*  X r d S u t B u c k S t r                                                 */
/******************************************************************************/
//______________________________________________________________________________
const char *XrdSutBuckStr(int kbck)
{
   // Return bucket string
   static const char *ukn = "Unknown";

   kbck = (kbck < 0) ? 0 : kbck;
   kbck = (kbck > kXRS_reserved) ? 0 : kbck;
   kbck = (kbck >= kXRS_cryptomod) ?  (kbck - kXRS_cryptomod + 2) : kbck;

   if (kbck < 0 || kbck > (kXRS_reserved - kXRS_cryptomod + 2))
      return ukn;  
   else
      return gXRSBucketTypes[kbck];  
}

/******************************************************************************/
/*  X r d S u t M e m S e t                                                   */
/******************************************************************************/
//______________________________________________________________________________
volatile void *XrdSutMemSet(volatile void *dst, int c, int len)
{
   // To avoid problems due to compiler optmization
   // Taken from Viega&Messier, "Secure Programming Cookbook", O'Really, #13.2
   // (see discussion there)
   volatile char *buf;

   for (buf = (volatile char *)dst; len; buf[--len] = c);
   return dst;
}

#ifndef USE_EXTERNAL_GETPASS
/******************************************************************************/
/*  X r d S u t G e t P a s s                                                 */
/******************************************************************************/
//_____________________________________________________________________________
int XrdSutGetPass(const char *prompt, XrdOucString &passwd)
{
   // Get password from command line using getpass
   // *** Use only if you cannot provide a better alternative ***
   // User will be prompted for 'prompt'; the entered password
   // is returned in 'passwd'.
   // Returns 0 if ok, -1 if any error occurs.
   EPNAME("GetPass");

   char *pw = getpass(prompt);
   if (pw) {
      // Get rid of special chars, if any
      int k = 0, i = 0, len = strlen(pw);
      for (; i<len ; i++)
         if (pw[i] > 0x20) pw[k++] = pw[i];
      pw[k] = 0;
      passwd = pw;
      XrdSutMemSet((volatile void *)pw,0,len);
   } else {
      DEBUG("error from getpass");
      return -1;
   }
   return 0;
}
#endif

/******************************************************************************/
/*  X r d S u t G e t L i n e                                                 */
/******************************************************************************/
int XrdSutGetLine(XrdOucString &line, const char *prompt)
{
   // Get line from main input stream.
   // Prompt 'prompt' if this is defined.
   // Returns number of chars entered.
   // NB: at most XrdSutMAXBUF-1 chars will be accepted
   char bin[XrdSutMAXBUF] = {0};
   
   // Print prompt, if requested
   if (prompt)
      cout << prompt;

   // Get line
   cin.getline(bin,XrdSutMAXBUF-1);

   // Fill input
   line = bin;

   return line.length();
}

/******************************************************************************/
/*  X r d S u t A s k C o n f i r m                                           */
/******************************************************************************/
bool XrdSutAskConfirm(const char *msg1, bool defact, const char *msg2)
{
   // Prompt for confirmation of action
   // If defined, msg1 is printed as prompt, followed by the default action
   // (  [y] == do-act, for defact = true; 
   //    [n] == do-not-act, for defact = false)
   // If defined, msg2 is printed before prompting.

   bool rc = defact;

   if (msg2)
      cout << msg2;
   XrdOucString ask;
   XrdOucString prompt = defact ? " [y]: " : " [n]: ";
   if (msg1)
      prompt.insert(msg1,0);
   XrdSutGetLine(ask,prompt.c_str());
   ask.lower(0);
   if (ask.length()) {
      if (defact && (ask == 'n' || ask == "no")) {
         rc = 0;
      } else if (!defact && (ask == 'y' || ask == "yes")) {
         rc = 1;
      }
   }
   // we are done
   return rc;
}

/******************************************************************************/
/*  X r d S u t T o H e x                                                     */
/******************************************************************************/
int XrdSutToHex(const char *in, int lin, char *out)
{
   // Content of lin bytes at in are transformed into an hexadecimal,
   // null-terminated, string of length 2*lin; the result is returned
   // in the buffer pointed by out, which must be allocated by the caller
   // to contain at least 2*lin+1 bytes. 
   // Return 0 in case of success, -1 in case of error (errno set to EINVAL if
   // any of in or out are not defined).

   if (!in || !out) {
      errno = EINVAL;
      return -1;
   }

   int lbuf = 2*lin+1;
   int i = 0;
   out[0] = 0;
   for ( ; i < lin; i++)
      sprintf(out,"%s%02x",out,(0xFF & in[i]));
   // Null termination
   out[lbuf-1] = 0;

   // ok
   return 0;
}

/******************************************************************************/
/*  X r d S u t F r o m H e x                                                 */
/******************************************************************************/
int XrdSutFromHex(const char *in, char *out, int &lout)
{
   // Content of the hexadecimal, null-terminated, string at in, is
   // transformed into lout bytes returned in out.
   // The output buffer should be allocated by the caller to contain
   // at least lin/2 bytes if lin=strlen(in) is even, and lin/2+1 bytes
   // if lin is odd (in this case an additional char equal 0 is appended
   // to in).
   // Return 0 in case of success, -1 in case of error (errno set to EINVAL if
   // any of in or out are not defined).

   lout = 0;
   if (!in || !out) {
      errno = EINVAL;
      return -1;
   }

   int lin = strlen(in);
   char st[3] = {0};
   int i = 0, k = 0;
   for ( ; i<lin; i += 2) {
      st[0] = in[i];
      st[1] = ((i+1) < lin) ? in[i+1] : 0;
      int c;
      sscanf(st,"%x",&c);
      out[k++] = (char)(0x000000FF & c);
   }

   lout = k;

   return 0;
}

/******************************************************************************/
/*  X r d S u t T i m e S t r i n g                                           */
/*                                                                            */
/******************************************************************************/
int XrdSutTimeString(int t, char *st, int opt)
{
   //  Trasform a time in secs since 1Jan1970 in a string of the format
   //     24Apr2006:09:10:23       (opt = 0, default)
   //     24Apr2006-091023         (opt = 1)
   // The buffer st must be supplied by the caller to contain at least 20.
   // This length is returned when calling the function with t=-1 
   static char month[12][4] = {"Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"};
   static short flen = strlen("24Apr2006:09:10:23");

   // Check if the length is required
   if (t == -1)
      return (flen+1);

   // Now check inputs
   if (t < 0 || !st)
      return -1;

   // Get the breakdown
   struct tm tst;
   time_t ttmp = t;
   if (!localtime_r(&ttmp,&tst))
      return -2;

   // Now fill the output
   if (opt == 1) {
      sprintf(st,"%2d%3s%4d-%2d%2d%2d",tst.tm_mday,month[tst.tm_mon],
                                    1900+tst.tm_year,
                                    tst.tm_hour,tst.tm_min,tst.tm_sec);
      // Make sure is null terminated at the right point
      st[flen-2] = '\0';
   } else {
      sprintf(st,"%2d%3s%4d:%2d:%2d:%2d",tst.tm_mday,month[tst.tm_mon],
                                         1900+tst.tm_year,
                                         tst.tm_hour,tst.tm_min,tst.tm_sec);
   }

   // Make sure there are no empty spaces
   if (st[0] == 0x20) st[0] = 0x30;
   int i = 10;
   for (; i <= 16; i++ )
       if (st[i] == 0x20) st[i] = 0x30;


   // Null termination
   st[flen] = 0;

   // Ok
   return 0;
}

/******************************************************************************/
/*  X r d S u t E x p a n d                                                   */
/******************************************************************************/
int XrdSutExpand(XrdOucString &path)
{
   // Expand '~' or $PWD for incomplete absolute path specification
   // Returns 0 in case of success, -EINVAL if path is not defined;
   // -errno if failure of the pwnam functions; -ENOENT if PWD is not
   // defined
   EPNAME("Expand");

   // Path must be defined
   if (!path.length())
      return -EINVAL;

   // If path is absolute, do nothing
   if (path[0] == '/')
      return 0;

   if (path[0] == '~') {
      XrdOucString unam, home;
      XrdOucString sdir(path);
      int iu = path.find('/');
      if (iu != STR_NPOS) {
         if (iu > 1)
            unam.assign(path, 1, iu-1);
         sdir.erase(0, iu);
      } else
         sdir = '/';
      if (unam.length() > 0) {
         struct passwd *pw = 0;
         if (!(pw = getpwnam(unam.c_str()))) {
            DEBUG("cannot pwnam information for local user "<<
                 ((unam.length() > 0) ? unam : XrdOucString("")));
            return -errno;
         }
         home = pw->pw_dir;
      } else
         home = XrdSutHome();
      if (home.length() > 0) {
         sdir.insert(home.c_str(),0);
         path = sdir;
      }
   } else {
      // relative path, add local dir
      char *pwd = getenv("PWD");
      if (pwd) {
         path.insert('/',0);
         path.insert(pwd,0);
         path.erase("//");
      } else {
         DEBUG("PWD undefined ");
         return -ENOENT;
      }
   }
   return 0;
}

/******************************************************************************/
/*  X r d S u t R e s o l v e                                                 */
/******************************************************************************/
int XrdSutResolve(XrdOucString &path,
                  const char *ho, const char *vo, const char *gr, const char *us)
{
   // Resolve templates <host>, <vo>, <group>, <user> (if any)
   // Returns 0 in case of success, -EINVAL if path is not defined.

   // Path must be defined
   if (!path.length())
      return -EINVAL;

   // No templates, nothing to do
   if (path.find("<") == STR_NPOS)
      return 0;

   // Replace <host>, if defined
   if (ho && strlen(ho) > 0) path.replace("<host>", ho);

   // Replace <vo>, if defined
   if (vo && strlen(vo) > 0) path.replace("<vo>", vo);

   // Replace <group>, if defined
   if (gr && strlen(gr) > 0) path.replace("<group>", gr);

   // Replace <user>, if defined
   if (us && strlen(us) > 0) path.replace("<user>", us);

   // Done
   return 0;
}

/******************************************************************************/
/*  X r d S u t H o m e                                                       */
/******************************************************************************/
const char *XrdSutHome()
{
   // Gets the home directory preferentially from HOME or from getpwuid()
   EPNAME("Home");

   // Use the save value, if any
   static XrdOucString homedir;
   if (homedir.length() <= 0) {
      // Check the HOME environment variable
      if (getenv("HOME"))
         homedir = getenv("HOME");
      if (homedir.length() <= 0) {
         struct passwd *pw = getpwuid(getuid());
         homedir = pw->pw_dir;
      }
      if (homedir.length() <= 0)
         DEBUG("Warning: home directory undefined! ");
   }

   // Done
   return homedir.c_str();
}

/******************************************************************************/
/*  X r d S u t M k d i r                                                     */
/*                                                                            */
/******************************************************************************/
int XrdSutMkdir(const char *dir, unsigned int mode, const char *opt)
{
   //  Make directory dir
   //  mode specifies permissions
   //  opt == "-p" : make parent directories as needed

   if (!dir) {
      errno = EINVAL;
      return -1;
   }

   if (!strncmp(opt,"-p",2)) {
      //
      //  make also parent directories, if needed
      XrdOucString dd(dir);
      XrdSutExpand(dd);
      if (dd[dd.length()-1] != '/')
         dd.append('/');
      int lsl = dd.find('/',1);
      while (lsl > -1) {
         XrdOucString pd(dd,0,lsl-1);
         struct stat st;
         if (stat(pd.c_str(),&st) == -1) {
            if (errno == ENOENT) {
               // path does not exists: create it
               if (mkdir(pd.c_str(),mode) != 0)
                  return -1;
            } else {
               return -1;
            }
         }
         // Go to next
         lsl = dd.find('/',lsl+1);
      }      

   } else {
      return mkdir(dir,mode);
   }

   return 0;
}

/******************************************************************************/
/*  X r d S u t P a r s e T i m e                                             */
/*                                                                            */
/******************************************************************************/
//______________________________________________________________________  
int XrdSutParseTime(const char *tstr, int opt)
{
   // Parse time string of the form "<val1><unit1>:<val2><unit2>:..."
   // with <val> any integer and <unit> one of the following chars:
   //       'y'     for years
   //       'd'     for days
   //       'h'     for hours
   //       'm'     for minutes
   //       's'     for seconds
   // (e.g. "34d:10h:20s")
   // If opt == 1, assume a string in the form "<hh>[:<ss>[:<mm>]]"
   // (e.g. "12:24:35" for 12 hours, 24 minutes and 35 secs)
   // Return the corresponding number of seconds
   EPNAME("ParseTime");

   XrdOucString ts = tstr;
   XrdOucString fr = "";
   int i = 0;
   int tsec = 0;
   // Parse list
   if (ts.length()) {
      int ls = 0;
      int ld = ts.find(':',1);
      ld = (ld == -1) ? ts.length() - 1 : ld;
      while (ld >= ls) {
         fr.assign(ts, ls, ld);
         fr.erase(":");
         // Check this fraction
         if (opt == 0) {
            if (fr.length() > 1) {
               // The unit must be known
               char u = fr[fr.length()-1];
               fr.erase(fr.length()-1);
               if (u == 'y') {
                  tsec += atoi(fr.c_str())*31536000;
               } else if (u == 'd') {
                  tsec += atoi(fr.c_str())*86400;
               } else if (u == 'h') {
                  tsec += atoi(fr.c_str())*3600;
               } else if (u == 'm') {
                  tsec += atoi(fr.c_str())*60;
               } else if (u == 's') {
                  tsec += atoi(fr.c_str());
               } else {
                  DEBUG("unknown unit: "<<u);
               }
            } else {
               DEBUG("Incomplete fraction: "<<fr.c_str());
            }
         } else {
            if (i == 0) {
               tsec += atoi(fr.c_str())*3600;
            } else if (i == 1) {
               tsec += atoi(fr.c_str())*60;
            } else if (i == 2) {
               tsec += atoi(fr.c_str());
            }
         }
         i++;
         ls = ld + 1;
         ld = ts.find(':',ls);
         ld = (ld == -1) ? ts.length() - 1 : ld;
      }
   }
   return tsec;
}

/******************************************************************************/
/*  X r d S u t F i l e L o c k e r                                           */
/*                                                                            */
/*  Guard class for file locking                                              */
/*  Usage:                                                                    */
/*  {                                                                         */
/*     XrdSutFileLocker fl(filename,1);                                       */
/*     // File exclusively locked                                             */
/*     ...                                                                    */
/*  } // Unlocks file 'filename'                                              */
/*        's'     for seconds                                                 */
/*                                                                            */
/******************************************************************************/
//______________________________________________________________________________
XrdSutFileLocker::XrdSutFileLocker(int fd, ELockType lock)
{
   // Constructor: locks the file in 'lock' mode.
   // Use IsValid() to test success.

   valid = 0;
   fdesk = fd;

   // Exclusive lock of the whole file
   int lockmode = (lock == XrdSutFileLocker::kExcl) ? (F_WRLCK | F_RDLCK)
                                                    :  F_RDLCK;
#ifdef __macos__
   struct flock flck = {0, 0, 0, lockmode, SEEK_SET};
#else
   struct flock flck = {lockmode, SEEK_SET, 0, 0};
#endif
   if (fcntl(fdesk, F_SETLK, &flck) != 0)
      // Failure
      return;

   // Success
   valid = 1;
}
//______________________________________________________________________________
XrdSutFileLocker::~XrdSutFileLocker()
{
   // Destructor: unlocks the file if locked.

   if (fdesk < 0 || !IsValid())
      return;
   //
   // Unlock the file
#ifdef __macos__
   struct flock flck = {0, 0, 0, F_UNLCK, SEEK_SET};
#else
   struct flock flck = {F_UNLCK, SEEK_SET, 0, 0};
#endif
   fcntl(fdesk, F_SETLK, &flck);
}

