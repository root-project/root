// $Id$
#ifndef __SUT_AUX_H__
#define __SUT_AUX_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d S u t A u x . h h                             */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#ifndef WIN32
#include "XrdSys/XrdSysHeaders.hh"
#endif
#ifndef __XPROTOCOL_H
#include <XProtocol/XProtocol.hh>
#endif

class XrdCryptoFactory;

class XrdOucString;
class XrdSutBucket;
class XrdSutBuffer;

/******************************************************************************/
/*  U t i l i t y   D e f i n i t i o n s                                     */
/******************************************************************************/

#define XrdSutMAXBUF      4096
#define XrdSutMAXPPT      512
#define XrdSutMAXBUCKS    10
#define XrdSutMAXINT64LEN 25
#define XrdSutPRINTLEN    100

enum kXRSBucketTypes {
   kXRS_none     =     0,      // end-of-vector
   kXRS_inactive =     1,      // inactive (dropped at serialization)
   kXRS_cryptomod = 3000,      // 3000    Name of crypto module to use
   kXRS_main,                  // 3001    Main buffer
   kXRS_srv_seal,              // 3002    Server secrets sent back as they are
   kXRS_clnt_seal,             // 3003    Client secrets sent back as they are
   kXRS_puk,                   // 3004    Public Key
   kXRS_cipher,                // 3005    Cipher
   kXRS_rtag,                  // 3006    Random Tag
   kXRS_signed_rtag,           // 3007    Random Tag signed by the client
   kXRS_user,                  // 3008    User name
   kXRS_host,                  // 3009    Remote Host name
   kXRS_creds,                 // 3010    Credentials (password, ...)
   kXRS_message,               // 3011    Message (null-terminated string)
   kXRS_srvID,                 // 3012    Server unique ID
   kXRS_sessionID,             // 3013    Handshake session ID
   kXRS_version,               // 3014    Package version 
   kXRS_status,                // 3015    Status code  
   kXRS_localstatus,           // 3016    Status code(s) saved in sealed buffer  
   kXRS_othercreds,            // 3017    Alternative creds (e.g. other crypto)  
   kXRS_cache_idx,             // 3018    Cache entry index  
   kXRS_clnt_opts,             // 3019    Client options, if any  
   kXRS_error_code,            // 3020    Error code
   kXRS_timestamp,             // 3021    Time stamp
   kXRS_x509,                  // 3022    X509 certificate
   kXRS_issuer_hash,           // 3023    Issuer hash
   kXRS_x509_req,              // 3024    X509 certificate request
   kXRS_cipher_alg,            // 3025    Cipher algorithm (list)
   kXRS_md_alg,                // 3026    MD algorithm (list)
   kXRS_afsinfo,               // 3027    AFS information
   kXRS_reserved               //         Reserved
};

/******************************************************************************/
/*  X r d S u t B u c k S t r                                                 */
/*  Return bucket string                                                      */
/******************************************************************************/
const char *XrdSutBuckStr(int kbck);

/******************************************************************************/
/*          E r r o r   L o g g i n g / T r a c i n g   F l a g s             */
/******************************************************************************/
#define sutTRACE_ALL       0x0007
#define sutTRACE_Dump      0x0004
#define sutTRACE_Debug     0x0002
#define sutTRACE_Notify    0x0001

/******************************************************************************/
/*  U t i l i t y   F u n c t i o n s                                         */
/******************************************************************************/

/******************************************************************************/
/*  X r d S u t S e t T r a c e                                               */
/*                                                                            */
/*  Set trace flags according to 'trace'                                      */
/*                                                                            */
/******************************************************************************/
//______________________________________________________________________________
void XrdSutSetTrace(kXR_int32 trace);

/******************************************************************************/
/*  X r d S u t M e m S e t                                                   */
/*                                                                            */
/*  Memory setter avoiding problems from compiler optmization                 */
/*  Taken from Viega&Messier, "Secure Programming Cookbook", O'Really, #13.2  */
/*                                                                            */
/******************************************************************************/
volatile void *XrdSutMemSet(volatile void *dst, int c, int len);

/******************************************************************************/
/*  X r d S u t G e t P a s s                                                 */
/*                                                                            */
/*  Getter for secret input: can be user defined                              */
/*                                                                            */
/******************************************************************************/
#ifdef USE_EXTERNAL_GETPASS
extern int XrdSutGetPass(const char *prompt, XrdOucString &passwd);
#else
int XrdSutGetPass(const char *prompt, XrdOucString &passwd);
#endif

/******************************************************************************/
/*  X r d S u t G e t L i n e                                                 */
/*                                                                            */
/*  Get line from main input stream                                           */
/*                                                                            */
/******************************************************************************/
int XrdSutGetLine(XrdOucString &line, const char *prompt = 0);

/******************************************************************************/
/*  X r d S u t A s k C o n f i r m                                           */
/*                                                                            */
/*  Ask confirmation to main input stream                                     */
/*                                                                            */
/******************************************************************************/
bool XrdSutAskConfirm(const char *msg1, bool defact, const char *msg2 = 0);

/******************************************************************************/
/*  X r d S u t T o H e x                                                     */
/*                                                                            */
/*  Transform a buffer in an hexadecimal string                               */
/*                                                                            */
/******************************************************************************/
int XrdSutToHex(const char *in, int lin, char *out);

/******************************************************************************/
/*  X r d S u t F r o m H e x                                                 */
/*                                                                            */
/*  Extract buffer from an hexadecimal string                                 */
/*                                                                            */
/******************************************************************************/
int XrdSutFromHex(const char *in, char *out, int &lout);

/******************************************************************************/
/*  X r d S u t T i m e S t r i n g                                           */
/*                                                                            */
/*  Trasform a time in secs since 1Jan1970 in a string of the format          */
/*     24Apr2006:09:10:23                                                     */
/*  The buffer st must be supplied by the caller to contain at least 20 bytes.*/
/*  This length is returned when calling the function with t=-1.              */ 
/*                                                                            */
/******************************************************************************/
int XrdSutTimeString(int t, char *st, int opt = 0);

/******************************************************************************/
/*  X r d S u t E x p a n d                                                   */
/*                                                                            */
/*  Expand '~' or $PWD for relative paths                                     */
/******************************************************************************/
int XrdSutExpand(XrdOucString &path);

/******************************************************************************/
/*  X r d S u t R e s o l v e                                                 */
/*                                                                            */
/*  Resolve templates <host>, <vo>, <group>, <user> (if any)                  */
/******************************************************************************/
int XrdSutResolve(XrdOucString &path,
                  const char *ho, const char *vo, const char *gr, const char *us);

/******************************************************************************/
/*  X r d S u t H o m e                                                       */
/*                                                                            */
/*  Return the home directory                                                 */
/*  Checks, in the order, HOME and getpwuid()                                 */
/******************************************************************************/
const char *XrdSutHome();

/******************************************************************************/
/*  X r d S u t M k d i r                                                     */
/*                                                                            */
/*  Make directory dir                                                        */
/******************************************************************************/
int XrdSutMkdir(const char *dir, unsigned int mode = 0777,
                                 const char *opt = "-p");
/******************************************************************************/
/*  X r d S u t P a r s e T i m e                                             */
/*                                                                            */
/*  Parse time string of the form "<val1><unit1>:<val2><unit2>:..."           */
/*  with <val> any integer and <unit> one of the following chars:             */
/*        'y'     for years                                                   */
/*        'd'     for days                                                    */
/*        'h'     for hours                                                   */
/*        'm'     for minutes                                                 */
/*        's'     for seconds                                                 */
/*  (e.g. "34d:10h:20s")                                                      */
/*  If opt == 1, assume a string in the form "<hh>[:<ss>[:<mm>]]"             */
/*  (e.g. "12:24:35" for 12 hours, 24 minutes and 35 secs)                    */
/*  Return the corresponding number of seconds                                */
/******************************************************************************/
int XrdSutParseTime(const char *tstr, int opt = 0);

/******************************************************************************/
/*  X r d S u t F i l e L o c k e r                                           */
/*                                                                            */
/*  Guard class for file locking                                              */
/*  Usage:                                                                    */
/*  {                                                                         */
/*     XrdSutFileLocker fl(fd,XrdSutFileLocker::kExcl);                       */
/*     // File exclusively locked                                             */
/*     ...                                                                    */
/*  } // Unlocks file descriptor 'fd'                                         */
/*                                                                            */
/******************************************************************************/
class XrdSutFileLocker {
private:
   int  fdesk;
   bool valid;
public:
   enum ELockType { kShared = 0, kExcl = 1 };
   XrdSutFileLocker(int fd, ELockType lock);
   ~XrdSutFileLocker();
   bool IsValid() const { return valid; }
};

#endif

