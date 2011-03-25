#ifndef __YPROTOCOL_H
#define __YPROTOCOL_H

#ifdef __CINT__
#define __attribute__(x)
#endif

//        $Id$

#include "XProtocol/XPtypes.hh"

// We need to pack structures sent all over the net!
// __attribute__((packed)) assures no padding bytes.
//
// Note all binary values shall be in network byte order.
//
// Data is serialized as explained in XrdOucPup.
  
/******************************************************************************/
/*                C o m m o n   R e q u e s t   S e c t i o n                 */
/******************************************************************************/

namespace XrdCms
{

static const char kYR_Version = 2;

struct CmsRRHdr
{  kXR_unt32  streamid;    // Essentially opaque
   kXR_char   rrCode;      // Request or Response code
   kXR_char   modifier;    // RR dependent
   kXR_unt16  datalen;
};
  
enum CmsReqCode            // Request Codes
{    kYR_login   =  0,     // Same as kYR_data
     kYR_chmod   =  1,
     kYR_locate  =  2,
     kYR_mkdir   =  3,
     kYR_mkpath  =  4,
     kYR_mv      =  5,
     kYR_prepadd =  6,
     kYR_prepdel =  7,
     kYR_rm      =  8,
     kYR_rmdir   =  9,
     kYR_select  = 10,
     kYR_stats   = 11,
     kYR_avail   = 12,
     kYR_disc    = 13,
     kYR_gone    = 14,
     kYR_have    = 15,
     kYR_load    = 16,
     kYR_ping    = 17,
     kYR_pong    = 18,
     kYR_space   = 19,
     kYR_state   = 20,
     kYR_statfs  = 21,
     kYR_status  = 22,
     kYR_trunc   = 23,
     kYR_try     = 24,
     kYR_update  = 25,
     kYR_usage   = 26,
     kYR_xauth   = 27,
     kYR_MaxReq            // Count of request numbers (highest + 1)
};

// The hopcount is used for forwarded requests. It is incremented upon each
// forwarding until it wraps to zero. At this point the forward is not done.
// Forwarding applies to: chmod, have, mkdir, mkpath, mv, prepdel, rm, and 
// rmdir. Any other modifiers must be encoded in the low order 6 bits.
//
enum CmsFwdModifier
{    kYR_hopcount = 0xc0,
     kYR_hopincr  = 0x40
};

enum CmsReqModifier
{    kYR_raw = 0x20,     // Modifier: Unmarshalled data
     kYR_dnf = 0x10      // Modifier: mv, rm, rmdir (do not forward)
};

/******************************************************************************/
/*               C o m m o n   R e s p o n s e   S e c t i o n                */
/******************************************************************************/
  
enum CmsRspCode            // Response codes
{    kYR_data    = 0,      // Same as kYR_login
     kYR_error   = 1,
     kYR_redirect= 2,
     kYR_wait    = 3,
     kYR_waitresp= 4,
     kYR_yauth   = 5
};

enum YErrorCode
{  kYR_ENOENT = 1,
   kYR_EPERM,
   kYR_EACCES,
   kYR_EINVAL,
   kYR_EIO,
   kYR_ENOMEM,
   kYR_ENOSPC,
   kYR_ENAMETOOLONG,
   kYR_ENETUNREACH,
   kYR_ENOTBLK,
   kYR_EISDIR
};

struct CmsResponse
{      CmsRRHdr      Hdr;

enum  {kYR_async   = 128                 // Modifier: Reply to prev waitresp
      };

       kXR_unt32     Val;                // Port, Wait val, rc, asyncid
//     kXR_char      Data[Hdr.datalen-4];// Target host, more data, or emessage
};

/******************************************************************************/
/*                         a v a i l   R e q u e s t                          */
/******************************************************************************/
  
// Request: avail <diskFree> <diskUtil>
// Respond: n/a
//
struct CmsAvailRequest
{      CmsRRHdr      Hdr;
//     kXR_int32     diskFree;
//     kXR_int32     diskUtil;
};

/******************************************************************************/
/*                         c h m o d   R e q u e s t                          */
/******************************************************************************/
  
// Request: chmod <ident> <mode> <path>
// Respond: n/a
//
struct CmsChmodRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_string    Mode;
//     kXR_string    Path;
};

/******************************************************************************/
/*                          d i s c   R e q u e s t                           */
/******************************************************************************/
  
// Request: disc
// Respond: n/a
//
struct CmsDiscRequest
{      CmsRRHdr      Hdr;
};

/******************************************************************************/
/*                          g o n e   R e q u e s t                           */
/******************************************************************************/
  
// Request: gone <path>
// Respond: n/a
//
struct CmsGoneRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Path;
};

/******************************************************************************/
/*                          h a v e   R e q u e s t                           */
/******************************************************************************/
  
// Request: have <path>
// Respond: n/a
//
struct CmsHaveRequest
{      CmsRRHdr      Hdr;
       enum          {Online = 1, Pending = 2};  // Modifiers
//     kXR_string    Path;
};

/******************************************************************************/
/*                        l o c a t e   R e q u e s t                         */
/******************************************************************************/

struct CmsLocateRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_unt32     Opts;

enum  {kYR_refresh = 0x01,
       kYR_asap    = 0x80
      };
//     kXR_string    Path;

static const int     RILen = 32;  // Max length of each response item
};

/******************************************************************************/
/*                         l o g i n   R e q u e s t                          */
/******************************************************************************/
  
// Request: login  <login_data>
// Respond: xauth  <auth_data>
//          login  <login_data>
//

struct CmsLoginData
{      kXR_unt16  Size;              // Temp area for packing purposes
       kXR_unt16  Version;
       kXR_unt32  Mode;              // From LoginMode
       kXR_int32  HoldTime;          // Hold time in ms(managers)
       kXR_unt32  tSpace;            // Tot  Space  GB (servers)
       kXR_unt32  fSpace;            // Free Space  MB (servers)
       kXR_unt32  mSpace;            // Minf Space  MB (servers)
       kXR_unt16  fsNum;             // File Systems   (servers /supervisors)
       kXR_unt16  fsUtil;            // FS Utilization (servers /supervisors)
       kXR_unt16  dPort;             // Data port      (servers /supervisors)
       kXR_unt16  sPort;             // Subs port      (managers/supervisors)
       kXR_char  *SID;               // Server ID      (servers/ supervisors)
       kXR_char  *Paths;             // Exported paths (servers/ supervisors)

       enum       LoginMode
                 {kYR_director=   0x00000001,
                  kYR_manager =   0x00000002,
                  kYR_peer    =   0x00000004,
                  kYR_server  =   0x00000008,
                  kYR_proxy   =   0x00000010,
                  kYR_suspend =   0x00000100,   // Suspended login
                  kYR_nostage =   0x00000200,   // Staging unavailable
                  kYR_trying  =   0x00000400,   // Extensive login retries
                  kYR_debug   =   0x80000000
                 };
};

struct CmsLoginRequest
{  CmsRRHdr     Hdr;
   CmsLoginData Data;
};

struct CmsLoginResponse
{  CmsRRHdr     Hdr;
   CmsLoginData Data;
};

/******************************************************************************/
/*                          l o a d   R e q u e s t                           */
/******************************************************************************/
  
// Request: load <cpu> <io> <load> <mem> <pag> <util> <dskfree>
// Respond: n/a
//
struct CmsLoadRequest
{      CmsRRHdr      Hdr;
       enum         {cpuLoad=0, netLoad, xeqLoad, memLoad, pagLoad, dskLoad,
                     numLoad};
//     kXR_char      theLoad[numload];
//     kXR_int       dskFree;
};

/******************************************************************************/
/*                         m k d i r   R e q u e s t                          */
/******************************************************************************/
  
// Request: mkdir <ident> <mode> <path>
// Respond: n/a
//
struct CmsMkdirRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_string    Mode;
//     kXR_string    Path;
};

/******************************************************************************/
/*                        m k p a t h   R e q u e s t                         */
/******************************************************************************/
  
// Request: <id> mkpath <mode> <path>
// Respond: n/a
//
struct CmsMkpathRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_string    Mode;
//     kXR_string    Path;
};

/******************************************************************************/
/*                            m v   R e q u e s t                             */
/******************************************************************************/
  
// Request: <id> mv <old_name> <new_name>
// Respond: n/a
//
struct CmsMvRequest {
       CmsRRHdr      Hdr;      // Subject to kYR_dnf modifier!
//     kXR_string    Ident;
//     kXR_string    Old_Path;
//     kXR_string    New_Path;
};

/******************************************************************************/
/*                          p i n g   R e q u e s t                           */
/******************************************************************************/
  
// Request: ping
// Respond: n/a
//
struct CmsPingRequest {
       CmsRRHdr      Hdr;
};

/******************************************************************************/
/*                          p o n g   R e q u e s t                           */
/******************************************************************************/
  
// Request: pong
// Respond: n/a
//
struct CmsPongRequest {
       CmsRRHdr      Hdr;
};

/******************************************************************************/
/*                       p r e p a d d   R e q u e s t                        */
/******************************************************************************/
  
// Request: <id> prepadd <reqid> <usr> <prty> <mode> <path>\n
// Respond: No response.
//
struct CmsPrepAddRequest
{      CmsRRHdr      Hdr;    // Modifier used with following options

enum  {kYR_stage   = 0x0001, // Stage   the data
       kYR_write   = 0x0002, // Prepare for writing
       kYR_coloc   = 0x0004, // Prepare for co-location
       kYR_fresh   = 0x0008, // Prepare by  time refresh
       kYR_metaman = 0x0010  // Prepare via meta-manager
      };
//     kXR_string    Ident;
//     kXR_string    reqid;
//     kXR_string    user;
//     kXR_string    prty;
//     kXR_string    mode;
//     kXR_string    Path;
//     kXR_string    Opaque; // Optional
};

/******************************************************************************/
/*                       p r e p d e l   R e q u e s t                        */
/******************************************************************************/
  
// Request: <id> prepdel <reqid>
// Respond: No response.
//
struct CmsPrepDelRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_string    reqid;
};

/******************************************************************************/
/*                            r m   R e q u e s t                             */
/******************************************************************************/
  
// Request: <id> rm <path>
// Respond: n/a
//
struct CmsRmRequest
{      CmsRRHdr      Hdr;    // Subject to kYR_dnf modifier!
//     kXR_string    Ident;
//     kXR_string    Path;
};

/******************************************************************************/
/*                         r m d i r   R e q u e s t                          */
/******************************************************************************/
  
// Request: <id> rmdir <path>
// Respond: n/a
//
struct CmsRmdirRequest
{      CmsRRHdr      Hdr;    // Subject to kYR_dnf modifier!
//     kXR_string    Ident;
//     kXR_string    Path;
};

/******************************************************************************/
/*                        s e l e c t   R e q u e s t                         */
/******************************************************************************/
  
// Request: <id> select[s] {c | d | m | r | w | s | t | x} <path> [-host]

// Note: selects - requests a cache refresh for <path>
// kYR_refresh   - refresh file location cache
// kYR_create  c - file will be created
// kYR_delete  d - file will be created or truncated
// kYR_metaop  m - inod will only be modified
// kYR_read    r - file will only be read
// kYR_replica   - file will replicated
// kYR_write   w - file will be read and writen
// kYR_stats   s - only stat information will be obtained
// kYR_online  x - consider only online files
//                 may be combined with kYR_stats (file must be resident)
//             - - the host failed to deliver the file.


struct CmsSelectRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_unt32     Opts;

enum  {kYR_refresh = 0x0001,
       kYR_create  = 0x0002, // May combine with trunc -> delete
       kYR_online  = 0x0004,
       kYR_read    = 0x0008, // Default
       kYR_trunc   = 0x0010, // -> write
       kYR_write   = 0x0020,
       kYR_stat    = 0x0040, // Exclsuive
       kYR_metaop  = 0x0080,
       kYR_replica = 0x0100  // Only in combination with create
      };
//     kXR_string    Path;
//     kXR_string    Opaque; // Optional
//     kXR_string    Host;   // Optional
};

/******************************************************************************/
/*                         s p a c e   R e q u e s t                          */
/******************************************************************************/
  
// Request: space
//

struct CmsSpaceRequest
{      CmsRRHdr      Hdr;
};
  
/******************************************************************************/
/*                         s t a t e   R e q u e s t                          */
/******************************************************************************/
  
// Request: state <path>
//

struct CmsStateRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Path;

enum  {kYR_refresh = 0x01,   // Modifier
       kYR_noresp  = 0x02,
       kYR_metaman = 0x08
      };
};
  
/******************************************************************************/
/*                        s t a t f s   R e q u e s t                         */
/******************************************************************************/
  
// Request: statfs <path>
//

struct CmsStatfsRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Path;
};

/******************************************************************************/
/*                         s t a t s   R e q u e s t                          */
/******************************************************************************/
  
// Request: stats or statsz (determined by modifier)
//

struct CmsStatsRequest
{      CmsRRHdr      Hdr;

enum  {kYR_size = 1  // Modifier
      };
};

/******************************************************************************/
/*                        s t a t u s   R e q u e s t                         */
/******************************************************************************/
  
// Request: status
//
struct CmsStatusRequest
{      CmsRRHdr      Hdr;

enum  {kYR_Stage  = 0x01, kYR_noStage = 0x02,  // Modifier
       kYR_Resume = 0x04, kYR_Suspend = 0x08,
       kYR_Reset  = 0x10                       // Exclusive
      };
};

/******************************************************************************/
/*                         t r u n c   R e q u e s t                          */
/******************************************************************************/
  
// Request: <id> trunc <path>
// Respond: n/a
//
struct CmsTruncRequest
{      CmsRRHdr      Hdr;
//     kXR_string    Ident;
//     kXR_string    Size;
//     kXR_string    Path;
};

/******************************************************************************/
/*                           t r y   R e q u e s t                            */
/******************************************************************************/
  
// Request: try
//
struct CmsTryRequest
{      CmsRRHdr      Hdr;
       kXR_unt16     sLen;   // This is the string length in PUP format

//     kYR_string    {ipaddr:port}[up to STMax];
};

/******************************************************************************/
/*                        u p d a t e   R e q u e s t                         */
/******************************************************************************/
  
// Request: update
//
struct CmsUpdateRequest
{      CmsRRHdr      Hdr;
};

/******************************************************************************/
/*                         u s a g e   R e q u e s t                          */
/******************************************************************************/
  
// Request: usage
//
struct CmsUsageRequest
{      CmsRRHdr      Hdr;
};

}; // namespace XrdCms
#endif
