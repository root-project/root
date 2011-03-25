#ifndef __XPROTOCOL_H
#define __XPROTOCOL_H

//#ifndef __GNUC__
//#define __attribute__(x)
//#ifdef SUNCC
//#pragma pack(4)
//#endif
//#endif

#ifdef __CINT__
#define __attribute__(x)
#endif

//        $Id$

#include "XProtocol/XPtypes.hh"

// KINDS of SERVERS
//
//
#define kXR_DataServer 1
#define kXR_LBalServer 0
#define kXR_maxReqRetry 10

//
// Kind of error inside a XTNetFile's routine (temporary)
//
enum XReqErrorType {
   kGENERICERR = 0,    // Generic error
   kREAD,              // Error while reading from stream
   kWRITE,             // Error while writing to stream
   kREDIRCONNECT,      // Error redirecting to a given host
   kOK,                // Everything seems ok
   kNOMORESTREAMS      // No more available stream IDs for
                       // async processing
};

//______________________________________________
// PROTOCOL DEFINITION: CLIENT'S REQUESTS TYPES
//______________________________________________
// 
enum XRequestTypes {
   kXR_auth    =  3000,
   kXR_query,   // 3001
   kXR_chmod,   // 3002
   kXR_close,   // 3003
   kXR_dirlist, // 3004
   kXR_getfile, // 3005
   kXR_protocol,// 3006
   kXR_login,   // 3007
   kXR_mkdir,   // 3008
   kXR_mv,      // 3009
   kXR_open,    // 3010
   kXR_ping,    // 3011
   kXR_putfile, // 3012
   kXR_read,    // 3013
   kXR_rm,      // 3014
   kXR_rmdir,   // 3015
   kXR_sync,    // 3016
   kXR_stat,    // 3017
   kXR_set,     // 3018
   kXR_write,   // 3019
   kXR_admin,   // 3020
   kXR_prepare, // 3021
   kXR_statx,   // 3022
   kXR_endsess, // 3023
   kXR_bind,    // 3024
   kXR_readv,   // 3025
   kXR_verifyw, // 3026
   kXR_locate,  // 3027
   kXR_truncate // 3028
};

// OPEN MODE FOR A REMOTE FILE
enum XOpenRequestMode {
   kXR_ur = 0x100,
   kXR_uw = 0x080,
   kXR_ux = 0x040,
   kXR_gr = 0x020,
   kXR_gw = 0x010,
   kXR_gx = 0x008,
   kXR_or = 0x004,
   kXR_ow = 0x002,
   kXR_ox = 0x001
};

enum XMkdirOptions {
   kXR_mknone  = 0,
   kXR_mkdirpath  = 1
};

// this is a bitmask
enum XLoginCapVer {
   kXR_lcvnone = 0,
   kXR_vermask = 63,
   kXR_asyncap = 128
};

// this is a single number that goes into capver as the version
//
enum XLoginVersion {
   kXR_ver000 = 0,  // Old clients predating history
   kXR_ver001 = 1,  // Generally implemented 2005 protocol
   kXR_ver002 = 2   // Same as 1 but adds asyncresp recognition
};

enum XStatRequestOption {
   kXR_vfs    = 1
};

enum XStatRespFlags {
   kXR_file    = 0,
   kXR_xset    = 1,
   kXR_isDir   = 2,
   kXR_other   = 4,
   kXR_offline = 8,
   kXR_readable=16,
   kXR_writable=32,
   kXR_poscpend=64
};

enum XDirlistRequestOption {
   kXR_online = 1
};

enum XOpenRequestOption {
   kXR_compress = 1,
   kXR_delete   = 2,
   kXR_force    = 4,
   kXR_new      = 8,
   kXR_open_read= 16,
   kXR_open_updt= 32,
   kXR_async    = 64,
   kXR_refresh  = 128,
   kXR_mkpath   = 256,
   kXR_open_apnd= 512,
   kXR_retstat  = 1024,
   kXR_replica  = 2048,
   kXR_posc     = 4096,
   kXR_nowait   = 8192,
   kXR_seqio    =16384
};

enum XQueryType {
   kXR_QStats = 1,
   kXR_QPrep  = 2,
   kXR_Qcksum = 3,
   kXR_Qxattr = 4,
   kXR_Qspace = 5,
   kXR_Qckscan= 6,
   kXR_Qconfig= 7,
   kXR_Qvisa  = 8,
   kXR_Qopaque=16,
   kXR_Qopaquf=32
};

enum XVerifyType {
   kXR_nocrc  = 0,
   kXR_crc32  = 1
};

enum XLogonType {
   kXR_useruser  = 0,
   kXR_useradmin = 1
};

// Andy's request for async/unsolicited
enum XPrepRequestOption {
   kXR_cancel = 1,
   kXR_notify = 2,
   kXR_noerrs = 4,
   kXR_stage  = 8,
   kXR_wmode  = 16,
   kXR_coloc  = 32,
   kXR_fresh  = 64
};

//_______________________________________________
// PROTOCOL DEFINITION: SERVER'S RESPONSES TYPES
//_______________________________________________
//
enum XResponseType {
   kXR_ok      = 0,
   kXR_oksofar = 4000,
   kXR_attn,
   kXR_authmore,
   kXR_error,
   kXR_redirect,
   kXR_wait,
   kXR_waitresp,
   kXR_noResponsesYet = 10000
};

//_______________________________________________
// PROTOCOL DEFINITION: SERVER"S ATTN CODES
//_______________________________________________

enum XActionCode {
   kXR_asyncab = 5000,
   kXR_asyncdi,
   kXR_asyncms,
   kXR_asyncrd,
   kXR_asyncwt,
   kXR_asyncav,
   kXR_asynunav,
   kXR_asyncgo,
   kXR_asynresp
};

//_______________________________________________
// PROTOCOL DEFINITION: SERVER'S ERROR CODES
//_______________________________________________
//
enum XErrorCode {
   kXR_ArgInvalid = 3000,
   kXR_ArgMissing,
   kXR_ArgTooLong,
   kXR_FileLocked,
   kXR_FileNotOpen,
   kXR_FSError,
   kXR_InvalidRequest,
   kXR_IOError,
   kXR_NoMemory,
   kXR_NoSpace,
   kXR_NotAuthorized,
   kXR_NotFound,
   kXR_ServerError,
   kXR_Unsupported,
   kXR_noserver,
   kXR_NotFile,
   kXR_isDirectory,
   kXR_Cancelled,
   kXR_ChkLenErr,
   kXR_ChkSumErr,
   kXR_inProgress,
   kXR_noErrorYet = 10000
};


//______________________________________________
// PROTOCOL DEFINITION: CLIENT'S REQUESTS STRUCTS
//______________________________________________
// 
// We need to pack structures sent all over the net!
// __attribute__((packed)) assures no padding bytes.
//
// Nice bodies of the headers for the client requests.
// Note that the protocol specifies these values to be in network
//  byte order when sent
//
// G.Ganis: use of flat structures to avoid packing options

struct ClientAdminRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientAuthRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[12];
   kXR_char credtype[4];
   kXR_int32  dlen;
};
struct ClientBindRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  sessid[16];
   kXR_int32  dlen;
};
struct ClientChmodRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  reserved[14];
   kXR_unt16 mode;
   kXR_int32  dlen;
};
struct ClientCloseRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char fhandle[4];
   kXR_int64 fsize;
   kXR_char reserved[4];
   kXR_int32  dlen;
};
struct ClientDirlistRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[15];
   kXR_char options[1];
   kXR_int32  dlen;
};
struct ClientEndsessRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  sessid[16];
   kXR_int32  dlen;
};
struct ClientGetfileRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 options;
   kXR_char reserved[8];
   kXR_int32 buffsz;
   kXR_int32  dlen;
};
struct ClientLocateRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_unt16 options;
   kXR_char reserved[14];
   kXR_int32  dlen;
};
struct ClientLoginRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 pid;
   kXR_char username[8];
   kXR_char reserved[2];
   kXR_char capver[1];
   kXR_char role[1];
   kXR_int32  dlen;
};
struct ClientMkdirRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char options[1];
   kXR_char reserved[13];
   kXR_unt16 mode;
   kXR_int32  dlen;
};
struct ClientMvRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientOpenRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_unt16 mode;
   kXR_unt16 options;
   kXR_char  reserved[12];
   kXR_int32  dlen;
};
struct ClientPingRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientProtocolRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientPrepareRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char options;
   kXR_char prty;
   kXR_char reserved[14];
   kXR_int32  dlen;
};
struct ClientPutfileRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 options;
   kXR_char  reserved[8];
   kXR_int32 buffsz;
   kXR_int32  dlen;
};
struct ClientQueryRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_unt16 infotype;
   kXR_char  reserved1[2];
   kXR_char  fhandle[4];
   kXR_char  reserved2[8];
   kXR_int32 dlen;
};
struct ClientReadRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char fhandle[4];
   kXR_int64 offset;
   kXR_int32 rlen;
   kXR_int32  dlen;
};
struct ClientReadVRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  reserved[16];
   kXR_int32  dlen;
};
struct ClientRmRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientRmdirRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientSetRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char reserved[16];
   kXR_int32  dlen;
};
struct ClientStatRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  options;
   kXR_char reserved[11];
   kXR_char fhandle[4];
   kXR_int32  dlen;
};
struct ClientSyncRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char fhandle[4];
   kXR_char reserved[12];
   kXR_int32  dlen;
};
struct ClientTruncateRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char fhandle[4];
   kXR_int64 offset;
   kXR_char reserved[4];
   kXR_int32  dlen;
};
struct ClientWriteRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char fhandle[4];
   kXR_int64 offset;
   kXR_char  pathid;
   kXR_char reserved[3];
   kXR_int32  dlen;
};
struct ClientVerifywRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  fhandle[4];
   kXR_int64 offset;
   kXR_char  pathid;
   kXR_char  vertype;       // One of XVerifyType
   kXR_char  reserved[2];
   kXR_int32 dlen;          // Includes crc length
};

struct ClientRequestHdr {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_char  body[16];
   kXR_int32  dlen;
};

typedef union {
   struct ClientRequestHdr header;
   struct ClientAdminRequest admin;
   struct ClientAuthRequest auth;
   struct ClientBindRequest bind;
   struct ClientChmodRequest chmod;
   struct ClientCloseRequest close;
   struct ClientDirlistRequest dirlist;
   struct ClientEndsessRequest endsess;
   struct ClientGetfileRequest getfile;
   struct ClientLocateRequest locate;
   struct ClientLoginRequest login;
   struct ClientMkdirRequest mkdir;
   struct ClientMvRequest mv;
   struct ClientOpenRequest open;
   struct ClientPingRequest ping;
   struct ClientPrepareRequest prepare;
   struct ClientProtocolRequest protocol;
   struct ClientPutfileRequest putfile;
   struct ClientQueryRequest query;
   struct ClientReadRequest read;
   struct ClientReadVRequest readv;
   struct ClientRmRequest rm;
   struct ClientRmdirRequest rmdir;
   struct ClientSetRequest set;
   struct ClientStatRequest stat;
   struct ClientSyncRequest sync;
   struct ClientTruncateRequest truncate;
   struct ClientWriteRequest write;
} ClientRequest;

struct readahead_list {
   kXR_char fhandle[4];
   kXR_int32 rlen;
   kXR_int64 offset;
};

struct read_args {
   kXR_char       pathid;
   kXR_char       reserved[7];
   // his struct is followed by an array of readahead_list
};

//_____________________________________________________________________
//   PROTOCOL DEFINITION: SERVER'S RESPONSE
//_____________________________________________________________________
//

// Nice header for the server response.
// Note that the protocol specifies these values to be in network
// byte order when sent
//
// G.Ganis: The following structures never need padding bytes:
//          no need of packing options

struct ServerResponseHeader {
   kXR_char streamid[2];
   kXR_unt16 status;
   kXR_int32  dlen;
};

// Body for the kXR_bind response... useful
struct ServerResponseBody_Bind {
    kXR_char substreamid;
};

// Body for the kXR_open response... useful
struct ServerResponseBody_Open {
   kXR_char fhandle[4];
   kXR_int32 cpsize;   // cpsize & cptype returned if kXR_compress *or*
   kXR_char cptype[4]; // kXR_retstat is specified
}; // info will follow if kXR_retstat is specified

// Body for the kXR_protocol response... useful
struct ServerResponseBody_Protocol {
   kXR_int32 pval;
   kXR_int32 flags;
};

struct ServerResponseBody_Redirect {
   kXR_int32 port;
   char host[4096]; // Should be sufficient for every use
};

struct ServerResponseBody_Error {
   kXR_int32 errnum;
   char errmsg[4096]; // Should be sufficient for every use
};

struct ServerResponseBody_Wait {
   kXR_int32 seconds;
   char infomsg[4096]; // Should be sufficient for every use
};

struct ServerResponseBody_Attn {
   kXR_int32 actnum;
   char parms[4096]; // Should be sufficient for every use
};

struct ServerResponseBody_Attn_asyncrd {
   kXR_int32 actnum;
   kXR_int32 port;
   char host[4092];
};

struct ServerResponseBody_Attn_asynresp {
   kXR_int32            actnum;
   char reserved[4];
   ServerResponseHeader resphdr;
   char respdata[4096];
};

struct ServerResponseBody_Attn_asyncwt {
   kXR_int32 actnum;
   kXR_int32 wsec;
};

struct ServerResponseBody_Attn_asyncdi {
   kXR_int32 actnum;
   kXR_int32 wsec;
   kXR_int32 msec;
};

void ServerResponseHeader2NetFmt(struct ServerResponseHeader *srh);

// The fields to be sent as initial handshake
struct ClientInitHandShake {
   kXR_int32 first;
   kXR_int32 second;
   kXR_int32 third;
   kXR_int32 fourth;
   kXR_int32 fifth;
};

// The body received after the first handshake's header
struct ServerInitHandShake {
   kXR_int32 msglen;
   kXR_int32 protover;
   kXR_int32 msgval;
};



typedef kXR_int32 ServerResponseType;

struct ALIGN_CHECK {char chkszreq[25-sizeof(ClientRequest)];
   char chkszrsp[ 9-sizeof(ServerResponseHeader)];
};

/******************************************************************************/
/*                   X P r o t o c o l   U t i l i t i e s                    */
/******************************************************************************/

#include <errno.h>
#if defined(WIN32)
#if !defined(ENOTBLK)
#  define ENOTBLK 15
#endif
#if !defined(ETXTBSY)
#define ETXTBSY 26
#endif
#if !defined(ENOBUFS)
#define ENOBUFS 105
#endif
#if !defined(ENETUNREACH)
#define ENETUNREACH 114
#endif
#endif
  
class XProtocol
{
public:

// mapError() is the occicial mapping from errno to xrootd protocol error.
//
static int mapError(int rc)
      {if (rc < 0) rc = -rc;
       switch(rc)
          {case ENOENT:       return kXR_NotFound;
           case EPERM:        return kXR_NotAuthorized;
           case EACCES:       return kXR_NotAuthorized;
           case EIO:          return kXR_IOError;
           case ENOMEM:       return kXR_NoMemory;
           case ENOBUFS:      return kXR_NoMemory;
           case ENOSPC:       return kXR_NoSpace;
           case ENAMETOOLONG: return kXR_ArgTooLong;
           case ENETUNREACH:  return kXR_noserver;
           case ENOTBLK:      return kXR_NotFile;
           case EISDIR:       return kXR_isDirectory;
           case EEXIST:       return kXR_InvalidRequest;
           case ETXTBSY:      return kXR_inProgress;
           default:           return kXR_FSError;
          }
      }
};
#endif
