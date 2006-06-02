// @(#)root/proofd:$Name:  $:$Id: XProofProtocol.h,v 1.3 2006/04/19 10:52:46 rdm Exp $
// Author: G. Ganis  June 2005

#ifndef ROOT_XProofProtocol
#define ROOT_XProofProtocol

#ifdef __CINT__
#define __attribute__(x)
#endif

#include "XProtocol/XProtocol.hh"

// KINDS of SERVERS
//
//
#define kXP_MasterServer 1
#define kXR_SlaveServer 0

//______________________________________________
// XPROOFD PROTOCOL DEFINITION: CLIENT'S REQUESTS TYPES
//
// These are used by TXProofMgr to interact with its image on
// the server side and the sessions
//
//______________________________________________
//
enum XProofRequestTypes {
   kXP_version      = 3100,   // protocol version
   kXP_login,      // 3101    // request to start session-context
   kXP_auth,       // 3102    // credentials to login
   kXP_create,     // 3103    // create a new session-ctx
   kXP_destroy,    // 3104    // destroy a session-ctx
   kXP_attach,     // 3105    // attach to an existing session-ctx
   kXP_detach,     // 3106    // detach from an existing session-ctx
   kXP_stop,       // 3107    // stop (reset) a running query (hard interrupt)
   kXP_pause,      // 3108    // suspend processing on a query
   kXP_resume,     // 3109    // resume processing on a query
   kXP_retrieve,   // 3110    // retrieve (partial) results of a query
   kXP_archive,    // 3111    // archive instructions for query results
   kXP_sendmsg,    // 3112    // send a msg to a session-ctx
   kXP_admin,      // 3113    // admin request handled by the coordinator (not forwarded)
   kXP_interrupt,  // 3114    // urgent message
   kXP_ping,       // 3115    // ping request
   kXP_submit,     // 3116    // query submission (record query entry)
   kXP_cleanup     // 3117    // cleanup query field
};

// XPROOFD VERSION  (0xMMmnpp : MM major, mm minor, pp patch)
#define XPD_VERSION  0x010000

// KINDS of SERVERS (modes)
#define kXPD_Internal     3
#define kXPD_TopMaster    2
#define kXPD_MasterServer 1
#define kXPD_SlaveServer  0
#define kXPD_WorkerServer 0
#define kXPD_AnyServer   -1

// XPROOFD SERVER STATUS
#define kXPD_idle    0
#define kXPD_running 1

// XPROOFD EXEC STATUS
#define kXPD_terminated 0
#define kXPD_killed     1

// XPROOFD MESSAGE TYPE
#define kXPD_internal     0x1
#define kXPD_async        0x2
#define kXPD_startprocess 0x4
#define kXPD_setidle      0x8
#define kXPD_fb_prog      0x10
#define kXPD_logmsg       0x20
#define kXPD_querynum     0x40

//_______________________________________________
// PROTOCOL DEFINITION: SERVER'S RESPONSES TYPES
//_______________________________________________
//
enum XProofResponseType {
   kXP_ok            = 0,
   kXP_oksofar       = 4100,
   kXP_attn,        // 4101
   kXP_authmore,    // 4102
   kXP_error,       // 4103
   kXP_wait         // 4104
};

//_______________________________________________
// PROTOCOL DEFINITION: SERVER"S ATTN CODES
//_______________________________________________
enum XProofActionCode {
   kXPD_msg         = 5100,
   kXPD_ping,      // 5101,
   kXPD_interrupt, // 5102,
   kXPD_feedback,  // 5103
   kXPD_srvmsg,    // 5104
   kXPD_msgsid,    // 5105
   kXPD_errmsg     // 5106
};

//_______________________________________________
// PROTOCOL DEFINITION: QUERY STATUS
//_______________________________________________
//
enum XProofQueryStatus {
   kXP_pending       = 0,
   kXP_done,        // 1
   kXP_processing,  // 2
   kXP_aborted      // 3
};

//_______________________________________________
// PROTOCOL DEFINITION: SERVER'S ERROR CODES
//_______________________________________________
//
enum XPErrorCode {
   kXP_ArgInvalid = 3100,
   kXP_ArgMissing,
   kXP_ArgTooLong,
   kXP_InvalidRequest,
   kXP_IOError,
   kXP_NoMemory,
   kXP_NoSpace,
   kXP_NotAuthorized,
   kXP_NotFound,
   kXP_ServerError,
   kXP_Unsupported,
   kXP_noserver,
   kXP_nosession,
   kXP_nomanager
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

struct XPClientProofRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 sid;
   kXR_int32 int1;
   kXR_int32 int2;
   kXR_int32 int3;
   kXR_int32 dlen;
};

struct XPClientSendRcvRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 sid;
   kXR_int32 opt;
   kXR_int32 cid;
   kXR_char  reserved[4];
   kXR_int32 dlen;
};

struct XPClientArchiveRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 sid;
   kXR_int32 opt;
   kXR_char  reserved[8];
   kXR_int32 dlen;
};

struct XPClientInterruptRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 sid;
   kXR_int32 type;
   kXR_char  reserved[8];
   kXR_int32 dlen;
};

typedef union {
   struct ClientLoginRequest login;
   struct ClientAuthRequest auth;
   struct XPClientProofRequest proof;
   struct XPClientSendRcvRequest sendrcv;
   struct XPClientArchiveRequest archive;
   struct XPClientInterruptRequest interrupt;
   struct ClientRequestHdr header;
} XPClientRequest;

// Server structures and codes are identical to the one in XProtocol.hh, for
// the time being

#endif
