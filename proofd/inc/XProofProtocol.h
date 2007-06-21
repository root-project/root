// @(#)root/proofd:$Name:  $:$Id: XProofProtocol.h,v 1.6 2006/12/03 23:34:04 rdm Exp $
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
   kXP_login        = 3101,    // client login
   kXP_auth         = 3102,    // credentials to login
   kXP_create       = 3103,    // create a new session-ctx
   kXP_destroy      = 3104,    // destroy a session-ctx
   kXP_attach       = 3105,    // attach to an existing session-ctx
   kXP_detach       = 3106,    // detach from an existing session-ctx
   kXP_urgent       = 3111,    // urgent msg for a processing session (e.g. stop/abort)
   kXP_sendmsg      = 3112,    // send a msg to a session-ctx
   kXP_admin        = 3113,    // admin request handled by the coordinator (not forwarded)
   kXP_interrupt    = 3114,    // urgent message
   kXP_ping         = 3115,    // ping request
   kXP_cleanup      = 3116,    // clean-up a session-ctx or a client section
   kXP_readbuf      = 3117     // read a buffer from a file
};

// XPROOFD VERSION  (0xMMmnpp : MM major, mm minor, pp patch)
#define XPD_VERSION  0x010000

// KINDS of SERVERS (modes)
#define kXPD_Admin        4
#define kXPD_Internal     3
#define kXPD_TopMaster    2
#define kXPD_MasterServer 1
#define kXPD_SlaveServer  0
#define kXPD_WorkerServer 0
#define kXPD_AnyServer   -1

// Operations modes
#define kXPD_OpModeOpen       0
#define kXPD_OpModeControlled 1

// Type of resources
enum EResourceType {
   kRTStatic,
   kRTDynamic
};

// Worker selection options
enum EStaticSelOpt {
   kSSORoundRobin = 0,
   kSSORandom     = 1,
   kSSOLoadBased  = 2
};

// Should be the same as in proofx/inc/TXSocket.h
enum EAdminMsgType {
   kQuerySessions     = 1000,
   kSessionTag        = 1001,
   kSessionAlias      = 1002,
   kGetWorkers        = 1003,
   kQueryWorkers      = 1004,
   kCleanupSessions   = 1005,
   kQueryLogPaths     = 1005,
   kReadBuffer        = 1007,
   kQueryROOTVersions = 1008,
   kROOTVersion       = 1009,
   kGroupProperties   = 1010
};

// XPROOFD Worker CPU load sharing options
enum XProofSchedOpts {
   kXPD_sched_off = 0,
   kXPD_sched_fraction = 1,
   kXPD_sched_priority = 2
};

// XPROOFD SERVER STATUS
enum XProofSessionStatus {
   kXPD_idle            = 0,
   kXPD_running         = 1,
   kXPD_shutdown        = 2,
   kXPD_unknown         = 3
};

// XPROOFD MESSAGE TYPE
#define kXPD_internal     0x1
#define kXPD_async        0x2
#define kXPD_startprocess 0x4
#define kXPD_setidle      0x8
#define kXPD_fb_prog      0x10
#define kXPD_logmsg       0x20
#define kXPD_querynum     0x40
#define kXPD_process      0x80

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
   kXPD_msg         = 5100,    // Generic message from server
   kXPD_ping,      // 5101     // Ping request
   kXPD_interrupt, // 5102     // Interrupt request
   kXPD_feedback,  // 5103     // Feedback message
   kXPD_srvmsg,    // 5104     // Log string from server
   kXPD_msgsid,    // 5105     // Generic message from server with ID
   kXPD_errmsg,    // 5106     // Error message from server with log string
   kXPD_timer,     // 5107     // Server request to start a timer for delayed termination
   kXPD_urgent,    // 5108     // Urgent message to be processed in the reader thread
   kXPD_flush,     // 5109     // Server request to flush stdout (before retrieving logs)
   kXPD_inflate    // 5110     // Server request to inflate processing times
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

struct XPClientReadbufRequest {
   kXR_char  streamid[2];
   kXR_unt16 requestid;
   kXR_int32 len;
   kXR_int64 ofs;
   kXR_int32 int1;
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
   struct XPClientReadbufRequest readbuf;
   struct XPClientSendRcvRequest sendrcv;
   struct XPClientArchiveRequest archive;
   struct XPClientInterruptRequest interrupt;
   struct ClientRequestHdr header;
} XPClientRequest;

// Server structures and codes are identical to the one in XProtocol.hh, for
// the time being

#endif
