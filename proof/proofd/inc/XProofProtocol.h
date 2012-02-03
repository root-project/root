// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2005

#ifndef ROOT_XProofProtocol
#define ROOT_XProofProtocol

#ifdef __CINT__
#define __attribute__(x)
#endif

#include "XProtocol/XProtocol.hh"

#define XPD_DEF_PORT 1093

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
// Changes should be propagated to the related strings
// for notification in XrdProofdAux::ProofRequestTypes(...) and
// in XProofProtUtils.cxx
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
   kXP_readbuf      = 3117,    // read a buffer from a file
   kXP_touch        = 3118,    // touch the client admin path
   kXP_ctrlc        = 3119,    // propagate a Ctrl-C issued by the client 
   kXP_direct       = 3120,    // direct data connection
//
   kXP_Undef        = 3121     // This should always be last: just increment
};

// XPROOFD VERSION  (0xMMmmpp : MM major, mm minor, pp patch)
#define XPD_VERSION  0x010600

// KINDS of connections (modes)
#define kXPD_Admin        4
#define kXPD_Internal     3
#define kXPD_ClientMaster 2
#define kXPD_MasterMaster 1
#define kXPD_MasterWorker 0
#define kXPD_AnyConnect  -1

// KINDS of servers
#define kXPD_TopMaster    2
#define kXPD_Master       1
#define kXPD_Worker       0
#define kXPD_AnyServer   -1

// Operations modes
#define kXPD_OpModeOpen       0
#define kXPD_OpModeControlled 1

// Type of resources
enum EResourceType {
   kRTNone    = -1,
   kRTStatic  = 0,
   kRTDynamic = 1
};

// Worker selection options
enum EStaticSelOpt {
   kSSORoundRobin = 0,
   kSSORandom     = 1,
   kSSOLoadBased  = 2
};

// Message types used in SendCoordinator(...)
// Must be consistent with the names in XrdProofdAux.cxx
enum EAdminMsgType {
   kQuerySessions     = 1000,
   kSessionTag        = 1001,
   kSessionAlias      = 1002,
   kGetWorkers        = 1003,
   kQueryWorkers      = 1004,
   kCleanupSessions   = 1005,
   kQueryLogPaths     = 1006,
   kReadBuffer        = 1007,
   kQueryROOTVersions = 1008,
   kROOTVersion       = 1009,
   kGroupProperties   = 1010,
   kSendMsgToUser     = 1011,
   kReleaseWorker     = 1012,
   kExec              = 1013,
   kGetFile           = 1014,
   kPutFile           = 1015,
   kCpFile            = 1016,
   kQueryMssUrl       = 1017,
//
   kUndef             = 1018    // This should always be last: do not touch it
};

// Exec types
enum EAdminExecType {
   kRm                = 0,
   kLs                = 1,
   kMore              = 2,
   kGrep              = 3,
   kTail              = 4,
   kMd5sum            = 5,
   kStat              = 6,
   kFind              = 7
};

// XPROOFD Worker CPU load sharing options
enum XProofSchedOpts {
   kXPD_sched_off = 0,
   kXPD_sched_local = 1,     // Priorities defined in a local file on the worker
   kXPD_sched_central = 2    // Priorities communicated by the master
};

// XPROOFD SERVER STATUS
enum XProofSessionStatus {
   kXPD_idle            = 0,
   kXPD_running         = 1,
   kXPD_shutdown        = 2,
   kXPD_enqueued        = 3,
   kXPD_unknown         = 4
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

// Special GetWorkers reply tags
const char* const XPD_GW_Failed        = "|failed|";
const char* const XPD_GW_QueryEnqueued = "|enqueued|";
const char* const XPD_GW_Static        = "static:";

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
   kXPD_inflate,   // 5110     // Server request to inflate processing times
   kXPD_priority,  // 5111     // Server request to propagate a group priority
   kXPD_wrkmortem, // 5112     // A worker just died or terminated
   kXPD_touch,     // 5113     // Touch the connection and schedule an asynchronous remote touch
   kXPD_resume,     // 5114    // process the next query (to be sent to TXSocket in TXProofServ)
   kXPD_clusterinfo // 5115    // Information about running sessions
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
   kXP_ArgInvalid         = 3100,
   kXP_ArgMissing,       // 3101
   kXP_ArgTooLong,       // 3102
   kXP_InvalidRequest,   // 3103
   kXP_IOError,          // 3104
   kXP_NoMemory,         // 3105
   kXP_NoSpace,          // 3106
   kXP_NotAuthorized,    // 3107
   kXP_NotFound,         // 3108
   kXP_ServerError,      // 3109
   kXP_Unsupported,      // 3110
   kXP_noserver,         // 3111
   kXP_nosession,        // 3112
   kXP_nomanager,        // 3113
   kXP_reconnecting,     // 3114
   kXP_TooManySess       // 3115
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
