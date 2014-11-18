//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientMStream                                                     //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// Helper code for XrdClient to handle multistream behavior             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$


#ifndef XRD_CLI_MSTREAM
#define XRD_CLI_MSTREAM

#include "XrdClient/XrdClientConn.hh"

class XrdClientMStream {





public:
  
    // Compute the parameters to split blocks
    static void GetGoodSplitParameters(XrdClientConn *cliconn,
			      int &spltsize, int &reqsperstream,
			      kXR_int32 len);

    // Establish all the parallel streams, stop
    // adding streams at the first creation refusal/failure
    static int EstablishParallelStreams(XrdClientConn *cliconn);

    // Add a parallel stream to the pool used by the given client inst
   static int AddParallelStream(XrdClientConn *cliconn, int port, int windowsz, int tempid);

    // Remove a parallel stream to the pool used by the given client inst
    static int RemoveParallelStream(XrdClientConn *cliconn, int substream);

    // Binds the pending temporary parallel stream to the current session
    // Returns into newid the substreamid assigned by the server
    static bool BindPendingStream(XrdClientConn *cliconn, int substreamid, int &newid);

    struct ReadChunk {
	kXR_int64 offset;
	kXR_int32 len;
	int streamtosend;
    };
    

    // This splits a long requests into many smaller requests, to be sent in parallel
    //  through multiple streams
    // Returns false if the chunk is not worth splitting
    static bool SplitReadRequest(XrdClientConn *cliconn, kXR_int64 offset, kXR_int32 len,
				 XrdClientVector<ReadChunk> &reqlists);


};






#endif
