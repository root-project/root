//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientReadV                                                       //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// Helper functions for the vectored read functionality                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$



#ifndef XRD_CLIENT_READV
#define XRD_CLIENT_READV

class XrdClientConn;
#include "XProtocol/XPtypes.hh"
#include "XProtocol/XProtocol.hh"
#include "XrdClient/XrdClientVector.hh"

struct XrdClientReadVinfo {
  kXR_int64 offset;
  kXR_int32 len;
};

class XrdClientReadV {
public:
  
  // Builds a request and sends it to the server
    // If destbuf == 0 the request is sent asynchronously
  static kXR_int64 ReqReadV(XrdClientConn *xrdc, char *handle, char *destbuf,
			    XrdClientVector<XrdClientReadVinfo> &reqvect,
			    int firstreq, int nreq, int streamtosend);
  
  // Picks a readv response and puts the individual chunks into the dest buffer
  static kXR_int32 UnpackReadVResp(char *destbuf, char *respdata, kXR_int32 respdatalen,
				   readahead_list *buflis, int nbuf);
  
  // Picks a readv response and puts the individual chunks into the cache
  static kXR_int32 SubmitToCacheReadVResp(XrdClientConn *xrdc, char *respdata,
					    kXR_int32 respdatalen);
  
  static void PreProcessChunkRequest(XrdClientVector<XrdClientReadVinfo> &reqvect,
				     kXR_int64 offs, kXR_int32 len,
				     kXR_int64 filelen);
				     
  static void PreProcessChunkRequest(XrdClientVector<XrdClientReadVinfo> &reqvect,
				     kXR_int64 offs, kXR_int32 len,
				     kXR_int64 filelen,
				     kXR_int32 spltsize);


};





#endif
