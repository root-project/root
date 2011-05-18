//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientProtocol                                                    // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// utility functions to deal with the protocol                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//           $Id$

#ifndef XRD_CPROTOCOL_H
#define XRD_CPROTOCOL_H

#include "XProtocol/XProtocol.hh"


void clientMarshall(ClientRequest* str);
void clientMarshallReadAheadList(readahead_list *buf_list, kXR_int32 dlen);
void clientUnMarshallReadAheadList(readahead_list *buf_list, kXR_int32 dlen);
void clientUnmarshall(struct ServerResponseHeader* str);

void ServerResponseHeader2NetFmt(struct ServerResponseHeader *srh);
void ServerInitHandShake2HostFmt(struct ServerInitHandShake *srh);

bool isRedir(struct ServerResponseHeader *ServerResponse);

char *convertRequestIdToChar(kXR_unt16 requestid);

void PutFilehandleInRequest(ClientRequest* str, char *fHandle);

char *convertRespStatusToChar(kXR_unt16 status);

void smartPrintClientHeader(ClientRequest* hdr);
void smartPrintServerHeader(struct ServerResponseHeader* hdr);


#endif
