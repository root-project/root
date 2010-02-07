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

const char *XrdClientReadVCVSID = "$Id$";

#include "XrdClient/XrdClientReadV.hh"
#include "XrdClient/XrdClientConn.hh"
#include "XrdClient/XrdClientDebug.hh"
#include <memory.h>

// Builds a request and sends it to the server
// If destbuf == 0 the request is sent asynchronously
// nbuf returns the number of processed buffers
kXR_int64 XrdClientReadV::ReqReadV(XrdClientConn *xrdc, char *handle, char *destbuf,
				   XrdClientVector<XrdClientReadVinfo> &reqvect,
				   int firstreq, int nreq, int streamtosend) {

    readahead_list buflis[READV_MAXCHUNKS];

    Info(XrdClientDebug::kUSERDEBUG, "ReqReadV",
	 "Requesting to read " << nreq <<
	 " chunks.");

    kXR_int64 total_len = 0;

    // Now we build the protocol-ready read ahead list
    //  and also put the correct placeholders inside the cache
    for (int i = 0; i < nreq; i++) {

	    memcpy( &(buflis[i].fhandle), handle, 4 ); 


	    if (!destbuf)
		xrdc->SubmitPlaceholderToCache(reqvect[firstreq+i].offset,
					       reqvect[firstreq+i].offset +
					       reqvect[firstreq+i].len-1);

	    buflis[i].offset = reqvect[firstreq+i].offset;
	    buflis[i].rlen = reqvect[firstreq+i].len;
	    total_len += buflis[i].rlen;
    }

    if (nreq > 0) {

	// Prepare a request header 
	ClientRequest readvFileRequest;
	memset( &readvFileRequest, 0, sizeof(readvFileRequest) );
	xrdc->SetSID(readvFileRequest.header.streamid);
	readvFileRequest.header.requestid = kXR_readv;
	readvFileRequest.readv.dlen = nreq * sizeof(struct readahead_list);

	if (destbuf) {
	    // A buffer able to hold the data and the info about the chunks
	    char *res_buf = new char[total_len + (nreq * sizeof(struct readahead_list))];

	    clientMarshallReadAheadList(buflis, readvFileRequest.readv.dlen);
	    bool r = xrdc->SendGenCommand(&readvFileRequest, buflis, 0, 
					  (void *)res_buf, FALSE, (char *)"ReadV");
	    clientUnMarshallReadAheadList(buflis, readvFileRequest.readv.dlen);

	    if ( r ) {
        
		total_len = UnpackReadVResp(destbuf, res_buf,
					    xrdc->LastServerResp.dlen,
					    buflis,
					    nreq);
            }
            else
	      total_len = -1;
	
	    delete [] res_buf;
	}
	else {
	  clientMarshallReadAheadList(buflis, readvFileRequest.readv.dlen);
	  if (xrdc->WriteToServer_Async(&readvFileRequest,
					buflis) != kOK )
	    total_len = 0;

	}

    }

    Info(XrdClientDebug::kHIDEBUG, "ReqReadV",
         "Returning: total_len " << total_len);
    return total_len;
}


// Picks a readv response and puts the individual chunks into the dest buffer
kXR_int32 XrdClientReadV::UnpackReadVResp(char *destbuf, char *respdata, kXR_int32 respdatalen,
				    readahead_list *buflis, int nbuf) {

    int res = respdatalen;

    // I just rebuild the readahead_list element
    struct readahead_list header;
    kXR_int32 pos_from = 0, pos_to = 0;
    int i = 0;
    kXR_int64 cur_buf_offset = -1;
    int cur_buf_len = 0, cur_buf = 0;
    
    while ( (pos_from < respdatalen) && (i < nbuf) ) {
	memcpy(&header, respdata + pos_from, sizeof(struct readahead_list));
       
	kXR_int64 tmpl;
	memcpy(&tmpl, &header.offset, sizeof(kXR_int64) );
	tmpl = ntohll(tmpl);
	memcpy(&header.offset, &tmpl, sizeof(kXR_int64) );

	header.rlen  = ntohl(header.rlen);       

        // Do some consistency checks
        if (cur_buf_len == 0) {
           cur_buf_offset = header.offset;
           if (cur_buf_offset != buflis[cur_buf].offset) {
              res = -1;  
              break;
           }
           cur_buf_len += header.rlen;
           if (cur_buf_len == buflis[cur_buf].rlen) {
              cur_buf++;
              cur_buf_len = 0;
           }
        }

	pos_from += sizeof(struct readahead_list);
	memcpy( &destbuf[pos_to], &respdata[pos_from], header.rlen);
	pos_from += header.rlen;
	pos_to += header.rlen;
        i++;
    }

    if (pos_from != respdatalen || i != nbuf)
       Error("UnpackReadVResp","Inconsistency: pos_from " << pos_from <<
	     " respdatalen " << respdatalen <<
	     " i " << i <<
	     " nbuf " << nbuf );

    if (res > 0)
      res = pos_to;

    return res;
}

// Picks a readv response and puts the individual chunks into the cache
int XrdClientReadV::SubmitToCacheReadVResp(XrdClientConn *xrdc, char *respdata,
					   kXR_int32 respdatalen) {

    // This probably means that the server doesnt support ReadV
    // ( old version of the server )
    int res = -1;


	res = respdatalen;

	// I just rebuild the readahead_list element

	struct readahead_list header;
	kXR_int32 pos_from = 0;
        kXR_int32 rlen = 0;
	kXR_int64 offs=0;

// 	// Just to log the entries
// 	while ( pos_from < respdatalen ) {
// 	    header = ( readahead_list * )(respdata + pos_from);

// 	    memcpy(&offs, &header->offset, sizeof(kXR_int64) );
// 	    offs = ntohll(offs);
// 	    rlen = ntohl(header->rlen);   

// 	    pos_from += sizeof(struct readahead_list);

// 	    Info(XrdClientDebug::kHIDEBUG, "ReadV",
// 		 "Received chunk " << rlen << " @ " << offs );

// 	    pos_from += rlen;
// 	}

	pos_from = 0;


	while ( pos_from < respdatalen ) {
            memcpy(&header, respdata + pos_from, sizeof(struct readahead_list));

	    offs = ntohll(header.offset);
	    rlen = ntohl(header.rlen);      

	    pos_from += sizeof(struct readahead_list);

	    // NOTE: we must duplicate the buffer to be submitted, since a cache block has to be
	    // contained in one single memblock, while here we have one for multiple chunks.
	    void *newbuf = malloc(rlen);
	    memcpy(newbuf, &respdata[pos_from], rlen);

	    xrdc->SubmitRawDataToCache(newbuf, offs, offs + rlen - 1);

	    pos_from += rlen;

	}
	res = pos_from;

	free( respdata );

    return res;



}



void XrdClientReadV::PreProcessChunkRequest(XrdClientVector<XrdClientReadVinfo> &reqvect,
					    kXR_int64 offs, kXR_int32 len,
					    kXR_int64 filelen,
					    kXR_int32 spltsize) {
  // Process a single subchunk request, eventually splitting it into more than one

  kXR_int32 len_ok = 0;
  kXR_int32 newlen = xrdmin(filelen - offs, len);
  
  // We want blocks whose len does not exceed READV_MAXCHUNKSIZE
  spltsize = xrdmin(spltsize, READV_MAXCHUNKSIZE);

  while (len_ok < newlen) {
    XrdClientReadVinfo nfo;

    nfo.offset = offs+len_ok;
    nfo.len = xrdmin(newlen-len_ok, spltsize);

    reqvect.Push_back(nfo);

    len_ok += nfo.len;
  }
  
}


