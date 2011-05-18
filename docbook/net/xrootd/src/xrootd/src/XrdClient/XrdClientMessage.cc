//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientMessage                                                     // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// A message coming from a physical connection. I.e. a server response  //
//  or some kind of error                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

const char *XrdClientMessageCVSID = "$Id$";

#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientProtocol.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientPhyConnection.hh"

#include <stdlib.h> // for malloc
#include <string.h> // for memcpy


//__________________________________________________________________________
XrdClientMessage::XrdClientMessage(struct ServerResponseHeader header)
{
  // Constructor

  fStatusCode = kXrdMSC_ok;
  memcpy((void *)&fHdr, (const void*)&header, sizeof(ServerResponseHeader));
  fData = 0;
  fMarshalled = false;
  if (!CreateData()) {
    Error("XrdClientMessage", 
	  "Error allocating " << fHdr.dlen << " bytes.");
    fAllocated = false;
  } else 
    fAllocated = true;
}

//__________________________________________________________________________
XrdClientMessage::XrdClientMessage()
{
  // Default constructor

  memset(&fHdr, 0, sizeof(fHdr));
  fStatusCode = kXrdMSC_ok;
  fData = 0;
  fMarshalled = false;
  fAllocated = false;
}

//__________________________________________________________________________
XrdClientMessage::~XrdClientMessage()
{
  // Destructor

  if (fData)
    free(fData);
}

//__________________________________________________________________________
void *XrdClientMessage::DonateData()
{
  // Unlink the owned data in order to pass them elsewhere

  void *res = fData;
  fData = 0;
  fAllocated = false;
  
  return (res);
}

//__________________________________________________________________________
bool XrdClientMessage::CreateData()
{
  // Allocate data

  if (!fAllocated) {
    if (fHdr.dlen > 0) {

      long pgsz = sysconf(_SC_PAGESIZE);
      int memtrbl = 0;
      if ((pgsz > 0) && (fHdr.dlen+1 > pgsz))
	memtrbl = posix_memalign(&fData, pgsz, fHdr.dlen+1);
      else
	fData = malloc(fHdr.dlen+1);

      if (!fData || memtrbl) {
         Error("XrdClientMessage::CreateData",
               "Fatal ERROR *** memory allocation alloc of " <<
               fHdr.dlen+1 << " bytes failed."
               " Probable system resources exhausted.");
         return FALSE;
      }
      char *tmpPtr = (char *)fData;

      // Useful to get always 0-terminated strings
      tmpPtr[fHdr.dlen] = '\0';
    }
    if (!fData)
      return FALSE;
    else
      return TRUE;
  } else
    return TRUE;
}

//__________________________________________________________________________
void XrdClientMessage::Marshall()
{
  // Marshall, i.e. put in network byte order

  if (!fMarshalled) {
    ServerResponseHeader2NetFmt(&fHdr);
    fMarshalled = TRUE;
  }
}

//__________________________________________________________________________
void XrdClientMessage::Unmarshall()
{
  // Unmarshall, i.e. from network byte to normal order

  if (fMarshalled) {
    clientUnmarshall(&fHdr);
    fMarshalled = FALSE;
  }
}

//__________________________________________________________________________
int XrdClientMessage::ReadRaw(XrdClientPhyConnection *phy)
{
  // Given a physical connection, we completely build the content
  // of the message, reading it from the socket of a phyconn

  int readres;
  int readLen = sizeof(ServerResponseHeader);
  int usedsubstreamid = 0;

  phy->ReadLock();

  Info(XrdClientDebug::kDUMPDEBUG,
       "XrdClientMessage::ReadRaw",
       "Reading header (" << readLen << " bytes).");
  
  // Read a header from any substream and report it
  readres = phy->ReadRaw((void *)&fHdr, readLen, -1, &usedsubstreamid);
  if (readres == readLen) phy->PauseSelectOnSubstream(usedsubstreamid);

  phy->ReadUnLock();

  if (readres != readLen) {

    if (readres == TXSOCK_ERR_TIMEOUT)
      SetStatusCode(kXrdMSC_timeout);
    else {
      Info(XrdClientDebug::kNODEBUG,"XrdClientMessage::ReadRaw",
	   "Failed to read header (" << readLen << " bytes).");
      SetStatusCode(kXrdMSC_readerr);

    }
    memset(&fHdr, 0, sizeof(fHdr));
  }

  // the data arrive marshalled from the server (i.e. network byte order)
  SetMarshalled(TRUE);
  Unmarshall();

  Info(XrdClientDebug::kDUMPDEBUG,
       "XrdClientMessage::ReadRaw"," sid: "<<HeaderSID() <<
       ", IsAttn: " << IsAttn() <<
       ", substreamid: " << usedsubstreamid);

  if (fHdr.dlen > 0) {

    Info(XrdClientDebug::kDUMPDEBUG,
	 "XrdClientMessage::ReadRaw",
	 "Reading data (" << fHdr.dlen << " bytes) from substream " << usedsubstreamid);

    if (!CreateData()) {

      Info(XrdClientDebug::kNODEBUG,"XrdClientMessage::ReadRaw",
	   "Failed to create data (" << fHdr.dlen << " bytes) from substream " <<
	   usedsubstreamid << ".");

      SetStatusCode(kXrdMSC_timeout);

      memset(&fHdr, 0, sizeof(fHdr));

    } else if (phy->ReadRaw(fData, fHdr.dlen, usedsubstreamid) != fHdr.dlen) {

      Info(XrdClientDebug::kNODEBUG,"XrdClientMessage::ReadRaw",
	   "Failed to read data (" << fHdr.dlen << " bytes) from substream " <<
	   usedsubstreamid << ".");

      free( DonateData() );

      if (readres == TXSOCK_ERR_TIMEOUT)
	SetStatusCode(kXrdMSC_timeout);
      else
	SetStatusCode(kXrdMSC_readerr);

      memset(&fHdr, 0, sizeof(fHdr));
    }
  }
  phy->RestartSelectOnSubstream(usedsubstreamid);
  //  phy->ReadUnLock();
  return 1;
}

//___________________________________________________________________________
void XrdClientMessage::Int2CharStreamid(kXR_char *charstreamid, short intstreamid)
{
  // Converts a streamid given as an integer to its representation
  // suitable for the streamid inside the messages (i.e. ascii)

  memcpy(charstreamid, &intstreamid, sizeof(intstreamid));
}

//___________________________________________________________________________
kXR_unt16 XrdClientMessage::CharStreamid2Int(kXR_char *charstreamid)
{
  // Converts a streamid given as an integer to its representation
  // suitable for the streamid inside the messages (i.e. ascii)

  kXR_unt16 res = *((short *)charstreamid);

  return res;
}
