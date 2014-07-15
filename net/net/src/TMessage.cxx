// @(#)root/net:$Id$
// Author: Fons Rademakers   19/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMessage                                                             //
//                                                                      //
// Message buffer class used for serializing objects and sending them   //
// over a network. This class inherits from TBuffer the basic I/O       //
// serializer.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMessage.h"
#include "Compression.h"
#include "TVirtualStreamerInfo.h"
#include "Bytes.h"
#include "TFile.h"
#include "TProcessID.h"
#include "RZip.h"

Bool_t TMessage::fgEvolution = kFALSE;


ClassImp(TMessage)

//______________________________________________________________________________
TMessage::TMessage(UInt_t what, Int_t bufsiz) :
   TBufferFile(TBuffer::kWrite, bufsiz + 2*sizeof(UInt_t))
{
   // Create a TMessage object for storing objects. The "what" integer
   // describes the type of message. Predifined ROOT system message types
   // can be found in MessageTypes.h. Make sure your own message types are
   // unique from the ROOT defined message types (i.e. 0 - 10000 are
   // reserved by ROOT). In case you OR "what" with kMESS_ACK, the message
   // will wait for an acknowledgement from the remote side. This makes
   // the sending process synchronous. In case you OR "what" with kMESS_ZIP,
   // the message will be compressed in TSocket using the zip algorithm
   // (only if message is > 256 bytes).

   // space at the beginning of the message reserved for the message length
   UInt_t   reserved = 0;
   *this << reserved;

   fWhat  = what;
   *this << what;

   fClass      = 0;
   fCompress   = 0;
   fBufComp    = 0;
   fBufCompCur = 0;
   fCompPos    = 0;
   fInfos      = 0;
   fEvolution  = kFALSE;

   SetBit(kCannotHandleMemberWiseStreaming);
}

//______________________________________________________________________________
TMessage::TMessage(void *buf, Int_t bufsize) : TBufferFile(TBuffer::kRead, bufsize, buf)
{
   // Create a TMessage object for reading objects. The objects will be
   // read from buf. Use the What() method to get the message type.

   // skip space at the beginning of the message reserved for the message length
   fBufCur += sizeof(UInt_t);

   *this >> fWhat;

   fCompress   = 0;
   fBufComp    = 0;
   fBufCompCur = 0;
   fCompPos    = 0;
   fInfos      = 0;
   fEvolution  = kFALSE;

   if (fWhat & kMESS_ZIP) {
      // if buffer has kMESS_ZIP set, move it to fBufComp and uncompress
      fBufComp    = fBuffer;
      fBufCompCur = fBuffer + bufsize;
      fBuffer     = 0;
      Uncompress();
   }

   if (fWhat == kMESS_OBJECT) {
      InitMap();
      fClass = ReadClass();     // get first the class stored in message
      SetBufferOffset(sizeof(UInt_t) + sizeof(fWhat));
      ResetMap();
   } else {
      fClass = 0;
   }
}

//______________________________________________________________________________
TMessage::~TMessage()
{
   // Clean up compression buffer.

   delete [] fBufComp;
   delete fInfos;
}

//______________________________________________________________________________
void TMessage::EnableSchemaEvolutionForAll(Bool_t enable)
{
   // Static function enabling or disabling the automatic schema evolution.
   // By default schema evolution support is off.

   fgEvolution = enable;
}

//______________________________________________________________________________
Bool_t TMessage::UsesSchemaEvolutionForAll()
{
   // Static function returning status of global schema evolution.

   return fgEvolution;
}

//______________________________________________________________________________
void TMessage::ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t /* force */)
{
   // Force writing the TStreamerInfo to the message.

   if (fgEvolution || fEvolution) {
      if (!fInfos) fInfos = new TList();
      fInfos->Add(info);
   }
}

//______________________________________________________________________________
void TMessage::Forward()
{
   // Change a buffer that was received into one that can be send, i.e.
   // forward a just received message.

   if (IsReading()) {
      SetWriteMode();
      SetBufferOffset(fBufSize);
      SetBit(kCannotHandleMemberWiseStreaming);

      if (fBufComp) {
         fCompPos = fBufCur;
      }
   }
}

//______________________________________________________________________________
void TMessage::TagStreamerInfo(TVirtualStreamerInfo *info)
{
   // Remember that the StreamerInfo is being used in writing.

   if (fgEvolution || fEvolution) {
      if (!fInfos) fInfos = new TList();
      fInfos->Add(info);
   }
}

//______________________________________________________________________________
void TMessage::Reset()
{
   // Reset the message buffer so we can use (i.e. fill) it again.

   SetBufferOffset(sizeof(UInt_t) + sizeof(fWhat));
   ResetMap();

   if (fBufComp) {
      delete [] fBufComp;
      fBufComp    = 0;
      fBufCompCur = 0;
      fCompPos    = 0;
   }
}

//______________________________________________________________________________
void TMessage::SetLength() const
{
   // Set the message length at the beginning of the message buffer.
   // This method is only called by TSocket::Send().

   if (IsWriting()) {
      char *buf = Buffer();
      tobuf(buf, (UInt_t)(Length() - sizeof(UInt_t)));

      if (fBufComp) {
         buf = fBufComp;
         tobuf(buf, (UInt_t)(CompLength() - sizeof(UInt_t)));
      }
   }
}

//______________________________________________________________________________
void TMessage::SetWhat(UInt_t what)
{
   // Using this method one can change the message type a-posteriory.
   // In case you OR "what" with kMESS_ACK, the message will wait for
   // an acknowledgement from the remote side. This makes the sending
   // process synchronous.

   fWhat = what;

   char *buf = Buffer();
   buf += sizeof(UInt_t);   // skip reserved length space
   tobuf(buf, what);

   if (fBufComp) {
      buf = fBufComp;
      buf += sizeof(UInt_t);   // skip reserved length space
      tobuf(buf, what | kMESS_ZIP);
   }
}

//______________________________________________________________________________
void TMessage::SetCompressionAlgorithm(Int_t algorithm)
{
   if (algorithm < 0 || algorithm >= ROOT::kUndefinedCompressionAlgorithm) algorithm = 0;
   Int_t newCompress;
   if (fCompress < 0) {
      newCompress = 100 * algorithm + 1;
   } else {
      int level = fCompress % 100;
      newCompress = 100 * algorithm + level;
   }
   if (newCompress != fCompress && fBufComp) {
      delete [] fBufComp;
      fBufComp    = 0;
      fBufCompCur = 0;
      fCompPos    = 0;
   }
   fCompress = newCompress;
}

//______________________________________________________________________________
void TMessage::SetCompressionLevel(Int_t level)
{
   if (level < 0) level = 0;
   if (level > 99) level = 99;
   Int_t newCompress;
   if (fCompress < 0) {
      newCompress = level;
   } else {
      int algorithm = fCompress / 100;
      if (algorithm >= ROOT::kUndefinedCompressionAlgorithm) algorithm = 0;
      newCompress = 100 * algorithm + level;
   }
   if (newCompress != fCompress && fBufComp) {
      delete [] fBufComp;
      fBufComp    = 0;
      fBufCompCur = 0;
      fCompPos    = 0;
   }
   fCompress = newCompress;
}

//______________________________________________________________________________
void TMessage::SetCompressionSettings(Int_t settings)
{
   if (settings != fCompress && fBufComp) {
      delete [] fBufComp;
      fBufComp    = 0;
      fBufCompCur = 0;
      fCompPos    = 0;
   }
   fCompress = settings;
}

//______________________________________________________________________________
Int_t TMessage::Compress()
{
   // Compress the message. The message will only be compressed if the
   // compression level > 0 and the if the message is > 256 bytes.
   // Returns -1 in case of error (when compression fails or
   // when the message increases in size in some pathological cases),
   // otherwise returns 0.

   Int_t compressionLevel = GetCompressionLevel();
   Int_t compressionAlgorithm = GetCompressionAlgorithm();
   if (compressionLevel <= 0) {
      // no compression specified
      if (fBufComp) {
         delete [] fBufComp;
         fBufComp    = 0;
         fBufCompCur = 0;
         fCompPos    = 0;
      }
      return 0;
   }

   if (fBufComp && fCompPos == fBufCur) {
      // the message was already compressed
      return 0;
   }

   // remove any existing compressed buffer before compressing modified message
   if (fBufComp) {
      delete [] fBufComp;
      fBufComp    = 0;
      fBufCompCur = 0;
      fCompPos    = 0;
   }

   if (Length() <= (Int_t)(256 + 2*sizeof(UInt_t))) {
      // this message is too small to be compressed
      return 0;
   }

   Int_t hdrlen   = 2*sizeof(UInt_t);
   Int_t messlen  = Length() - hdrlen;
   Int_t nbuffers = 1 + (messlen - 1) / kMAXZIPBUF;
   Int_t chdrlen  = 3*sizeof(UInt_t);   // compressed buffer header length
   Int_t buflen   = TMath::Max(512, chdrlen + messlen + 9*nbuffers);
   fBufComp       = new char[buflen];
   char *messbuf  = Buffer() + hdrlen;
   char *bufcur   = fBufComp + chdrlen;
   Int_t noutot   = 0;
   Int_t nzip     = 0;
   Int_t nout, bufmax;
   for (Int_t i = 0; i < nbuffers; ++i) {
      if (i == nbuffers - 1)
         bufmax = messlen - nzip;
      else
         bufmax = kMAXZIPBUF;
      R__zipMultipleAlgorithm(compressionLevel, &bufmax, messbuf, &bufmax, bufcur, &nout, compressionAlgorithm);
      if (nout == 0 || nout >= messlen) {
         //this happens when the buffer cannot be compressed
         delete [] fBufComp;
         fBufComp    = 0;
         fBufCompCur = 0;
         fCompPos    = 0;
         return -1;
      }
      bufcur  += nout;
      noutot  += nout;
      messbuf += kMAXZIPBUF;
      nzip    += kMAXZIPBUF;
   }
   fBufCompCur = bufcur;
   fCompPos    = fBufCur;

   bufcur = fBufComp;
   tobuf(bufcur, (UInt_t)(CompLength() - sizeof(UInt_t)));
   Int_t what = fWhat | kMESS_ZIP;
   tobuf(bufcur, what);
   tobuf(bufcur, Length());    // original uncompressed buffer length

   return 0;
}

//______________________________________________________________________________
Int_t TMessage::Uncompress()
{
   // Uncompress the message. The message will only be uncompressed when
   // kMESS_ZIP is set. Returns -1 in case of error, 0 otherwise.

   if (!fBufComp || !(fWhat & kMESS_ZIP))
      return -1;

   Int_t buflen;
   Int_t hdrlen = 2*sizeof(UInt_t);
   char *bufcur1 = fBufComp + hdrlen;
   frombuf(bufcur1, &buflen);
   UChar_t *bufcur = (UChar_t*)bufcur1;

   /* early consistency check */
   Int_t nin, nbuf;
   if(R__unzip_header(&nin, bufcur, &nbuf)!=0) {
      Error("Uncompress", "Inconsistency found in header (nin=%d, nbuf=%d)", nin, nbuf);
      return -1;
   }

   fBuffer  = new char[buflen];
   fBufSize = buflen;
   fBufCur  = fBuffer + sizeof(UInt_t) + sizeof(fWhat);
   fBufMax  = fBuffer + fBufSize;
   char *messbuf = fBuffer + hdrlen;

   Int_t nout;
   Int_t noutot = 0;
   while (1) {
      Int_t hc = R__unzip_header(&nin, bufcur, &nbuf);
      if (hc!=0) break;
      R__unzip(&nin, bufcur, &nbuf, (unsigned char*) messbuf, &nout);
      if (!nout) break;
      noutot += nout;
      if (noutot >= buflen - hdrlen) break;
      bufcur  += nin;
      messbuf += nout;
   }

   fWhat &= ~kMESS_ZIP;
   fCompress = 1;

   return 0;
}

//______________________________________________________________________________
void TMessage::WriteObject(const TObject *obj)
{
   // Write object to message buffer.
   // When support for schema evolution is enabled the list of TStreamerInfo
   // used to stream this object is kept in fInfos. This information is used
   // by TSocket::Send that sends this list through the socket. This list is in
   // turn used by TSocket::Recv to store the TStreamerInfo objects in the
   // relevant TClass in case the TClass does not know yet about a particular
   // class version. This feature is implemented to support clients and servers
   // with either different ROOT versions or different user classes versions.

   if (fgEvolution || fEvolution) {
      if (fInfos)
         fInfos->Clear();
      else
         fInfos = new TList();
   }

   fBitsPIDs.ResetAllBits();
   WriteObjectAny(obj, TObject::Class());
}

//______________________________________________________________________________
UShort_t TMessage::WriteProcessID(TProcessID *pid)
{
   // Check if the ProcessID pid is already in the message.
   // If not, then:
   //   - mark bit 0 of fBitsPIDs to indicate that a ProcessID has been found
   //   - mark bit uid+1 where uid id the uid of the ProcessID

   if (fBitsPIDs.TestBitNumber(0)) return 0;
   if (!pid)
      pid = TProcessID::GetPID();
   if (!pid) return 0;
   fBitsPIDs.SetBitNumber(0);
   UInt_t uid = pid->GetUniqueID();
   fBitsPIDs.SetBitNumber(uid+1);
   return 1;
}
