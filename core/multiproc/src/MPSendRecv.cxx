/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
 
#include "MPSendRecv.h"
#include "TBufferFile.h"
#include "MPCode.h"
#include <memory> //unique_ptr

//////////////////////////////////////////////////////////////////////////
/// Send a message with the specified code on the specified socket.
/// This standalone function can be used to send a code
/// on a given socket. It does not check whether the socket connection is
/// in a valid state. The message code can then be retrieved via MPRecv().\n
/// **Note:** only objects the headers of which have been parsed by
/// cling can be sent by MPSend(). User-defined types can be made available to
/// cling via a call like `gSystem->ProcessLine("#include \"header.h\"")`.
/// Pointer types are not supported (with the exception of const char*),
/// but the user can simply dereference the pointer and send the
/// pointed object instead.\n
/// **Note:** for readability, codes should be enumerated as in EMPCode.\n
/// \param s a pointer to a valid TSocket. No validity checks are performed\n
/// \param code the code to be sent
/// \return the number of bytes sent, as per TSocket::SendRaw
int MPSend(TSocket *s, unsigned code)
{
   TBufferFile wBuf(TBuffer::kWrite);
   wBuf.WriteUInt(code);
   wBuf.WriteULong(0);
   return s->SendRaw(wBuf.Buffer(), wBuf.Length());
}


//////////////////////////////////////////////////////////////////////////
/// Receive message from a socket.
/// This standalone function can be used to read a message that
/// has been sent via MPSend(). The smart pointer contained in the returned
/// ::MPCodeBufPair is null if the message does not contain an object,
/// otherwise it points to a TBufferFile.
/// To retrieve the object from the buffer different methods must be used
/// depending on the type of the object to be read:\n
/// * non-pointer built-in types: TBufferFile::operator>> must be used\n
/// * c-strings: TBufferFile::ReadString must be used\n
/// * class types: TBufferFile::ReadObjectAny must be used\n
/// \param s a pointer to a valid TSocket. No validity checks are performed\n
/// \return ::MPCodeBufPair, i.e. an std::pair containing message code and (possibly) object
MPCodeBufPair MPRecv(TSocket *s)
{
   char *rawbuf = new char[sizeof(UInt_t)];
   //receive message code
   unsigned nBytes = s->RecvRaw(rawbuf, sizeof(UInt_t));
   if (nBytes == 0) {
      return std::make_pair(MPCode::kRecvError, nullptr);
   }
   //read message code
   TBufferFile bufReader(TBuffer::kRead, sizeof(UInt_t), rawbuf, false);
   unsigned code;
   bufReader.ReadUInt(code);
   delete [] rawbuf;

   //receive object size
   //ULong_t is sent as 8 bytes irrespective of the size of the type
   rawbuf = new char[8];
   s->RecvRaw(rawbuf, 8);
   bufReader.SetBuffer(rawbuf, 8, false);
   ULong_t classBufSize;
   bufReader.ReadULong(classBufSize);
   delete [] rawbuf;

   //receive object if needed
   std::unique_ptr<TBufferFile> objBuf; //defaults to nullptr
   if (classBufSize != 0) {
      char *classBuf = new char[classBufSize];
      s->RecvRaw(classBuf, classBufSize);
      objBuf.reset(new TBufferFile(TBuffer::kRead, classBufSize, classBuf, true)); //the buffer is deleted by TBuffer's dtor
   }

   return std::make_pair(code, std::move(objBuf));
}
