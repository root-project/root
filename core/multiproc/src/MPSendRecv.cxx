#include "MPSendRecv.h"
#include "TBufferFile.h"
#include "EMPCode.h"
#include <memory>

//////////////////////////////////////////////////////////////////////////
/// Send a message through socket s with the specified code.
/// For readability, codes should be enumerated as in EMPCode.
/// false is returned on error, true otherwise.
int MPSend(TSocket *s, unsigned code)
{
   TBufferFile wBuf(TBuffer::kWrite);
   wBuf.WriteUInt(code);
   wBuf.WriteULong(0);
   return s->SendRaw(wBuf.Buffer(), wBuf.Length());
}


//////////////////////////////////////////////////////////////////////////
/// Receive message from the socket.
MPCodeBufPair MPRecv(TSocket *s)
{
   char* rawbuf = new char[sizeof(UInt_t)];
   //receive message code
   unsigned nBytes = s->RecvRaw(rawbuf, sizeof(UInt_t));
   if (nBytes == 0) {
      return std::make_pair(EMPCode::kRecvError, nullptr);
   }
   //read message code
   TBufferFile bufReader(TBuffer::kRead, sizeof(UInt_t), rawbuf, false);
   unsigned code;
   bufReader.ReadUInt(code);
   delete [] rawbuf;

   //receive object size
   rawbuf = new char[sizeof(ULong_t)];
   s->RecvRaw(rawbuf, sizeof(ULong_t));
   bufReader.SetBuffer(rawbuf, sizeof(ULong_t), false);
   ULong_t classBufSize;
   bufReader.ReadULong(classBufSize);
   delete [] rawbuf;

   //receive object if needed
   std::shared_ptr<TBufferFile> objBuf; //defaults to nullptr
   if(classBufSize != 0) {
      char *classBuf = new char[classBufSize];
      s->RecvRaw(classBuf, classBufSize);
      objBuf.reset(new TBufferFile(TBuffer::kRead, classBufSize, classBuf, true)); //the buffer is deleted by TBuffer's dtor
   }

   return std::make_pair(code, objBuf);
}
