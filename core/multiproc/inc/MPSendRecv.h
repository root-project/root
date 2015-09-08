#ifndef ROOT_MPSendRecv
#define ROOT_MPSendRecv

#include <typeinfo>
#include <utility> //pair
#include <memory> //shared_ptr
#include <iostream>
#include <type_traits>
#include "TSocket.h"
#include "TClass.h"
#include "TBufferFile.h"

/// An std::pair containing the message contents.
/// first entry is the message code, second entry is a smart pointer.
/// This pointer is null if the message only contained a code,
/// otherwise it points to a TBufferFile containing the message object.
using MPCodeBufPair = std::pair<unsigned, std::shared_ptr<TBufferFile>>;

//MPSend
/// This standalone function can be used to send a code and possibly
/// an object on a given socket. MPSend does not check whether the
/// socket connection is in a valid state. The object is written in
/// a TBufferFile that can be retrieved using MPRecv. See its
/// documentation for methods to retrieve the object from the buffer.
/// Note that only objects the headers of which have been parsed by
/// cling can be sent. User-defined types can be made available to 
/// cling via a call like gSystem->ProcessLine("#include \"header.h\"").
/// Pointer types cannot be sent (with the exception of const char*).
/// The number of bytes sent is returned, as per TSocket::SendRaw.
//There are several versions of this function: this is one sends a 
//message with a code and no object. The templated versions are used
//to send a code and an object of any non-pointer type.
//the others have signature template<T> MPSend(TSocket*, unsigned, T)
//and are specialized for different types of T
int MPSend(TSocket *s, unsigned code);

//MPRecv
/// This standalone function can be used to read a message that
/// has been sent via MPSend. It returns an std::pair containing
/// the message code and a smart pointer. The pointer is null if
/// the message does not contain an object, otherwise it points
/// to a TBufferFile. To retrieve the object from the buffer
/// different methods must be used depending on the type of the
/// object to be read:\n
/// non-pointer built-in types: TBufferFile::operator>> must be used\n
/// c-strings: TBufferFile::ReadString must be used\n
/// class types: TBufferFile::ReadObjectAny must be used.
MPCodeBufPair MPRecv(TSocket *s);


/************ TEMPLATE METHOD IMPLEMENTATIONS *******************/

//////////////////////////////////////////////////////////////////////////
/// Send a message with a code and an object to socket s.
/// The number of bytes sent is returned, as per TSocket::SendRaw.
template<class T, typename std::enable_if<std::is_class<T>::value>::type* = nullptr>
int MPSend(TSocket *s, unsigned code, T obj)
{
   TClass *c = TClass::GetClass(typeid(T));
   if(!c) {
      std::cerr << "[E] Could not find cling definition for class " << typeid(T).name() << "\n";
      return -1;
   }
   TBufferFile objBuf(TBuffer::kWrite);
   objBuf.WriteObjectAny(&obj, c); 
   TBufferFile wBuf(TBuffer::kWrite);
   wBuf.WriteUInt(code);
   wBuf.WriteULong(objBuf.Length());
   wBuf.WriteBuf(objBuf.Buffer(), objBuf.Length());
   return s->SendRaw(wBuf.Buffer(), wBuf.Length());
}

/// \cond
// send a built-in type that is not a pointer (under the hypothesis that
// TBuffer's operator<< works with any built-in type that is not a pointer)
template<class T, typename std::enable_if<!std::is_class<T>::value && !std::is_pointer<T>::value>::type* = nullptr>
int MPSend(TSocket *s, unsigned code, T obj)
{
   TBufferFile wBuf(TBuffer::kWrite);
   ULong_t size = sizeof(T);
   wBuf << code << size << obj;
   return s->SendRaw(wBuf.Buffer(), wBuf.Length());
}

// send an null-terminated c-string or an std::string (which is converted to a c-string)
//TODO can this become a partial specialization instead?
template<class T, typename std::enable_if<std::is_same<const char *, T>::value>::type* = nullptr>
int MPSend(TSocket *s, unsigned code, T obj)
{
   TBufferFile wBuf(TBuffer::kWrite);
   wBuf.WriteUInt(code);
   wBuf.WriteULong(strlen(obj)+1); //strlen does not count the trailing \0
   wBuf.WriteString(obj);
   return s->SendRaw(wBuf.Buffer(), wBuf.Length());
}

// send a TObject*. Allows polymorphic behaviour and pters to derived classes
template<class T, typename std::enable_if<std::is_pointer<T>::value && std::is_constructible<TObject *, T>::value>::type* = nullptr>
int MPSend(TSocket *s, unsigned code, T obj)
{
   //find out the size of the object
   TBufferFile objBuf(TBuffer::kWrite);
   objBuf.WriteObjectAny(obj, obj->IsA());
   //write everything together in a buffer
   TBufferFile wBuf(TBuffer::kWrite);
   wBuf.WriteUInt(code);
   wBuf.WriteULong(objBuf.Length());
   wBuf.WriteBuf(objBuf.Buffer(), objBuf.Length());
   return s->SendRaw(wBuf.Buffer(), wBuf.Length());
}

/// \endcond
#endif
