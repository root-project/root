/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file FastAllocString.h
 ************************************************************************
 * Description:
 * String object with fast allocation (pooled memory)
 ************************************************************************
 * Copyright(c) 1995~2009  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__FASTALLOGSTRING_H
#define G__FASTALLOGSTRING_H

#include <stdarg.h>
#include <stddef.h>

// For G__EXPORT
#include "G__ci.h"

namespace Cint {
   namespace Internal {
      class G__BufferReservoir;
   }
}

//_____________________________________________________________
//
// A tiny object representing a char array.
// Create it with the desired size of the char array and it
// will try to retrieve a previsouly allocated buffer from
// an internal resrevoir of buffers, or allocate a new one
// if none is available. This is a lot faster than mallocs /
// free calls for each char array, and it considerably reduces
// the used stack size by functions previsouly using static
// size, stack based chart arrays. It also allows to make the
// buffer size dynamic, adopted e.g. to strlen(expression),
// instead of a value defined at compile time (a la G__LONGBUF).
// When the G__FastAllocString object leaves the scope it will put its
// buffer (back) into the internal buffer reservoir for later
// use by a G__FastAllocString object requesting a same of smaller size
// buffer. This class is optimized for both speed and low memory
// use despite the reservoir.
//
class 
#ifndef __CINT__
G__EXPORT
#endif
G__FastAllocString {
public:
   G__FastAllocString(size_t reqsize = 1024): fBuf(0), fCapacity(reqsize) {
      // GetBuf takes as parameter the size in bytes
      // and modify the parameter (fBucket) to hold the 
      // bucket number.
      fBuf = GetBuf(fCapacity);
   }
   G__FastAllocString(const char* s);
   G__FastAllocString(const G__FastAllocString&);

   ~G__FastAllocString();

   // plenty of char* conversion functions:
   operator char*() { return fBuf; }
   operator const char*() const { return fBuf; }
   const char* operator()() const { return fBuf; }

   // DON'T: these create ambiguities with ::op[char*, int] etc
   //char& operator[](int i) { return fBuf[i]; }
   //char operator[](int i) const { return fBuf[i]; }
   //char* operator+(int i) { return fBuf + i; }
   //const char* operator+(int i) const { return fBuf + i; }

   const char* data() const { return fBuf; }

   int FormatArgList(const char *fmt, va_list args);
   int FormatArgList(size_t offset, const char *fmt, va_list args);
   G__FastAllocString& Format(const char *fmt, ...);
   G__FastAllocString& Format(size_t offset, const char *fmt, ...);

   size_t Capacity() const { return fCapacity; }

   G__FastAllocString& operator=(const G__FastAllocString& s) {
      // Copy s into *this.
      // Cannot rely on operator=(const char*) overload - compiler-generated one wins resolution!
      operator=(s.data());
      return *this;
   }
   G__FastAllocString& operator=(const char*);
   G__FastAllocString& operator+=(const char*);
   G__FastAllocString& Swap(G__FastAllocString&);
   void Resize(size_t cap);

   void Set(size_t pos, char c) {
      // Set character at position pos to c; resize if needed.
      Resize(pos + 1);
      fBuf[pos] = c;
   }
   /*
   size_t Set(size_t& pos, const char* s) {
      // Overwrite string at position pos with s; resize if needed.
      // Return pos incremented by strlen(s)
      size_t len = strlen(s);
      Resize(pos + len + 1);
      memcpy(fBuf + pos, s, len + 1);
      return pos + len;
      }*/

   void Replace(size_t where, const char *replacement);
                
protected:
   static char* GetBuf(size_t &size);

   void ResizeToBucketNoCopy(int newbucket);
   void ResizeNoCopy(size_t cap);
         
private:
   char*  fBuf;    // the buffer
   size_t fCapacity; // measure representing the buffer's size, used by the internal reservoir
};

// Those 6 functions are intentionally not implemented as their are 'illegal'
// and we should call the equivalent member function instead.
void G__strlcpy(G__FastAllocString&, const char *, size_t);
void G__strlcat(G__FastAllocString&, const char *, size_t);
void G__snprintf(G__FastAllocString&, size_t, const char *, ...);
void strcpy(G__FastAllocString&, const char *);
void strcat(G__FastAllocString&, const char *);
void sprintf(G__FastAllocString&, const char *, ...);

#endif // G__FASTALLOGSTRING_H
