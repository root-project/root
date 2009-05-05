/* /% C+ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file strbuf.cxx
 ************************************************************************
 * Description:
 * String object with re-used buffers
 ************************************************************************
 * Copyright(c) 1995~2008  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "strbuf.h"
#include "math.h"
#include "stdio.h"
#include <map>

#define G__STRBUF_RESERVOIR_CHUNKSIZE  1024
#define G__STRBUF_RESERVOIR_NUMBUFFERS 32
#define G__STRBUF_RESERVOIR_NUMBUCKETS 1024*10
namespace Cint {
   namespace Internal {
      //___________________________________________________________________
      //
      // Reservoir for allocated and unused buffers.
      // When requesting a buffer of a certain size,
      // a buffer >= that size is returned (or 0 if
      // there is no available buffer). The buffers
      // are kept in a fixed-size array (size-dimension,
      // G__STRBUF_RESERVOIR_NUMBUFFERS) of a fixed
      // size array (available buffers dimension,
      // G__STRBUF_RESERVOIR_NUMBUCKETS) of char*.
      // The size-to-index mapping is done logarithmically,
      // which increases the probability of re-using buffers,
      // and allows the size-dimenion array to span a large
      // amount of possible buffer sizes (up to 
      // (1<<32)*1024bytes for 
      // G__STRBUF_RESERVOIR_CHUNKSIZE == 1024,
      // G__STRBUF_RESERVOIR_NUMBUFFERS == 32.
      class G__BufferReservoir {
      public:
         class Bucket {
         public:
            Bucket():
               fWatermark(fBuffers + G__STRBUF_RESERVOIR_NUMBUFFERS) {}
            ~Bucket() {
               // delete all buffers
               char* buf;
               while ((buf = pop()))
                  delete [] buf;
            }

            bool push(char* buf) {
               if (fWatermark == fBuffers) return false;
               *(--fWatermark) = buf;
               return true;
            }

            char* pop() {
               if (fWatermark < fBuffers + G__STRBUF_RESERVOIR_NUMBUFFERS)
                  return *(fWatermark++);
               return 0;
            }
         private:
            char*  fBuffers[G__STRBUF_RESERVOIR_NUMBUFFERS]; // array of buffers,
            char** fWatermark; // most recently filled slot
         };

         static int logtwo(int i)  {
            // Return the index of the highest set bit.
            // A fast imprecise version of log(i).
            int j = 0;
            while ( i > 255 ) {
               i = i >> 8;
               j += 8;
            }
            if (i & 0xf0) {
               i = i >> 4;
               j += 4;
            }
            if (i & 12) {
               i = i >> 2;
               j += 2;
            }
            if (i & 2) {
               i = i / 2;
               j += 1;
            }
            return j + i;
         }

         static int bucket(int size)  {
            // get the bucket index for a given buffer size
            int b = (size - 1) / G__STRBUF_RESERVOIR_CHUNKSIZE;
            b = logtwo(b);
            if (b >= G__STRBUF_RESERVOIR_NUMBUCKETS)
               return -1;
            return b;
         }

         bool push(int buck, char* buf) {
            // add buf into buck; return false if there is no space
            if (buck < 0) return false;
            return fMap[buck].push(buf);
         }

         char* pop(int &size) {
            // retrieve a buffer of given size;
            // when returning, size will be the bucket index.
            size = bucket(size);
            if (size < 0) return 0;
            return fMap[size].pop();
         }

         static int allocsize(int size) {
            // Determine the allocation size that should be used for
            // a buffer of size.
            int asize = bucket(size) + 1;
            asize = (1 << asize) / 2 * G__STRBUF_RESERVOIR_CHUNKSIZE;
            return asize;
         }

         static int bucketallocsize(int bucket_index) {
            // Determine the allocation size that is used for
            // the specified bucket index.
            int asize = bucket_index + 1;
            asize = (1 << asize) / 2 * G__STRBUF_RESERVOIR_CHUNKSIZE;
            return asize;
         }
         
      private:
         Bucket fMap[G__STRBUF_RESERVOIR_NUMBUCKETS]; // the buckets
      };
   } // Internal
} // Cint

Cint::Internal::G__StrBuf::~G__StrBuf()
{
   // Give our buffer back to the BufMap, i.e. make it available again.
   if (fBucket < 0 || !GetReservoir().push(fBucket, fBuf)) {
      delete [] fBuf;
   }
}

char* Cint::Internal::G__StrBuf::GetBuf(int &size_then_bucket_index)
{
   // Return a buffer of given size (or larger).
   // If there is one in the map, return that one, otherwise allocatea new one.
   // When entering GetBuf 'size_then_bucket_index' is the requested size in bytes
   // it is then updated to return the corresponding bucket number. 

   int origsize = size_then_bucket_index;
   // Look for an existing bucket and update with the bucket index.
   char* buf = GetReservoir().pop(size_then_bucket_index);
   if (!buf) {
      buf = new char[G__BufferReservoir::allocsize(origsize)];
   }
   return buf;
}

Cint::Internal::G__BufferReservoir& Cint::Internal::G__StrBuf::GetReservoir()
{
   // Return the static BufferReservoir
   static G__BufferReservoir sReservoir;
   return sReservoir;
}

int Cint::Internal::G__StrBuf::FormatArgList(const char *fmt, va_list args)
{
   if (!fmt) {
      fBuf[0] = 0;
      return 0;
   }
   int result = -1;
   int bucket_req = fBucket;
   
   while (result == -1)
   {
      int length = G__BufferReservoir::bucketallocsize(bucket_req);
#ifdef _MSC_VER
      result = _vsnprintf(fBuf, length, fmt, args);
#else
      result = vsnprintf(fBuf, length, fmt, args);
#endif               
      if (result == -1) {
         ++bucket_req;
         ResizeNoCopy( bucket_req );
      }
   }
   return result;
}

int Cint::Internal::G__StrBuf::Format(const char *fmt, ...)
{
   va_list args;
   va_start(args, fmt);
   int res = FormatArgList(fmt, args);
   va_end(args);
   return res;
}

void Cint::Internal::G__StrBuf::ResizeNoCopy(int newbucket)
{
   // Extend the size used by this buffer to at least newsize.
   // This does NOT copy the content.

   if (newbucket > fBucket) {
   
      int newsize_then_index = G__BufferReservoir::bucketallocsize(newbucket);
      char *newbuf = GetBuf(newsize_then_index);
      
      if (fBucket < 0 || !GetReservoir().push(fBucket, fBuf))
         delete [] fBuf;
      
      fBuf = newbuf;
      fBucket = newsize_then_index;
   }
}

