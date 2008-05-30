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

         int logtwo(int i) const {
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

         int bucket(int size) const {
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

         int allocsize(int size) {
            // determine the allocation size that should be used for
            // a buffer of size.
            int asize = bucket(size) + 1;
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
   if (fSize < 0 || !GetReservoir().push(fSize, fBuf))
      delete [] fBuf;
}

char* Cint::Internal::G__StrBuf::GetBuf(int &size)
{
   // Return a buffer of given size (or larger).
   // If there is one in the map, return that one, otherwise allocatea new one.

   int origsize = size;
   char* buf = GetReservoir().pop(size);
   if (!buf)
      buf = new char[GetReservoir().allocsize(origsize)];
   return buf;
}

Cint::Internal::G__BufferReservoir& Cint::Internal::G__StrBuf::GetReservoir()
{
   // Return the static BufferReservoir
   static G__BufferReservoir sReservoir;
   return sReservoir;
}
