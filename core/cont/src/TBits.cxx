// @(#)root/cont:$Id$
// Author: Philippe Canal 05/02/2001
//    Feb  5 2001: Creation
//    Feb  6 2001: Changed all int to unsigned int.

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBits                                                                //
//                                                                      //
// Container of bits                                                    //
//                                                                      //
// This class provides a simple container of bits.                      //
// Each bit can be set and tested via the functions SetBitNumber and    //
// TestBitNumber.                                             .         //
// The default value of all bits is kFALSE.                             //
// The size of the container is automatically extended when a bit       //
// number is either set or tested.  To reduce the memory size of the    //
// container use the Compact function, this will discard the memory     //
// occupied by the upper bits that are 0.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBits.h"
#include "string.h"
#include "Riostream.h"

ClassImp(TBits)

//______________________________________________________________________________
TBits::TBits(UInt_t nbits) : fNbits(nbits)
{
   // TBits constructor.  All bits set to 0

   if (nbits <= 0) nbits = 8;
   fNbytes  = ((nbits-1)/8) + 1;
   fAllBits = new UChar_t[fNbytes];
   // this is redundant only with libNew
   memset(fAllBits,0,fNbytes);
}

//______________________________________________________________________________
TBits::TBits(const TBits &original) : TObject(original), fNbits(original.fNbits),
   fNbytes(original.fNbytes)
{
   // TBits copy constructor

   fAllBits = new UChar_t[fNbytes];
   memcpy(fAllBits,original.fAllBits,fNbytes);

}

//______________________________________________________________________________
TBits& TBits::operator=(const TBits& rhs)
{
   // TBits assignment operator

   if (this != &rhs) {
      TObject::operator=(rhs);
      fNbits   = rhs.fNbits;
      fNbytes  = rhs.fNbytes;
      delete [] fAllBits;
      if (fNbytes != 0) {
         fAllBits = new UChar_t[fNbytes];
         memcpy(fAllBits,rhs.fAllBits,fNbytes);
      } else {
         fAllBits = 0;
      }
   }
   return *this;
}

//______________________________________________________________________________
TBits::~TBits()
{
   // TBits destructor

   delete [] fAllBits;
}

//______________________________________________________________________________
void TBits::Clear(Option_t * /*option*/)
{
   // Clear the value.

   delete [] fAllBits;
   fAllBits = 0;
   fNbits   = 0;
   fNbytes  = 0;
}

//______________________________________________________________________________
void TBits::Compact()
{
   // Reduce the storage used by the object to a minimun

   if (!fNbits || !fAllBits) return;
   UInt_t needed;
   for(needed=fNbytes-1;
       needed > 0 && fAllBits[needed]==0; ) { needed--; };
   needed++;

   if (needed!=fNbytes) {
      UChar_t *old_location = fAllBits;
      fAllBits = new UChar_t[needed];

      memcpy(fAllBits,old_location,needed);
      delete [] old_location;

      fNbytes = needed;
      fNbits = 8*fNbytes;
   }
}

//______________________________________________________________________________
UInt_t TBits::CountBits(UInt_t startBit) const
{
   // Return number of bits set to 1 starting at bit startBit

   static const Int_t nbits[256] = {
             0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
             4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

   UInt_t i,count = 0;
   if (startBit == 0) {
      for(i=0; i<fNbytes; i++) {
         count += nbits[fAllBits[i]];
      }
      return count;
   }
   if (startBit >= fNbits) return count;
   UInt_t startByte = startBit/8;
   UInt_t ibit = startBit%8;
   if (ibit) {
      for (i=ibit;i<8;i++) {
         if (fAllBits[startByte] & (1<<ibit)) count++;
      }
      startByte++;
   }
   for(i=startByte; i<fNbytes; i++) {
      count += nbits[fAllBits[i]];
   }
   return count;
}

//______________________________________________________________________________
void TBits::DoAndEqual(const TBits& rhs)
{
   // Execute (*this) &= rhs;
   // Extra bits in rhs are ignored
   // Missing bits in rhs are assumed to be zero.

   UInt_t min = (fNbytes<rhs.fNbytes) ? fNbytes : rhs.fNbytes;
   for(UInt_t i=0; i<min; ++i) {
      fAllBits[i] &= rhs.fAllBits[i];
   }
   if (fNbytes>min) {
      memset(&(fAllBits[min]),0,fNbytes-min);
   }
}

//______________________________________________________________________________
void TBits::DoOrEqual(const TBits& rhs)
{
   // Execute (*this) &= rhs;
   // Extra bits in rhs are ignored
   // Missing bits in rhs are assumed to be zero.

   UInt_t min = (fNbytes<rhs.fNbytes) ? fNbytes : rhs.fNbytes;
   for(UInt_t i=0; i<min; ++i) {
      fAllBits[i] |= rhs.fAllBits[i];
   }
}

//______________________________________________________________________________
void TBits::DoXorEqual(const TBits& rhs)
{
   // Execute (*this) ^= rhs;
   // Extra bits in rhs are ignored
   // Missing bits in rhs are assumed to be zero.

   UInt_t min = (fNbytes<rhs.fNbytes) ? fNbytes : rhs.fNbytes;
   for(UInt_t i=0; i<min; ++i) {
      fAllBits[i] ^= rhs.fAllBits[i];
   }
}

//______________________________________________________________________________
void TBits::DoFlip()
{
   // Execute ~(*this)

   for(UInt_t i=0; i<fNbytes; ++i) {
      fAllBits[i] = ~fAllBits[i];
   }
   // NOTE: out-of-bounds bit were also flipped!
}

//______________________________________________________________________________
void TBits::DoLeftShift(UInt_t shift)
{
   // Execute the left shift operation.

   if (shift==0) return;
   const UInt_t wordshift = shift / 8;
   const UInt_t offset = shift % 8;
   if (offset==0) {
      for(UInt_t n = fNbytes - 1; n >= wordshift; --n) {
         fAllBits[n] = fAllBits[ n - wordshift ];
      }
   } else {
      const UInt_t sub_offset = 8 - offset;
      for(UInt_t n = fNbytes - 1; n > wordshift; --n) {
         fAllBits[n] = (fAllBits[n - wordshift] << offset) |
                       (fAllBits[n - wordshift - 1] >> sub_offset);
      }
      fAllBits[wordshift] = fAllBits[0] << offset;
   }
   memset(fAllBits,0,wordshift);
}

//______________________________________________________________________________
void TBits::DoRightShift(UInt_t shift)
{
   // Execute the left shift operation.

   if (shift==0) return;
   const UInt_t wordshift = shift / 8;
   const UInt_t offset = shift % 8;
   const UInt_t limit = fNbytes - wordshift - 1;

   if (offset == 0)
      for (UInt_t n = 0; n <= limit; ++n)
         fAllBits[n] = fAllBits[n + wordshift];
   else
   {
      const UInt_t sub_offset = 8 - offset;
      for (UInt_t n = 0; n < limit; ++n)
         fAllBits[n] = (fAllBits[n + wordshift] >> offset) |
                       (fAllBits[n + wordshift + 1] << sub_offset);
      fAllBits[limit] = fAllBits[fNbytes-1] >> offset;
   }

   memset(&(fAllBits[limit + 1]),0, fNbytes - limit - 1);
}

//______________________________________________________________________________
UInt_t TBits::FirstNullBit(UInt_t startBit) const
{
   // Return position of first null bit (starting from position 0 and up)

   static const Int_t fbits[256] = {
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,6,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,7,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,6,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,5,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,4,
             0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,8};

   UInt_t i;
   if (startBit == 0) {
      for(i=0; i<fNbytes; i++) {
         if (fAllBits[i] != 255) return 8*i + fbits[fAllBits[i]];
      }
      return fNbits;
   }
   if (startBit >= fNbits) return fNbits;
   UInt_t startByte = startBit/8;
   UInt_t ibit = startBit%8;
   if (ibit) {
      for (i=ibit;i<8;i++) {
         if ((fAllBits[startByte] & (1<<i)) == 0) return 8*startByte+i;
      }
      startByte++;
   }
   for(i=startByte; i<fNbytes; i++) {
      if (fAllBits[i] != 255) return 8*i + fbits[fAllBits[i]];
   }
   return fNbits;
}

//______________________________________________________________________________
UInt_t TBits::FirstSetBit(UInt_t startBit) const
{
   // Return position of first non null bit (starting from position 0 and up)

   static const Int_t fbits[256] = {
             8,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             7,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             6,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             5,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0,
             4,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0};

   UInt_t i;
   if (startBit == 0) {
      for(i=0; i<fNbytes; i++) {
         if (fAllBits[i] != 0) return 8*i + fbits[fAllBits[i]];
      }
      return fNbits;
   }
   if (startBit >= fNbits) return fNbits;
   UInt_t startByte = startBit/8;
   UInt_t ibit = startBit%8;
   if (ibit) {
      for (i=ibit;i<8;i++) {
         if ((fAllBits[startByte] & (1<<i)) != 0) return 8*startByte+i;
      }
      startByte++;
   }
   for(i=startByte; i<fNbytes; i++) {
      if (fAllBits[i] != 0) return 8*i + fbits[fAllBits[i]];
   }
   return fNbits;
}

//______________________________________________________________________________
void TBits::Output(ostream &os) const
{
   // Print the value to the ostream

   for(UInt_t i=0; i<fNbytes; ++i) {
      UChar_t val = fAllBits[fNbytes - 1 - i];
      for (UInt_t j=0; j<8; ++j) {
         os << (Bool_t)(val&0x80);
         val <<= 1;
      }
   }
}

//______________________________________________________________________________
void TBits::Paint(Option_t *)
{
   // Once implemented, it will draw the bit field as an histogram.
   // use the TVirtualPainter as the usual trick

}

//______________________________________________________________________________
void TBits::Print(Option_t *) const
{
   // Print the list of active bits

   Int_t count = 0;
   for(UInt_t i=0; i<fNbytes; ++i) {
      UChar_t val = fAllBits[i];
      for (UInt_t j=0; j<8; ++j) {
         if (val & 1) printf(" bit:%4d = 1\n",count);
         count++;
         val = val >> 1;
      }
   }
}

//______________________________________________________________________________
void TBits::ResetAllBits(Bool_t)
{
   // Reset all bits to 0 (false).

   if (fAllBits) memset(fAllBits,0,fNbytes);
}

//______________________________________________________________________________
void TBits::ReserveBytes(UInt_t nbytes)
{
   // Reverse each bytes.

   if (nbytes > fNbytes) {
      // do it in this order to remain exception-safe.
      UChar_t *newBits=new UChar_t[nbytes];
      delete[] fAllBits;
      fNbytes=nbytes;
      fAllBits=newBits;
   }
}

//______________________________________________________________________________
void TBits::Set(UInt_t nbits, const Char_t *array)
{
   // Set all the bytes.

   UInt_t nbytes=(nbits+7)>>3;

   ReserveBytes(nbytes);

   fNbits=nbits;
   memcpy(fAllBits, array, nbytes);
}

//______________________________________________________________________________
void TBits::Get(Char_t *array) const
{
   // Copy all the byes.

   memcpy(array, fAllBits, (fNbits+7)>>3);
}

 #ifdef R__BYTESWAP  /* means we are on little endian */

 /*
 If we are on a little endian machine, a bitvector represented using
 any integer type is identical to a bitvector represented using bytes.
 -- FP.
 */

void TBits::Set(UInt_t nbits, const Short_t *array)
{
   // Set all the bytes.

   Set(nbits, (const Char_t*)array);
}

void TBits::Set(UInt_t nbits, const Int_t *array)
{
   // Set all the bytes.

   Set(nbits, (const Char_t*)array);
}

void TBits::Set(UInt_t nbits, const Long64_t *array)
{
   // Set all the bytes.

   Set(nbits, (const Char_t*)array);
}

void TBits::Get(Short_t *array) const
{
   // Get all the bytes.

   Get((Char_t*)array);
}

void TBits::Get(Int_t *array) const
{
   // Get all the bytes.

   Get((Char_t*)array);
}

void TBits::Get(Long64_t *array) const
{
   // Get all the bytes.

   Get((Char_t*)array);
}

#else

 /*
 If we are on a big endian machine, some swapping around is required.
 */

void TBits::Set(UInt_t nbits, const Short_t *array)
{
   // make nbytes even so that the loop below is neat.
   UInt_t nbytes = ((nbits+15)>>3)&~1;

   ReserveBytes(nbytes);

   fNbits=nbits;

   const UChar_t *cArray = (const UChar_t*)array;
   for (UInt_t i=0; i<nbytes; i+=2) {
      fAllBits[i] = cArray[i+1];
      fAllBits[i+1] = cArray[i];
   }
}

void TBits::Set(UInt_t nbits, const Int_t *array)
{
   // make nbytes a multiple of 4 so that the loop below is neat.
   UInt_t nbytes = ((nbits+31)>>3)&~3;

   ReserveBytes(nbytes);

   fNbits=nbits;

   const UChar_t *cArray = (const UChar_t*)array;
   for (UInt_t i=0; i<nbytes; i+=4) {
      fAllBits[i] = cArray[i+3];
      fAllBits[i+1] = cArray[i+2];
      fAllBits[i+2] = cArray[i+1];
      fAllBits[i+3] = cArray[i];
   }
}

void TBits::Set(UInt_t nbits, const Long64_t *array)
{
   // make nbytes a multiple of 8 so that the loop below is neat.
   UInt_t nbytes = ((nbits+63)>>3)&~7;

   ReserveBytes(nbytes);

   fNbits=nbits;

   const UChar_t *cArray = (const UChar_t*)array;
   for (UInt_t i=0; i<nbytes; i+=8) {
      fAllBits[i] = cArray[i+7];
      fAllBits[i+1] = cArray[i+6];
      fAllBits[i+2] = cArray[i+5];
      fAllBits[i+3] = cArray[i+4];
      fAllBits[i+4] = cArray[i+3];
      fAllBits[i+5] = cArray[i+2];
      fAllBits[i+6] = cArray[i+1];
      fAllBits[i+7] = cArray[i];
   }
}

void TBits::Get(Short_t *array) const
{
   // Get all the bytes.

   UInt_t nBytes = (fNbits+7)>>3;
   UInt_t nSafeBytes = nBytes&~1;

   UChar_t *cArray=(UChar_t*)array;
   for (UInt_t i=0; i<nSafeBytes; i+=2) {
      cArray[i] = fAllBits[i+1];
      cArray[i+1] = fAllBits[i];
   }

   if (nBytes>nSafeBytes) {
      cArray[nSafeBytes+1] = fAllBits[nSafeBytes];
   }
}

void TBits::Get(Int_t *array) const
{
   // Get all the bytes.

   UInt_t nBytes = (fNbits+7)>>3;
   UInt_t nSafeBytes = nBytes&~3;

   UChar_t *cArray=(UChar_t*)array;
   UInt_t i;
   for (i=0; i<nSafeBytes; i+=4) {
      cArray[i] = fAllBits[i+3];
      cArray[i+1] = fAllBits[i+2];
      cArray[i+2] = fAllBits[i+1];
      cArray[i+3] = fAllBits[i];
   }

   for (i=0; i<nBytes-nSafeBytes; ++i) {
      cArray[nSafeBytes + (3 - i)] = fAllBits[nSafeBytes + i];
   }
}

void TBits::Get(Long64_t *array) const
{
   // Get all the bytes.

   UInt_t nBytes = (fNbits+7)>>3;
   UInt_t nSafeBytes = nBytes&~7;

   UChar_t *cArray=(UChar_t*)array;
   UInt_t i;
   for (i=0; i<nSafeBytes; i+=8) {
      cArray[i] = fAllBits[i+7];
      cArray[i+1] = fAllBits[i+6];
      cArray[i+2] = fAllBits[i+5];
      cArray[i+3] = fAllBits[i+4];
      cArray[i+4] = fAllBits[i+3];
      cArray[i+5] = fAllBits[i+2];
      cArray[i+6] = fAllBits[i+1];
      cArray[i+7] = fAllBits[i];
   }

   for (i=0; i<nBytes-nSafeBytes; ++i) {
      cArray[nSafeBytes + (7 - i)] = fAllBits[nSafeBytes + i];
   }
}

#endif

Bool_t TBits::operator==(const TBits &other) const
{
   // Compare object.

   if (fNbits == other.fNbits) {
      return !memcmp(fAllBits, other.fAllBits, (fNbits+7)>>3);
   } else if (fNbits <  other.fNbits) {
      return !memcmp(fAllBits, other.fAllBits, (fNbits+7)>>3) && other.FirstSetBit(fNbits) == other.fNbits;
   } else {
      return !memcmp(fAllBits, other.fAllBits, (other.fNbits+7)>>3) && FirstSetBit(other.fNbits) == fNbits;
   }
}

