// @(#)root/base:$Name:  $:$Id: TBits.cxx,v 1.1 2001/02/08 16:58:22 brun Exp $
// Author: Philippe Canal 05/02/2001
//    Feb  5 2001: Creation
//______________________________________________________________________________

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
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBits.h"
#include "string.h"

ClassImp(TBits)

//______________________________________________________________________________
TBits::TBits(Int_t nbits) : fNbits(nbits) 
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
      fAllBits = new UChar_t[fNbytes];
      memcpy(fAllBits,rhs.fAllBits,fNbytes);
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
void TBits::Compact() 
{
   // Reduce the storage used by the object to a minimun
   
   Int_t needed;
   for(needed=fNbytes-1; 
       needed > 0 && fAllBits[needed]==0; ) { needed--; };
   needed++;

   if (needed!=fNbytes) {
      UChar_t *old_location = fAllBits;
      fAllBits = new UChar_t[needed];

      memcpy(fAllBits,old_location,needed);
      delete old_location;

      fNbytes = needed;
      fNbits = 8*fNbytes;
   }
}

//______________________________________________________________________________
Int_t TBits::CountBits() const
{
   // Return number of bits set to 1
   
   Int_t count = 0;
   for(Int_t i=0; i<fNbytes; i++) {
      Int_t val = fAllBits[i];
      for (Int_t j=0; j<8; j++) {
         count += val & 1;
         val = val >> 1;
      }
   }
   return count;
}

//______________________________________________________________________________
void TBits::Paint(Option_t *option) 
{
   // Once implemented, it will draw the bit field as an histogram.
   // use the TVirtualPainter as the usual trick

}

//______________________________________________________________________________
void TBits::Print(Option_t *option) const
{
   // Print the list of active bits
   
   Int_t count = 0;
   for(Int_t i=0; i<fNbytes; i++) {
      Int_t val = fAllBits[i];
      for (Int_t j=0; j<8; j++) {
         if (val & 1) printf(" bit:%4d = 1\n",count);
         count++;
         val = val >> 1;
      }
   }
}

//______________________________________________________________________________
void TBits::ResetAllBits(Int_t value) 
{
   // Reset all bits to 0 (false)

   memset(fAllBits,0,fNbytes);
}

//______________________________________________________________________________
void TBits::SetBitNumber(Int_t bitnumber, Bool_t value) 
{
   // Set bit number 'bitnumber' to be value

   if (bitnumber+1>fNbits) {
      int new_size = (bitnumber/8) + 1;
      if (new_size > fNbytes) {
         UChar_t *old_location = fAllBits;
         fAllBits = new UChar_t[new_size];
         memcpy(fAllBits,old_location,fNbytes);
         fNbytes = new_size;
         delete old_location;
      } 
      fNbits = bitnumber+1;
   }
   int loc = bitnumber/8;
   int bit = bitnumber%8;
   if (value) 
      fAllBits[loc] |= (1<<bit);
   else 
      fAllBits[loc] &= (0xFF ^ (1<<bit));
}

//______________________________________________________________________________
Bool_t TBits::TestBitNumber(Int_t bitnumber) const
{
   // Return the current value of the bit

   if (bitnumber>fNbits) return kFALSE;
   int loc = bitnumber/8;
   int value = fAllBits[loc];
   int bit = bitnumber%8;
   Bool_t result = (value & (1<<bit)) != 0;
   return result;
   // short: return 0 != (fAllBits[bitnumber/8] & (1<< (bitnumber%8)));
}

