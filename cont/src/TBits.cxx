// @(#)root/base:$Name:  $:$Id: TBits.cxx,v 1.3 2001/02/09 16:47:51 brun Exp $
// Author: Philippe Canal 05/02/2001
//    Feb  5 2001: Creation
//    Feb  6 2001: Changed all int to unsigned int.
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
// The size of the container is automatically extended when a bit       //
// number is either set or tested.  To reduce the memory size of the    //
// container use the Compact function, this will discard the memory     //
// occupied by the upper bits that are 0.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBits.h"
#include "string.h"

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
   
   UInt_t needed;
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
UInt_t TBits::CountBits() const
{
   // Return number of bits set to 1
   
   UInt_t count = 0;
   // The following loop might be going a little to far.  However we are
   // guaranteed that any bit above fNbits is 0 (the default value) and thus
   // can not count.  [We also assume that the extra few shift operation are
   // faster that the extra code and the extra modulo operation that would be
   // needed to check only the active bits.
   for(UInt_t i=0; i<fNbytes; i++) {
       UChar_t val = fAllBits[i];
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
   for(UInt_t i=0; i<fNbytes; i++) {
      UChar_t val = fAllBits[i];
      for (UInt_t j=0; j<8; j++) {
         if (val & 1) printf(" bit:%4d = 1\n",count);
         count++;
         val = val >> 1;
      }
   }
}

//______________________________________________________________________________
void TBits::ResetAllBits(Bool_t value) 
{
   // Reset all bits to 0 (false)

   memset(fAllBits,0,fNbytes);
}

//______________________________________________________________________________
void TBits::SetBitNumber(UInt_t bitnumber, Bool_t value) 
{
   // Set bit number 'bitnumber' to be value

   if (bitnumber >= fNbits) {
      UInt_t new_size = (bitnumber/8) + 1;
      if (new_size > fNbytes) {
         UChar_t *old_location = fAllBits;
         fAllBits = new UChar_t[new_size];
         memcpy(fAllBits,old_location,fNbytes);
         fNbytes = new_size;
         delete old_location;
      } 
      fNbits = bitnumber+1;
   }
   UInt_t  loc = bitnumber/8;
   UChar_t bit = bitnumber%8;
   if (value) 
      fAllBits[loc] |= (1<<bit);
   else 
      fAllBits[loc] &= (0xFF ^ (1<<bit));
}

//______________________________________________________________________________
Bool_t TBits::TestBitNumber(UInt_t bitnumber) const
{
   // Return the current value of the bit

   if (bitnumber >= fNbits) return kFALSE;
   UInt_t  loc = bitnumber/8;
   UChar_t value = fAllBits[loc];
   UChar_t bit = bitnumber%8;
   Bool_t  result = (value & (1<<bit)) != 0;
   return result;
   // short: return 0 != (fAllBits[bitnumber/8] & (1<< (bitnumber%8)));
}

