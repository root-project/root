
#include "TFPBlock.h"
#include "TStorage.h"
#include <cstdlib>


ClassImp(TFPBlock)

//__________________________________________________________________
//constructor
TFPBlock::TFPBlock(Long64_t* offset, Int_t* length, Int_t nb)
{
   Int_t aux = 0;
  
   fNblock = nb;
   fPos = new Long64_t[nb];
   fLen = new Int_t[nb];
 
   for (Int_t i=0; i < nb; i++){
      fPos[i] = offset[i];
      fLen[i] = length[i];
      aux += length[i]; 
   }
   fFullSize = aux;
   fBuffer = new char[fFullSize];
}

//__________________________________________________________________
//destructor
TFPBlock::~TFPBlock()
{
   delete[] fPos;
   delete[] fLen;
   delete[] fBuffer;
}

//__________________________________________________________________
Long64_t* TFPBlock::GetPos()
{
   // Get pointer to the array of postions.
  
   return fPos;
}

//__________________________________________________________________
Int_t* TFPBlock::GetLen()
{
   // Get pointer to the array of lengths.

   return fLen;
}

//__________________________________________________________________
Int_t TFPBlock::GetFullSize()
{
   // Return size of the block.

   return fFullSize;
}

//__________________________________________________________________
Int_t TFPBlock::GetNoElem()
{
   // Return number of elements in the block.

   return fNblock;
}

//__________________________________________________________________
Long64_t TFPBlock::GetPos(Int_t i)
{
   // Get position of the element at index i.
  
   return fPos[i];
}

//__________________________________________________________________
Int_t TFPBlock::GetLen(Int_t i)
{
  // Get length of the element at index i.

   return fLen[i];
}

//__________________________________________________________________
char* TFPBlock::GetBuffer()
{
   // Get block buffer.

   return fBuffer;
}

//__________________________________________________________________
void TFPBlock::SetBuffer(char* buf)
{
   //Set block buffer.
  
   fBuffer = buf;
}

//__________________________________________________________________
void TFPBlock::ReallocBlock(Long64_t* offset, Int_t* length, Int_t nb)
{
   // Reallocate the block's buffer based on the length
   // of the elements it will contain.

   Int_t aux = 0;

   fNblock = nb;
   fPos = new Long64_t[nb];
   fLen = new Int_t[nb];

   for(Int_t i=0; i < nb; i++){

      fPos[i] = offset[i];
      fLen[i] = length[i];
      aux += fLen[i];
   }

   fBuffer = TStorage::ReAllocChar(fBuffer, aux, fFullSize);
   fFullSize = aux;
}


