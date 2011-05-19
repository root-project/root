#ifndef ROOT_TFPBlock
#define ROOT_TFPBlock

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TFPBlock : public TObject{

public:

   TFPBlock(Long64_t*, Int_t*, Int_t);                 //! constructor
   ~TFPBlock();                                        //! destructor

   Long64_t GetPos(Int_t);                             //! the position of the segment at a certain index
   Int_t GetLen(Int_t);                                //! the length of a segment at a certain index   

   Long64_t* GetPos();                                 //! a pointer to the array of positions
   Int_t* GetLen();                                    //! a pointer to the array of lengths
   Int_t GetFullSize();                                //! the full size of the buffer
   Int_t GetNoElem();                                  //! number of segments in the block
   char* GetBuffer();                                  //! pointer to the actual buffer

   void SetBuffer(char*);                              //! set the value of the buffer
   void ReallocBlock(Long64_t*, Int_t*, Int_t);        //! function used to reallocate the elemnts of the block
                                                       //  given the size and the two arrays of postions and lengths

private:

   char     *fBuffer;                        //! content of the block
   Int_t     fNblock;                        //! number of segment in the block
   Int_t     fFullSize;                      //! total size of segments that make up the block
   Int_t    *fLen;                           //! array of lengths of each segment
   Long64_t *fPos;                           //! array of positions of each segment
  
   ClassDef(TFPBlock, 0);
};
#endif

