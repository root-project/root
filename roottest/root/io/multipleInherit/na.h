#ifndef __TMrbNamedArray_h__
#define __TMrbNamedArray_h__

#include "TArrayC.h"
#include "TArrayI.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TNamed.h"

//////////////////////////////////////////////////////////////////////////////
// Name:           TMrbNamedArrayC, I, F, D
// Description:    An array of char, int, float, double with name and title.
// Keywords:       
//////////////////////////////////////////////////////////////////////////////


class  TMrbNamedArrayI: public TArrayI, public TNamed {
   public:
      TMrbNamedArrayI(){};
      TMrbNamedArrayI(const char* name, const char* title, 
                      Int_t len = 0, const Int_t * val = 0);
      ~TMrbNamedArrayI(){};

ClassDef(TMrbNamedArrayI, 1)
};

#endif

