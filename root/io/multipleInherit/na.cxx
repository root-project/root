#include "na.h"


ClassImp(TMrbNamedArrayI)

//__________________________________________________________________________

TMrbNamedArrayI::TMrbNamedArrayI(const char* name, const char* title,
                                 Int_t len, const Int_t* val ) :
                TNamed(name, title) {if (len > 0 && val) Set(len, val);};
