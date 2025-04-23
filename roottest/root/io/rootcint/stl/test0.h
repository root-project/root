#ifndef ROOT_TFColumn
#define ROOT_TFColumn

#include "TNamed.h"

#include <set>

class TFBaseCol : public TNamed
{
protected:
   std::set    <ULong64_t> fNull;   // set of (row,bins) which are NULL values
   ClassDef(TFBaseCol,1) // Abstract base column of TFTable
};

#endif
