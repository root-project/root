#ifndef ROOT_TFColumn
#define ROOT_TFColumn

#include <vector>
#include "TNamed.h"

template <class T> class TFColumn;

template <class T>
   class TFColumn : public TNamed
{
protected:
   std::vector <T>      fData;       // all Data of this column
   ClassDef(TFColumn, 1) // A column of TFTable
};

#endif
