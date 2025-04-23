#include "operators.h"

#ifdef __MAKECINT__
#pragma link C++ class myiterator;
#pragma link C++ operators myiterator;
#endif

#if 0
// This trick does not quite work in CINT because it does not
// properly lookup the arguments.
namespace enclosing {
   #include "operators.h"
}

#ifdef __MAKECINT__
#pragma link C++ class enclosing::myiterator;
#pragma link C++ operators enclosing::myiterator;
#endif
#endif

#include <vector>
#ifdef __MAKECINT__
#pragma link C++ class vector<myiterator>;
#pragma link C++ class vector<myiterator>::const_iterator;
#pragma link C++ class vector<myiterator>::iterator;
#pragma link C++ operators vector<myiterator>;
#pragma link C++ operators vector<myiterator>::const_iterator;
#pragma link C++ operators vector<myiterator>::iterator;
#endif
