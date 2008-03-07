#ifndef ROOT_Math_BrentMethods
#define ROOT_Math_BrentMethods

#include <Math/IFunction.h>

namespace ROOT {
namespace Math {

double MinimStep(const IGenFunction* f, int type, double &xmin, double &xmax, double fy, int fNpx = 100);
double MinimBrent(const IGenFunction* f, int type, double &xmin, double &xmax, double xmiddle, double fy, bool &ok);
   
}
}

#endif
