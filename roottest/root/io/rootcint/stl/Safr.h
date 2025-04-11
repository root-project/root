#include <TObject.h>
#include <TMatrixF.h>

#include <list>

using namespace std;

typedef std::list<int**>                lppI_t;
typedef std::list<int**>::iterator      lppI_i;

class Safr : public TMatrixF {
 public:
  Safr() : TMatrixF(5,5) {}

  ClassDef(Safr, 1)
};
