#include "TObject.h"


namespace ns {

template <class T>
class foo
  : public TObject
{
public:
  void a();
  ClassDefOverride (foo, 1);
};

}

