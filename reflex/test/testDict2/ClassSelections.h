#include "Reflex/Builder/DictSelection.h"


namespace ROOT {
  namespace Reflex {
    namespace Selection {
      class ClassA {
        TRANSIENT fA;
      };
    }
  }
}

#include <vector>
#include <utility>

namespace ns {

  class TestSelectionClass {
  private:
    int fI;
    float fF;
    int foo(bool b, char c) { if (b) return c; }
    std::vector<std::pair<int,float> > fV;
  };


  template < typename T0, typename T1, typename T2 = float >
    class TestTemplatedSelectionClass {
    private:
    int fI;
  };
}

namespace ROOT {
  namespace Reflex {
    namespace Selection {

      namespace ns {

        class TestSelectionClass {
          TRANSIENT  fI;
          AUTOSELECT fV;
        };

        template < typename T0, typename T1, typename T2 = float > class TestTemplatedSelectionClass {
          ::ns::TestTemplatedSelectionClass<T0,T1,T2>      fInstance;
          TEMPLATE_DEFAULTS<NODEFAULT,NODEFAULT,float> fDefaults;
        };
      }

    }
  }
}


namespace {

  struct _Instantiations {

    ROOT::Reflex::Selection::ns::TestTemplatedSelectionClass<int,int>      fI4;
    ROOT::Reflex::Selection::ns::TestTemplatedSelectionClass<float,float>  fI5;
    ROOT::Reflex::Selection::ns::TestTemplatedSelectionClass<int,int,bool> fI6;
    ROOT::Reflex::Selection::ns::TestTemplatedSelectionClass<int,int,char> fI7;

  };

}
