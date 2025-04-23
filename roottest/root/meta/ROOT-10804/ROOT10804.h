#include <list>
namespace Outer {
  inline namespace Inline {
    class Class {
      int MEMBER;
    };
    template <class T>
    class Template {
      T t;
    };
  }
  template <typename T> class Container {};
}

namespace Instantiations {
  // Simulating what the Gaudi experiments do due to legacy gccxml
  // "I only see what you instantiate" (not needed anymore with cling).
  std::list<Outer::Class> inst1;
  Outer::Container<Outer::Class> inst2;
  Outer::Template<Outer::Class> inst3;
  Outer::Container<Outer::Template<Outer::Class> > inst4;
  Outer::Template<Float16_t> inst5;
  Outer::Container<Outer::Template<Double32_t> > inst6;
}
