// This is ROOT-8980:
// Cannot forward declare param packs to rootmap files.

#include <string>
namespace edm {
  template <int I, typename... T>
  class Table {};
}
