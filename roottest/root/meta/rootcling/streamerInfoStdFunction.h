#include <functional>
#include <vector>

struct Foo {};
struct egammaMVACalibTool
{
  std::vector<std::vector<std::function<float(int, const Foo*)> > > m_funcs;
};

