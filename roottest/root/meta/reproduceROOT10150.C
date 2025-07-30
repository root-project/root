// reproducer from https://its.cern.ch/jira/browse/ROOT-10150

#include <TInterpreter.h>
#include <functional>
#include <iostream>

namespace Foo {
struct Particle {
   float m_pt;
   float pt() const { return m_pt; }
};
} // namespace Foo

template <typename FType>
FType get_functor(std::string const &functor_string)
{
   auto intern = gInterpreter->MakeInterpreterValue();
   gInterpreter->Evaluate(functor_string.c_str(), *intern);
   return *static_cast<FType *>(intern->GetAsPointer());
}

void reproduceROOT10150()
{
   auto functor = get_functor<std::function<bool(Foo::Particle const &)>>(
      "std::function<bool(Foo::Particle const&)>( [](Foo::Particle const& particle){return particle.pt() > 15;} );");
   Foo::Particle low_pT_particle{1.f}, high_pT_particle{40.f};
   std::cout << functor(high_pT_particle) << " " << functor(low_pT_particle) << std::endl;
}
