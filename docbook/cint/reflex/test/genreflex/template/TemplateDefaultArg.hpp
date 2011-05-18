// Check that template default arguments are handled correctly.
// Name caching can creak it, see https://savannah.cern.ch/bugs/?43356

#include "Reflex/Builder/DictSelection.h"

struct NoBase {};


namespace Reflex { namespace Selection {
template <class T, class BASE>
struct DataVector
{
  typedef DataVector<T, BASE> self;

  Reflex::Selection::TEMPLATE_DEFAULTS<
   Reflex::Selection::NODEFAULT, NoBase> dum1;
  Reflex::Selection::NO_SELF_AUTOSELECT dum2;
};
}}


template <class T, class BASE = NoBase>
class DataVector
{
public:
  typedef typename
    Reflex::Selection::DataVector<T, NoBase>::self self;
};


struct dictdummy {
  DataVector<int> m_vdummy;
};

