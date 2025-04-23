#include <vector>

namespace GaudiAlg{
   class ID{};
}

 namespace Histos
 {
   typedef GaudiAlg::ID HistoID;
 }
 // ============================================================================

 // ============================================================================
 namespace GaudiAlg
 {
   typedef Histos::HistoID HistoID ;
 }



class A{};

typedef A td0;
typedef td0 td1;
typedef td1 td2;

class IService;

std::vector<IService*> ciccio;

typedef std::vector<IService*>  IServicePtrs;

class B{};

template<class T>
class myTempl{};
myTempl<B***> myTemplOfB;

template<class T>
class myTempl2{};
myTempl2<myTempl<B>> myTempl2OfB;

