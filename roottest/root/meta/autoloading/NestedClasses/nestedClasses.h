class CastorElectronicsMap{
   public:
      class PrecisionItem{};
      typedef PrecisionItem TheTypedef;
};

class A{};
typedef A TheTypedef;

class B{};
namespace ns {
typedef B TheTypedef;
class C{};
}
