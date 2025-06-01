#include <functional>
#include <vector>
#include <string>
#include <map>

struct Foo {};
struct Geant4Sensitive {};
struct Geant4HitCollection {};
struct egammaMVACalibTool {
   typedef Geant4HitCollection *(*create_t)(const std::string &, const std::string &, Geant4Sensitive *);
   typedef std::pair<std::string, std::pair<Geant4Sensitive *, create_t>> HitCollection;
   typedef std::vector<HitCollection> HitCollections;

   std::vector<std::vector<std::function<float(int, const Foo *)>>> m_funcs;
   HitCollections m_collections;
};