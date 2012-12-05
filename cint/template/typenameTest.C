///// File A.h
template <class Config> class particle {
    public:
     typedef typename Config::PT PT;
     typedef typename Config::ParticleData ParticleData;
     typedef typename Config::DecayData DecayData;
    
     inline ParticleData getData();
     inline bool setData(ParticleData&);
   
     inline DecayData getDecay();
     inline bool setDecay(DecayData&);
   
   };

template <class Config> class DecayDataT {
    public:
     typedef typename Config::PT PT;
     typedef typename Config::ParticleData ParticleData;
   
     inline ParticleData getData();
     inline bool setData(ParticleData&);
   
   };

#ifdef __CINT__
// This was necessary to work around a bug in 
// the lookup code that prevented the class to use itself
// to instantiate templates.
#ifdef __CINT2__
class A;
class B;
class  DefaultConfig {
    public:
   
     typedef  int PT;
     typedef  A ParticleData;
     typedef  B DecayData;
   
   };
#endif
#endif

class  DefaultConfig {
    public:
     DefaultConfig () {};
   
     typedef  int PT;
     typedef  particle<DefaultConfig> ParticleData;
     typedef  DecayDataT<DefaultConfig>  DecayData;
   
   };

////// File alinkdef.h
#ifdef __CINT__

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;

#pragma link C++ class DefaultConfig;
#pragma link C++ class particle<DefaultConfig>;

#endif
