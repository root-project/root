


namespace Sash{
   class Telescope {
   public:
      int t;
   };
}

namespace Reco {
   class HillasParameters {
   public:
      int h;
   };
}

namespace Sash{
  template<class Env,class T> class EnvelopeEntry
  {
  public: 
      int e;
      ClassDef(EnvelopeEntry,1)
  };

  template<class Env,class T> class EnvelopeEntrySimple
  {
  public: 
      int e;
  };
}

