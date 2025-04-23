#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h
#include <vector>

namespace pat {
  class TauJetCorrFactors {
  public:
    TauJetCorrFactors():i(3){};
  private:
    int i;
  };

  class Tau {
    public:
      Tau(){};
      /// destructor
      ~Tau(){};
    private:
      TauJetCorrFactors tj;
  };
}

#endif
