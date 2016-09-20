#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_DataSetInfo
#include "TMVA/DataSetInfo.h"
#endif

class TTree;
class TFile;
class TDirectory;

namespace TMVA {

   class DataLoader;
   class MethodBase;
   class DataSetInfo;
   class Event;
   class DataSet;
   class MsgLogger;
   class DataInputHandler;
   class VarTransformHandler {
   public:

      VarTransformHandler(DataLoader*);
      ~VarTransformHandler();

      TMVA::DataLoader* VarianceThreshold(Double_t threshold);
      mutable MsgLogger* fLogger;             //! message logger
      MsgLogger& Log() const { return *fLogger; }

   private:

      DataSetInfo&                  fDataSetInfo;
      DataLoader*                   fDataLoader;
      const std::vector<Event*>&    fEvents;
      void                          UpdateNorm (Int_t ivar, Double_t x);
      void                          CalcNorm();
      void                          CopyDataLoader(TMVA::DataLoader* des, TMVA::DataLoader* src);
   };
}
