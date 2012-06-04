
#ifndef HIST_FACTORY_MODEL_UTILS_H
#define HIST_FACTORY_MODEL_UTILS_H

#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooDataSet.h"
#include "RooStats/HistFactory/ParamHistFunc.h"

namespace RooStats {
namespace HistFactory {
  
  std::string channelNameFromPdf( RooAbsPdf* channelPdf );

//   void getChannelsFromModel( RooAbsPdf* model, RooArgSet* channels, 
// 			     RooArgSet* channelsWithConstraints );


  void FactorizeHistFactoryPdf(const RooArgSet&, RooAbsPdf&, RooArgList&, RooArgList&);
  bool getStatUncertaintyFromChannel( RooAbsPdf* channel, ParamHistFunc*& paramfunc, 
				      RooArgList* gammaList );

  RooAbsPdf* getSumPdfFromChannel( RooAbsPdf* channel );

  void getDataValuesForObservables( std::map< std::string, std::vector<double> >& ChannelBinDataMap, 
				    RooAbsData* data, RooAbsPdf* simPdf );


  int getStatUncertaintyConstraintTerm( RooArgList* constraints, RooRealVar* gamma_stat, 
					RooAbsReal*& pois_mean, RooRealVar*& tau );

}
}



#endif
