

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;


#pragma link C++ namespace RooStats;
#pragma link C++ namespace RooStats::HistFactory;

// for auto-loading namespaces
#ifdef USE_FOR_AUTLOADING
#pragma link C++ class RooStats::HistFactory;
#pragma link C++ class RooStats;
#endif            


#pragma link C++ class PiecewiseInterpolation- ;
#pragma link C++ class ParamHistFunc+ ;
#pragma link C++ class RooStats::HistFactory::LinInterpVar+ ;
#pragma link C++ class RooStats::HistFactory::FlexibleInterpVar+ ;
#pragma link C++ class RooStats::HistFactory::EstimateSummary+ ;
#pragma link C++ class RooStats::HistFactory::HistoToWorkspaceFactory+ ;
#pragma link C++ class RooStats::HistFactory::HistoToWorkspaceFactoryFast+ ;
#pragma link C++ class RooStats::HistFactory::RooBarlowBeestonLL+ ;  
#pragma link C++ class RooStats::HistFactory::HistFactorySimultaneous+ ;  
#pragma link C++ class RooStats::HistFactory::HistFactoryNavigation+ ;  

#pragma link C++ class RooStats::HistFactory::ConfigParser+ ;
#pragma link C++ class RooStats::HistFactory::Measurement+ ;
#pragma read sourceClass="RooStats::HistFactory::Measurement" targetClass="RooStats::HistFactory::Measurement" checksum="[973506941]" source="std::string fPOI" target="fPOI"  code="{ fPOI.push_back(onfile.fPOI) ; }" 

#pragma link C++ class RooStats::HistFactory::RooBSplineBases+ ;
#pragma link C++ class RooStats::HistFactory::RooBSpline+ ;

#pragma link C++ class RooStats::HistFactory::Channel+ ;
#pragma link C++ class RooStats::HistFactory::Sample+ ;
#pragma link C++ class RooStats::HistFactory::Data+ ;
#pragma link C++ class RooStats::HistFactory::Asimov+ ;

#pragma link C++ class RooStats::HistFactory::HistRef+ ;
#pragma link C++ class RooStats::HistFactory::StatError+ ;
#pragma link C++ class RooStats::HistFactory::StatErrorConfig+ ;
#pragma link C++ class RooStats::HistFactory::PreprocessFunction+ ;
#pragma link C++ class RooStats::HistFactory::HistogramUncertaintyBase+ ;
#pragma link C++ class RooStats::HistFactory::HistoSys+ ;
#pragma read sourceClass="RooStats::HistFactory::HistoSys" checksum="[0xa79a9653]" \
    source="RooStats::HistFactory::HistRef fhLow; RooStats::HistFactory::HistRef fhHigh" \
    targetClass="RooStats::HistFactory::HistoSys" target="" \
    code="{newObj->SetHistoLow ( onfile.fhLow.ReleaseObject() ); \
           newObj->SetHistoHigh( onfile.fhHigh.ReleaseObject() ); }"

#pragma link C++ class std::vector< RooStats::HistFactory::Channel >+ ;
#pragma link C++ class std::vector< RooStats::HistFactory::Sample >+ ;
#pragma link C++ class std::vector< RooStats::HistFactory::HistRef >+ ;

// make dictionary for all the C++ classes defined in these following files
#pragma link C++ defined_in "RooStats/HistFactory/MakeModelAndMeasurementsFast.h"; 
#pragma link C++ defined_in "RooStats/HistFactory/Systematics.h"; 
#pragma link C++ defined_in "RooStats/HistFactory/HistFactoryModelUtils.h"; 

#endif
