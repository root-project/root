 

#include "RooStats/HistFactory/Data.h"


RooStats::HistFactory::Data::Data() : fName("") {
  ;
}

RooStats::HistFactory::Data::Data( std::string HistoName, std::string InputFile, 
				   std::string HistoPath ) :
  fInputFile( InputFile ), fHistoName( HistoName ), fHistoPath( HistoPath ) {;}

TH1* RooStats::HistFactory::Data::GetHisto() {
  return (TH1*) fhData.GetObject();
}


void RooStats::HistFactory::Data::Print( std::ostream& stream ) {


  stream << "\t \t InputFile: " << fInputFile
	 << "\t HistoName: " << fHistoName
	 << "\t HistoPath: " << fHistoPath
	 << std::endl;

}

void RooStats::HistFactory::Data::writeToFile( std::string OutputFileName, std::string DirName ) {

  TH1* histData = GetHisto();

  if( histData != NULL) {

    histData->Write();
  
    // Set the location of the data
    // in the output measurement
  
    fInputFile = OutputFileName;
    fHistoName = histData->GetName();
    fHistoPath = DirName;

  }

}
