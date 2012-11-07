#include "RooStats/HistFactory/Channel.h"
#include <stdlib.h>

#include "TFile.h"
#include "TTimeStamp.h"

#include "RooStats/HistFactory/HistFactoryException.h"

using namespace std;

RooStats::HistFactory::Channel::Channel() :
  fName( "" ) { ; }

RooStats::HistFactory::Channel::Channel(std::string ChanName, std::string ChanInputFile) :
  fName( ChanName ), fInputFile( ChanInputFile ) { ; }

namespace RooStats{
  namespace HistFactory{
    //BadChannel = Channel();
    Channel BadChannel;
    //    BadChannel.Name = "BadChannel"; // = Channel(); //.Name = "BadChannel";
  }
}


void RooStats::HistFactory::Channel::AddSample( RooStats::HistFactory::Sample sample ) { 
  sample.SetChannelName( GetName() );
  fSamples.push_back( sample ); 
}

void RooStats::HistFactory::Channel::Print( std::ostream& stream ) {


  stream << "\t Channel Name: " << fName
	 << "\t InputFile: " << fInputFile
	 << std::endl;

  stream << "\t Data:" << std::endl;
  fData.Print( stream );


  stream << "\t statErrorConfig:" << std::endl;
  fStatErrorConfig.Print( stream );


  if( fSamples.size() != 0 ) {

    stream << "\t Samples: " << std::endl;
    for( unsigned int i = 0; i < fSamples.size(); ++i ) {
      fSamples.at(i).Print( stream );
    }
  }

  
  stream << "\t End of Channel " << fName <<  std::endl;


}  


void RooStats::HistFactory::Channel::PrintXML( std::string Directory, std::string Prefix ) {

  // Create an XML file for this channel

  std::cout << "Printing XML Files for channel: " << GetName() << std::endl;
  
  std::string XMLName = Prefix + fName + ".xml";
  if( Directory != "" ) XMLName = Directory + "/" + XMLName;
  
  ofstream xml( XMLName.c_str() );


  // Add the time
  xml << "<!--" << std::endl;
  xml << "This xml file created automatically on: " << std::endl;
/*
  time_t t = time(0);   // get time now
  struct tm * now = localtime( &t );
  xml << (now->tm_year + 1900) << '-'
      << (now->tm_mon + 1) << '-'
      << now->tm_mday
      << std::endl;
*/
// LM: use TTimeStamp since time_t does not work on Windows
  TTimeStamp t; 
  UInt_t year = 0; 
  UInt_t month = 0; 
  UInt_t day = 0; 
  t.GetDate(true, 0, &year, &month, &day);
  xml << year << '-'
      << month << '-'
      << day
      << std::endl;

  xml << "-->" << std::endl;
  

  // Add the DOCTYPE
  xml << "<!DOCTYPE Channel  SYSTEM 'HistFactorySchema.dtd'>  " << std::endl << std::endl;

  // Add the Channel
  xml << "  <Channel Name=\"" << fName << "\" InputFile=\"" << fInputFile << "\" >" << std::endl << std::endl;

  xml << "    <Data HistoName=\"" << fData.GetHistoName() << "\" "
      << "InputFile=\"" << fData.GetInputFile() << "\" "
      << "HistoPath=\"" << fData.GetHistoPath() << "\" "
      << " /> " << std::endl << std::endl;  


  xml << "    <StatErrorConfig RelErrorThreshold=\"" << fStatErrorConfig.GetRelErrorThreshold() << "\" "
      << "ConstraintType=\"" << Constraint::Name( fStatErrorConfig.GetConstraintType() ) << "\" "
      << "/> " << std::endl << std::endl;            


  for( unsigned int i = 0; i < fSamples.size(); ++i ) {
    fSamples.at(i).PrintXML( xml );
    xml << std::endl << std::endl;
  }

  xml << std::endl;

  xml << "  </Channel>  " << std::endl;

  xml.close();

  std::cout << "Finished printing XML files" << std::endl;



}



void RooStats::HistFactory::Channel::SetData( std::string DataHistoName, std::string DataInputFile, std::string DataHistoPath ) {

  fData.SetHistoName( DataHistoName );
  fData.SetInputFile( DataInputFile );
  fData.SetHistoPath( DataHistoPath );

}



void RooStats::HistFactory::Channel::SetData( TH1* hData ) { 
  fData.SetHisto( hData ); 
}

void RooStats::HistFactory::Channel::SetData( double val ) {

  // For a NumberCounting measurement only
  // Set the value of data in a particular channel
  // 
  // Internally, this simply creates a 1-bin TH1F for you

  std::string DataHistName = fName + "_data";
  
  // Histogram has 1-bin (hard-coded)
  TH1F* hData = new TH1F( DataHistName.c_str(), DataHistName.c_str(), 1, 0, 1 );
  hData->SetBinContent( 1, val );

  // Set the histogram of the internally held data
  // node of this channel to this newly created histogram
  SetData( hData );

}


void RooStats::HistFactory::Channel::SetStatErrorConfig( double StatRelErrorThreshold, Constraint::Type StatConstraintType ) {

  fStatErrorConfig.SetRelErrorThreshold( StatRelErrorThreshold );
  fStatErrorConfig.SetConstraintType( StatConstraintType );

}

void RooStats::HistFactory::Channel::SetStatErrorConfig( double StatRelErrorThreshold, std::string StatConstraintType ) {

  fStatErrorConfig.SetRelErrorThreshold( StatRelErrorThreshold );
  fStatErrorConfig.SetConstraintType( Constraint::GetType(StatConstraintType) );

}



void RooStats::HistFactory::Channel::CollectHistograms() {

  // Loop through all Samples and Systematics
  // and collect all necessary histograms

  // Get the Data Histogram:

  if( fData.GetInputFile() != "" ) {
    fData.SetHisto( GetHistogram(fData.GetInputFile(), 
				 fData.GetHistoPath(),
				 fData.GetHistoName()) );
  }

  // Collect any histograms for additional Datasets
  for( unsigned int i=0; i < fAdditionalData.size(); ++i) {
    RooStats::HistFactory::Data& data = fAdditionalData.at(i);
    if( data.GetInputFile() != "" ) {
      data.SetHisto( GetHistogram(data.GetInputFile(), data.GetHistoPath(),data.GetHistoName()) );
    }
  }

  // Get the histograms for the samples:
  for( unsigned int sampItr = 0; sampItr < fSamples.size(); ++sampItr ) {

    RooStats::HistFactory::Sample& sample = fSamples.at( sampItr );


    // Get the nominal histogram:
    std::cout << "Collecting Nominal Histogram" << std::endl;
    TH1* Nominal =  GetHistogram(sample.GetInputFile(),
				 sample.GetHistoPath(),
				 sample.GetHistoName());

    sample.SetHisto( Nominal );


    // Get the StatError Histogram (if necessary)

    if( sample.GetStatError().GetUseHisto() ) {

      sample.GetStatError().SetErrorHist( GetHistogram(sample.GetStatError().GetInputFile(),
						       sample.GetStatError().GetHistoPath(),
						       sample.GetStatError().GetHistoName()) );
    }

      
    // Get the HistoSys Variations:
    for( unsigned int histoSysItr = 0; histoSysItr < sample.GetHistoSysList().size(); ++histoSysItr ) {

      RooStats::HistFactory::HistoSys& histoSys = sample.GetHistoSysList().at( histoSysItr );
	
      histoSys.SetHistoLow( GetHistogram(histoSys.GetInputFileLow(), 
					 histoSys.GetHistoPathLow(),
					 histoSys.GetHistoNameLow()) );
	
      histoSys.SetHistoHigh( GetHistogram(histoSys.GetInputFileHigh(),
					  histoSys.GetHistoPathHigh(),
					  histoSys.GetHistoNameHigh()) );
    } // End Loop over HistoSys


      // Get the HistoFactor Variations:
    for( unsigned int histoFactorItr = 0; histoFactorItr < sample.GetHistoFactorList().size(); ++histoFactorItr ) {

      RooStats::HistFactory::HistoFactor& histoFactor = sample.GetHistoFactorList().at( histoFactorItr );

      histoFactor.SetHistoLow( GetHistogram(histoFactor.GetInputFileLow(), 
					    histoFactor.GetHistoPathLow(),
					    histoFactor.GetHistoNameLow()) );
	
      histoFactor.SetHistoHigh( GetHistogram(histoFactor.GetInputFileHigh(),
					     histoFactor.GetHistoPathHigh(),
					     histoFactor.GetHistoNameHigh()) );
    } // End Loop over HistoFactor


      // Get the ShapeSys Variations:
    for( unsigned int shapeSysItr = 0; shapeSysItr < sample.GetShapeSysList().size(); ++shapeSysItr ) {
	
      RooStats::HistFactory::ShapeSys& shapeSys = sample.GetShapeSysList().at( shapeSysItr );

      shapeSys.SetErrorHist( GetHistogram(shapeSys.GetInputFile(), 
					  shapeSys.GetHistoPath(),
					  shapeSys.GetHistoName()) );
    } // End Loop over ShapeSys


  } // End Loop over Samples

  return;
  
}


bool RooStats::HistFactory::Channel::CheckHistograms() { 

  // Check that all internal histogram pointers
  // are properly configured (ie that they're not NULL)

  try {
  
    if( fData.GetHisto() == NULL && fData.GetInputFile() != "" ) {
      std::cout << "Error: Data Histogram for channel " << GetName() << " is NULL." << std::endl;
      throw hf_exc();
    }

    // Get the histograms for the samples:
    for( unsigned int sampItr = 0; sampItr < fSamples.size(); ++sampItr ) {

      RooStats::HistFactory::Sample& sample = fSamples.at( sampItr );

      // Get the nominal histogram:
      if( sample.GetHisto() == NULL ) {
	std::cout << "Error: Nominal Histogram for sample " << sample.GetName() << " is NULL." << std::endl;
	throw hf_exc();
      } 
      else {

	// Check if any bins are negative
	std::vector<int> NegativeBinNumber;
	std::vector<double> NegativeBinContent;
	TH1* histNominal = sample.GetHisto();
	for(int ibin=1; ibin<=histNominal->GetNbinsX(); ++ibin) {
	  if(histNominal->GetBinContent(ibin) < 0) {
	    NegativeBinNumber.push_back(ibin);
	    NegativeBinContent.push_back(histNominal->GetBinContent(ibin));
	  }
	}
	if(NegativeBinNumber.size()>0) {
	  std::cout << "WARNING: Nominal Histogram " << histNominal->GetName() << " for Sample = " << sample.GetName()
		    << " in Channel = " << GetName() << " has negative entries in bin numbers = ";

	  for(unsigned int ibin=0; ibin<NegativeBinNumber.size(); ++ibin) {
	    if(ibin>0) std::cout << " , " ;
	    std::cout << NegativeBinNumber[ibin] << " : " << NegativeBinContent[ibin] ;
	  }
	  std::cout << std::endl;
	}
	
      }

      // Get the StatError Histogram (if necessary)
      if( sample.GetStatError().GetUseHisto() ) {
	if( sample.GetStatError().GetErrorHist() == NULL ) {
	  std::cout << "Error: Statistical Error Histogram for sample " << sample.GetName() << " is NULL." << std::endl;
	  throw hf_exc();
	}
      }

      
      // Get the HistoSys Variations:
      for( unsigned int histoSysItr = 0; histoSysItr < sample.GetHistoSysList().size(); ++histoSysItr ) {

	RooStats::HistFactory::HistoSys& histoSys = sample.GetHistoSysList().at( histoSysItr );

	if( histoSys.GetHistoLow() == NULL ) {
	  std::cout << "Error: HistoSyst Low for Systematic " << histoSys.GetName() 
		    << " in sample " << sample.GetName() << " is NULL." << std::endl;
	  throw hf_exc();
	}
	if( histoSys.GetHistoHigh() == NULL ) {
	  std::cout << "Error: HistoSyst High for Systematic " << histoSys.GetName() 
		    << " in sample " << sample.GetName() << " is NULL." << std::endl;
	  throw hf_exc();
	}
	
      } // End Loop over HistoSys


      // Get the HistoFactor Variations:
      for( unsigned int histoFactorItr = 0; histoFactorItr < sample.GetHistoFactorList().size(); ++histoFactorItr ) {

	RooStats::HistFactory::HistoFactor& histoFactor = sample.GetHistoFactorList().at( histoFactorItr );

	if( histoFactor.GetHistoLow() == NULL ) {
	  std::cout << "Error: HistoSyst Low for Systematic " << histoFactor.GetName() 
		    << " in sample " << sample.GetName() << " is NULL." << std::endl;
	  throw hf_exc();
	}
	if( histoFactor.GetHistoHigh() == NULL ) {
	  std::cout << "Error: HistoSyst High for Systematic " << histoFactor.GetName() 
		    << " in sample " << sample.GetName() << " is NULL." << std::endl;
	  throw hf_exc();
	}

      } // End Loop over HistoFactor


      // Get the ShapeSys Variations:
      for( unsigned int shapeSysItr = 0; shapeSysItr < sample.GetShapeSysList().size(); ++shapeSysItr ) {
	
	RooStats::HistFactory::ShapeSys& shapeSys = sample.GetShapeSysList().at( shapeSysItr );

	if( shapeSys.GetErrorHist() == NULL ) {
	  std::cout << "Error: HistoSyst High for Systematic " << shapeSys.GetName() 
		    << " in sample " << sample.GetName() << " is NULL." << std::endl;
	  throw hf_exc();
	}

      } // End Loop over ShapeSys

    } // End Loop over Samples

  }
  catch(std::exception& e)
    {
      std::cout << e.what() << std::endl;
      return false;
    }

  return true;




}




TH1* RooStats::HistFactory::Channel::GetHistogram(std::string InputFile, std::string HistoPath, std::string HistoName) {

  std::cout << "Getting histogram. "  
	    << " InputFile " << InputFile
	    << " HistoPath " << HistoPath
	    << " HistoName " << HistoName
	    << std::endl;

  //  TFile* file = TFile::Open( InputFile.c_str() );

  TFile* inFile = TFile::Open( InputFile.c_str() );
  if( !inFile ) {
    std::cout << "Error: Unable to open input file: " << InputFile << std::endl;
    throw hf_exc();
  }

  std::cout << "Opened input file: " << InputFile << ": " << inFile << std::endl;

  std::string HistNameFull = HistoPath + HistoName;

  if( HistoPath != std::string("") ) {
    if( HistoPath[ HistoPath.length()-1 ] != std::string("/") ) {
      std::cout << "WARNING: Histogram path is set to: " << HistoPath
		<< " but it should end with a '/' " << std::endl;
      std::cout << "Total histogram path is now: " << HistNameFull << std::endl;
    }
  }

  TH1* hist = NULL;
  try{
    hist = dynamic_cast<TH1*>( inFile->Get( HistNameFull.c_str() ) );
  }
  catch(std::exception& e)
    {
      std::cout << "Failed to cast object to TH1*" << std::endl;
      std::cout << e.what() << std::endl;
      throw hf_exc();
    }
  if( !hist ) {
    std::cout << "Failed to get histogram: " << HistNameFull
	      << " in file: " << InputFile << std::endl;
    throw hf_exc();
  }


  TH1 * ptr = (TH1 *) hist->Clone();

  if(!ptr){
    std::cerr << "Not all necessary info are set to access the input file. Check your config" << std::endl;
    std::cerr << "filename: " << InputFile
	      << "path: " << HistoPath
	      << "obj: " << HistoName << std::endl;
    throw hf_exc();
  }
  else {
    ptr->SetDirectory(0); //         for the current histogram h
  }

  
#ifdef DEBUG
  std::cout << "Found Histogram: " << HistoName " at address: " << ptr 
	    << " with integral "   << ptr->Integral() << " and mean " << ptr->GetMean() 
	    << std::endl;
#endif


  inFile->Close();

  // Done
  return ptr;

}
