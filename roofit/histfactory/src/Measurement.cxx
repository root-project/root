
#include <ctime>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include "TSystem.h"
#include "TTimeStamp.h"

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistFactoryException.h"

using namespace std;

ClassImp(RooStats::HistFactory::Measurement) ;


RooStats::HistFactory::Measurement::Measurement() :
  fPOI(), fLumi( 1.0 ), fLumiRelErr( .10 ), 
  fBinLow( 0 ), fBinHigh( 1 ), fExportOnly( false )  { ; }

/*
RooStats::HistFactory::Measurement::Measurement(const Measurement& other) :
  POI( other.POI ), Lumi( other.Lumi ), LumiRelErr( other.LumiRelErr ), 
  BinLow( other.BinLow ), BinHigh( other.BinHigh ), ExportOnly( other.ExportOnly ),
  channels( other.channels ), OutputFilePrefix( other.outputFilePrefix ),
  constantParams( other.constantParams ), { ; }
*/

RooStats::HistFactory::Measurement::Measurement(const char* Name, const char* Title) :
  TNamed( Name, Title ),
  fPOI(), fLumi( 1.0 ), fLumiRelErr( .10 ), 
  fBinLow( 0 ), fBinHigh( 1 ), fExportOnly( false )  { ; }


void RooStats::HistFactory::Measurement::AddConstantParam( const std::string& param ) { 
  
  // Check if the parameter is already set constant
  // We don't need to set it constant twice,
  // and we issue a warning in case this is a hint
  // of a possible bug

  if( std::find(fConstantParams.begin(), fConstantParams.end(), param) != fConstantParams.end() ) {
    std::cout << "Warning: Setting parameter: " << param 
	      << " to constant, but it is already listed as constant.  "
	      << "You may ignore this warning."
	      << std::endl;
    return;
  }

  fConstantParams.push_back( param ); 

}

void RooStats::HistFactory::Measurement::SetParamValue( const std::string& param, double value ) {

  // Check if this parameter is already set to a value
  // If so, issue a warning
  // (Not sure if we want to throw an exception here, or
  // issue a warning and move along.  Thoughts?)
  if( fParamValues.find(param) != fParamValues.end() ) {
    std::cout << "Warning: Chainging parameter: " << param
	      << " value from: " << fParamValues[param]
	      << " to: " << value 
	      << std::endl;
  }

  // Store the parameter and its value
  std::cout << "Setting parameter: " << param
	    << " value to " << value
	    << std::endl;

  fParamValues[param] = value;

}

void RooStats::HistFactory::Measurement::AddPreprocessFunction( std::string name, std::string expression, std::string dependencies ) {

  PreprocessFunction func(name, expression, dependencies);
  AddFunctionObject(func);

}


std::vector<std::string> RooStats::HistFactory::Measurement::GetPreprocessFunctions() {
  std::vector<std::string> PreprocessFunctionExpressions;
  for( unsigned int i = 0; i < fFunctionObjects.size(); ++i ) {
    std::string expression = fFunctionObjects.at(i).GetCommand();
    PreprocessFunctionExpressions.push_back( expression );
  }
  return PreprocessFunctionExpressions;
}

void RooStats::HistFactory::Measurement::AddGammaSyst(std::string syst, double uncert) {
  fGammaSyst[syst] = uncert;
}

void RooStats::HistFactory::Measurement::AddLogNormSyst(std::string syst, double uncert) {
  fLogNormSyst[syst] = uncert;
}

void RooStats::HistFactory::Measurement::AddUniformSyst(std::string syst) {
  fUniformSyst[syst] = 1.0; // Is this parameter simply a dummy?
}

void RooStats::HistFactory::Measurement::AddNoSyst(std::string syst) {
  fNoSyst[syst] = 1.0; // dummy value
}


bool RooStats::HistFactory::Measurement::HasChannel( std::string ChanName ) {

  for( unsigned int i = 0; i < fChannels.size(); ++i ) {

    Channel& chan = fChannels.at(i);
    if( chan.GetName() == ChanName ) {
      return true;
    }

  }

  return false;

}

RooStats::HistFactory::Channel& RooStats::HistFactory::Measurement::GetChannel( std::string ChanName ) {

  for( unsigned int i = 0; i < fChannels.size(); ++i ) {

    Channel& chan = fChannels.at(i);
    if( chan.GetName() == ChanName ) {
      return chan;
    }

  }
  
  // If we get here, we didn't find the channel

  std::cout << "Error: Did not find channel: " << ChanName
	    << " in measurement: " << GetName() << std::endl;
  throw hf_exc();

  // No Need to return after throwing exception
  // return RooStats::HistFactory::BadChannel;


}

/*
  void RooStats::HistFactory::Measurement::Print( Option_t* option ) const {
  RooStats::HistFactory::Measurement::Print( std::cout );
  return;
  }
*/

void RooStats::HistFactory::Measurement::PrintTree( std::ostream& stream ) {
  
  stream << "Measurement Name: " << GetName()
	 << "\t OutputFilePrefix: " << fOutputFilePrefix
	 << "\t POI: ";
  for(unsigned int i = 0; i < fPOI.size(); ++i) {
    stream << fPOI.at(i);
  }
  stream << "\t Lumi: " << fLumi
	 << "\t LumiRelErr: " << fLumiRelErr
	 << "\t BinLow: " << fBinLow
	 << "\t BinHigh: " << fBinHigh
	 << "\t ExportOnly: " << fExportOnly
	 << std::endl;


  if( fConstantParams.size() != 0 ) {
    stream << "Constant Params: ";
    for( unsigned int i = 0; i < fConstantParams.size(); ++i ) {
      stream << " " << fConstantParams.at(i);
    }
    stream << std::endl;
  }

  if( fFunctionObjects.size() != 0 ) {
    stream << "Preprocess Functions: ";
    for( unsigned int i = 0; i < fFunctionObjects.size(); ++i ) {
      stream << " " << fFunctionObjects.at(i).GetCommand();
    }
    stream << std::endl;
  }
  
  if( fChannels.size() != 0 ) {
    stream << "Channels:" << std::endl;
    for( unsigned int i = 0; i < fChannels.size(); ++i ) {
      fChannels.at(i).Print( stream );
    }
  }

  std::cout << "End Measurement: " << GetName() << std::endl;

}



void RooStats::HistFactory::Measurement::PrintXML( std::string Directory, std::string NewOutputPrefix ) {

  // Create an XML file for this measurement
  // First, create the XML driver
  // Then, create xml files for each channel

  // First, check that the directory exists:


  // LM : fixes for Windows 
  if( gSystem->OpenDirectory( Directory.c_str() ) == 0 ) {
    int success = gSystem->MakeDirectory(Directory.c_str() );    
    if( success != 0 ) {
      std::cout << "Error: Failed to make directory: " << Directory << std::endl;
      throw hf_exc();
    }
  }

  // If supplied new Prefix, use that one:

  std::cout << "Printing XML Files for measurement: " << GetName() << std::endl;

  std::string XMLName = std::string(GetName()) + ".xml";
  if( Directory != "" ) XMLName = Directory + "/" + XMLName;

  ofstream xml( XMLName.c_str() );

  if( ! xml.is_open() ) {
    std::cout << "Error opening xml file: " << XMLName << std::endl;
    throw hf_exc();
  }


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
  // LM: use TTimeStamp 
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

  // Add the doctype
  xml << "<!DOCTYPE Combination  SYSTEM 'HistFactorySchema.dtd'>" << std::endl << std::endl;

  // Add the combination name
  xml << "<Combination OutputFilePrefix=\"" << NewOutputPrefix /*OutputFilePrefix*/ << "\" >" << std::endl << std::endl;

  // Add the Preprocessed Functions
  for( unsigned int i = 0; i < fFunctionObjects.size(); ++i ) {
    RooStats::HistFactory::PreprocessFunction func = fFunctionObjects.at(i);
    xml << "<Function Name=\"" << func.GetName() << "\" "
	<< "Expression=\""     << func.GetExpression() << "\" "
	<< "Dependents=\""     << func.GetDependents() << "\" "
	<< "/>" << std::endl;
  }
  
  xml << std::endl;

  // Add the list of channels
  for( unsigned int i = 0; i < fChannels.size(); ++i ) {
    xml << "  <Input>" << "./" << Directory << "/" << GetName() << "_" << fChannels.at(i).GetName() << ".xml" << "</Input>" << std::endl;
  }

  xml << std::endl;

  // Open the Measurement, Set Lumi
  xml << "  <Measurement Name=\"" << GetName() << "\" "
      << "Lumi=\""        << fLumi       << "\" " 
      << "LumiRelErr=\""  << fLumiRelErr << "\" "
      << "BinLow=\""      << fBinLow     << "\" "
      << "BinHigh=\""     << fBinHigh    << "\" "
      << "ExportOnly=\""  << (fExportOnly ? std::string("True") : std::string("False")) << "\" "
      << " >" <<  std::endl;


  // Set the POI
  xml << "    <POI>" ;
  for(unsigned int i = 0; i < fPOI.size(); ++i) {
    xml << fPOI.at(i) << " ";
  } 
  xml << "</POI>  " << std::endl;
  
  // Set the Constant Parameters
  xml << "    <ParamSetting Const=\"True\">";
  for( unsigned int i = 0; i < fConstantParams.size(); ++i ) {
    xml << fConstantParams.at(i) << " ";
  }
  xml << "</ParamSetting>" << std::endl;

  // Close the Measurement
  xml << "  </Measurement> " << std::endl << std::endl;

  // Close the combination
  xml << "</Combination>" << std::endl;

  xml.close();

  // Now, make the xml files 
  // for the individual channels:

  std::string Prefix = std::string(GetName()) + "_";

  for( unsigned int i = 0; i < fChannels.size(); ++i ) {
    fChannels.at(i).PrintXML( Directory, Prefix );
  }


  std::cout << "Finished printing XML files" << std::endl;

}



void RooStats::HistFactory::Measurement::writeToFile( TFile* file ) {

  // Write every histogram to the file.
  // Edit the measurement to point to this file
  // and to point to each histogram in this file

  // Then write the measurement itself.

  // Create a tempory measurement
  // (This is the one that is actually written)
  RooStats::HistFactory::Measurement outMeas( *this );

  std::string OutputFileName = file->GetName();

  // Collect all histograms from file:
  // HistCollector collector;


  for( unsigned int chanItr = 0; chanItr < outMeas.fChannels.size(); ++chanItr ) {
    
    // Go to the main directory 
    // in the file
    file->cd();
    file->Flush();

    // Get the name of the channel:
    RooStats::HistFactory::Channel& channel = outMeas.fChannels.at( chanItr );
    std::string chanName = channel.GetName();

    
    if( ! channel.CheckHistograms() ) {
      std::cout << "Measurement.writeToFile(): Channel: " << chanName
		<< " has uninitialized histogram pointers" << std::endl;
      throw hf_exc();
      return;
    }
    
    // Get and cache the histograms for this channel:
    //collector.CollectHistograms( channel );
    // Do I need this...?
    // channel.CollectHistograms();

    // Make a directory to store the histograms
    // for this channel

    TDirectory* chanDir = file->mkdir( (chanName + "_hists").c_str() );
    if( chanDir == NULL ) {
      std::cout << "Error: Cannot create channel " << (chanName + "_hists")
		<< std::endl;
      throw hf_exc();
    }
    chanDir->cd();

    // Save the data:
    
    TDirectory* dataDir = chanDir->mkdir( "data" );
    if( dataDir == NULL ) {
      std::cout << "Error: Cannot make directory " << chanDir << std::endl;
      throw hf_exc();
    }
    dataDir->cd();

    channel.fData.writeToFile( OutputFileName, GetDirPath(dataDir) );

    /*
    // Write the data file to this directory
    TH1* hData = channel.data.GetHisto();
    hData->Write();

    // Set the location of the data
    // in the output measurement

    channel.data.InputFile = OutputFileName;
    channel.data.HistoName = hData->GetName();
    channel.data.HistoPath = GetDirPath( dataDir );
    */

    // Loop over samples:

    for( unsigned int sampItr = 0; sampItr < channel.GetSamples().size(); ++sampItr ) {

      RooStats::HistFactory::Sample& sample = channel.GetSamples().at( sampItr );
      std::string sampName = sample.GetName();
      
      std::cout << "Writing sample: " << sampName << std::endl;

      file->cd();
      chanDir->cd();
      TDirectory* sampleDir = chanDir->mkdir( sampName.c_str() );
      if( sampleDir == NULL ) {
	std::cout << "Error: Directory " << sampName << " not created properly" << std::endl;
	throw hf_exc();
      }
      std::string sampleDirPath = GetDirPath( sampleDir );

      if( ! sampleDir ) {
	std::cout << "Error making directory: " << sampName 
		  << " in directory: " << chanName
		  << std::endl;
	throw hf_exc();
      }

      // Write the data file to this directory
      sampleDir->cd();      
      
      sample.writeToFile( OutputFileName, sampleDirPath );
      /*
      TH1* hSample = sample.GetHisto();
      if( ! hSample ) {
	std::cout << "Error getting histogram for sample: " 
		  << sampName << std::endl;
	throw -1;
      }
      sampleDir->cd();    
      hSample->Write();

      sample.InputFile = OutputFileName;
      sample.HistoName = hSample->GetName();
      sample.HistoPath = sampleDirPath;
      */

      // Write the histograms associated with
      // systematics
      sample.GetStatError().writeToFile( OutputFileName, sampleDirPath );

      // Must write all systematics that contain internal histograms
      // (This is not all systematics)

      for( unsigned int i = 0; i < sample.GetHistoSysList().size(); ++i ) {
	sample.GetHistoSysList().at(i).writeToFile( OutputFileName, sampleDirPath );
      }
      for( unsigned int i = 0; i < sample.GetHistoFactorList().size(); ++i ) {
	sample.GetHistoFactorList().at(i).writeToFile( OutputFileName, sampleDirPath );
      }
      for( unsigned int i = 0; i < sample.GetShapeSysList().size(); ++i ) {
	sample.GetShapeSysList().at(i).writeToFile( OutputFileName, sampleDirPath );
      }

      /*
      sample.statError.writeToFile( OutputFileName, sampleDirPath );

      // Now, get the Stat config histograms
      if( sample.statError.HistoName != "" ) {
	TH1* hStatError = sample.statError.GetErrorHist();
	if( ! hStatError ) {
	  std::cout << "Error getting stat error histogram for sample: " 
		    << sampName << std::endl;
	  throw -1;
	}
	hStatError->Write();
      
	sample.statError.InputFile = OutputFileName;
	sample.statError.HistoName = hStatError->GetName();
	sample.statError.HistoPath = sampleDirPath;

      }
      */

    }

  }
  
  
  // Finally, write the measurement itself:

  std::cout << "Saved all histograms" << std::endl;
  
  file->cd();
  outMeas.Write();

  std::cout << "Saved Measurement" << std::endl;

}


std::string RooStats::HistFactory::Measurement::GetDirPath( TDirectory* dir ) {

  // Return the directory's path,
  // stripped of unnecessary prefixes

  std::string path = dir->GetPath();

  if( path.find(":") != std::string::npos ) {
    size_t index = path.find(":");
    path.replace( 0, index+1, "" );
  }                   

  path = path + "/";

  return path;

  /*
      if( path.find(":") != std::string::npos ) {
	size_t index = path.find(":");
	SampleName.replace( 0, index, "" );
      }                   

      // Remove the file:
      */

}



void RooStats::HistFactory::Measurement::CollectHistograms() {

  for( unsigned int chanItr = 0; chanItr < fChannels.size(); ++chanItr) {

    RooStats::HistFactory::Channel& chan = fChannels.at( chanItr );
    
    chan.CollectHistograms();

  }

}



