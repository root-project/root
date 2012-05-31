// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
  BEGIN_HTML
  <p>
  </p>
  END_HTML
*/
//

#include "TDOMParser.h"

#include "RooStats/HistFactory/ConfigParser.h"
#include "RooStats/HistFactory/HistFactoryException.h"
#include "RooStats/HistFactory/Measurement.h"

#include "Helper.h"


using namespace RooStats;
using namespace HistFactory;

using namespace std;

std::vector< RooStats::HistFactory::Measurement > ConfigParser::GetMeasurementsFromXML( string input ) {
 
  // Open an input "Driver" XML file (input),
  // Parse that file and its channel files
  // and return a vector filled with 
  // the listed measurements


  // Create the list of measurements
  // (This is what will be returned)
  std::vector< HistFactory::Measurement > measurement_list;

  try {

    // Open the Driver XML File
    TDOMParser xmlparser;
    Int_t parseError = xmlparser.ParseFile( input.c_str() );
    if( parseError ) { 
      std::cerr << "Loading of xml document \"" << input
		<< "\" failed" << std::endl;
    } 


    // Read the Driver XML File
    cout << "reading input : " << input << endl;
    TXMLDocument* xmldoc = xmlparser.GetXMLDocument();
    TXMLNode* rootNode = xmldoc->GetRootNode();


    // Check that it is the proper DOCTYPE
    if( rootNode->GetNodeName() != TString( "Combination" ) ){
      std::cout << "Error: Driver DOCTYPE not equal to 'Combination'" << std::endl;
      throw hf_exc();
    }

    // Loop over the Combination's attributes
    std::string OutputFilePrefix;

    TListIter attribIt = rootNode->GetAttributes();
    TXMLAttr* curAttr = 0;
    while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

      // Get the Name, Val of this node
      TString attrName    = curAttr->GetName();
      std::string attrVal = curAttr->GetValue();

      if( attrName == TString( "" ) ) {
	std::cout << " Error: Attribute for 'Combination' with no name found" << std::endl;
	throw hf_exc();
      }

      else if( attrName == TString( "OutputFilePrefix" ) ) {
	OutputFilePrefix = string(curAttr->GetValue());
	std::cout << "output file prefix is : " << OutputFilePrefix << endl;
      }

      /*
	else if( attrName == TString( "InputFile" ) ) {
        channel.InputFile = attrVal ;
	}
      */

      else {
	std::cout << " Error: Unknown attribute for 'Combination' encountered: " 
		  << attrName << std::endl;
	throw hf_exc();
      }

      // node = node->GetNextNode();

    }

    TXMLNode* node = NULL;

    // Get the list of channel XML files to combine
    // Do this first so we can quickly exit
    // if no channels are found
    std::vector< std::string > xml_channel_files;
    node = rootNode->GetChildren();
    while( node != 0 ) {
      if( node->GetNodeName() == TString( "Input" ) ) {
	if( node->GetText() == NULL ) {
	  std::cout << "Error: node: " << node->GetName() 
		    << " has no text." << std::endl;
	  throw hf_exc();
	}
	xml_channel_files.push_back(node->GetText());
      }
      node = node->GetNextNode();
    }

    // If no channel xml files are found, exit
    if(xml_channel_files.empty()){
      cerr << "no input channels found" << endl;
      throw hf_exc();
      //return measurement_list;
    }
    else {
      std::cout << "Found Channels: ";
      for( unsigned int i=0; i < xml_channel_files.size(); ++i )   std::cout << " " << xml_channel_files.at(i);
      std::cout << std::endl;
    }

    // Get the list of functions
    // These apply to all measurements

    // For now, we create this list twice
    // simply for compatability
    std::vector< std::string > preprocessFunctions;
    std::vector< RooStats::HistFactory::PreprocessFunction > functionObjects;

    node = rootNode->GetChildren();
    while( node != 0 ) {
      if( node->GetNodeName() == TString( "Function" ) ) {
      
	// For now, add both the objects itself and
	// it's command string (for easy compatability)
	RooStats::HistFactory::PreprocessFunction Func = ParseFunctionConfig( node );
	preprocessFunctions.push_back( Func.GetCommand() ); 
	functionObjects.push_back( Func );
      }
      node = node->GetNextNode();
    }

    std::cout << std::endl;


    // Fill the list of measurements
    node = rootNode->GetChildren();
    while( node != 0 ) {

      if( node->GetNodeName() == TString( "" ) ) {
	std::cout << "Error: Node found in Measurement Driver XML with no name" << std::endl;
	throw hf_exc();
      }

      else if( node->GetNodeName() == TString( "Measurement" ) ) {
	HistFactory::Measurement measurement = CreateMeasurementFromDriverNode( node );
	// Set the prefix (obtained above)
	measurement.SetOutputFilePrefix( OutputFilePrefix );
	measurement_list.push_back( measurement );
      }

      else if( node->GetNodeName() == TString( "Function" ) ) {
	// Already processed these (directly above)
	;
      }

      else if( node->GetNodeName() == TString( "Input" ) ) {
	// Already processed these (directly above)
	;
      }

      else if( IsAcceptableNode( node ) ) { ; }
    
      else {
	std::cout << "Error: Unknown node found in Measurement Driver XML: "
		  << node->GetNodeName() << std::endl;
	throw hf_exc();
      }

      node = node->GetNextNode();

    }

    std::cout << "Done Processing Measurements" << std::endl;

    if( measurement_list.size() == 0 ) {
      std::cout << "Error: No Measurements found in XML Driver File" << std::endl;
      throw hf_exc();
    }
    else {
      std::cout << "Found Measurements: ";
      for( unsigned int i=0; i < measurement_list.size(); ++i )   std::cout << " " << measurement_list.at(i).GetName();
      std::cout << std::endl;
    }


    // Add the preprocessed functions to each measurement
    for( unsigned int i = 0; i < measurement_list.size(); ++i) {
      measurement_list.at(i).SetPreprocessFunctions( preprocessFunctions );
    }
    // Add the preprocessed functions to each measurement
    for( unsigned int i = 0; i < measurement_list.size(); ++i) {
      measurement_list.at(i).SetFunctionObjects( functionObjects );
    }


    // Create an instance of the class
    // that collects histograms
    //HistCollector collector;

    // Create the list of channels
    // (Each of these will be added 
    //  to every measurement)
    std::vector< HistFactory::Channel > channel_list;

    // Fill the list of channels
    for( unsigned int i = 0; i < xml_channel_files.size(); ++i ) {
      std::string channel_xml = xml_channel_files.at(i);
      std::cout << "Parsing Channel: " << channel_xml << std::endl;
      HistFactory::Channel channel =  ParseChannelXMLFile( channel_xml );

      // Get the histograms for the channel
      //collector.CollectHistograms( channel );
      //channel.CollectHistograms();
      channel_list.push_back( channel );
    }

    // Finally, add the channels to the measurements:
    for( unsigned int i = 0; i < measurement_list.size(); ++i) {

      HistFactory::Measurement& measurement = measurement_list.at(i);

      for( unsigned int j = 0; j < channel_list.size(); ++j ) {
	measurement.GetChannels().push_back( channel_list.at(j) );
      }

    }


  }
  catch(std::exception& e)
    {
      std::cout << e.what() << std::endl;
      throw hf_exc();
    }



  return measurement_list;


}
									     

// DEPRECATED:
/*
void ConfigParser::FillMeasurementsAndChannelsFromXML(string input, 
		   std::vector< RooStats::HistFactory::Measurement >& measurement_list,
		   std::vector< RooStats::HistFactory::Channel >&     channel_list ) {

  
  // Open an input "Driver" XML file (input),
  // Parse that file and its channel files
  // and fill the input vectors with the lists of
  // measurements and channels


  // Open the Driver XML File
  TDOMParser xmlparser;
  Int_t parseError = xmlparser.ParseFile( input.c_str() );
  if( parseError ) { 
    std::cerr << "Loading of xml document \"" << input
          << "\" failed" << std::endl;
  } 


  // Read the Driver XML File
  cout << "reading input : " << input << endl;
  TXMLDocument* xmldoc = xmlparser.GetXMLDocument();
  TXMLNode* rootNode = xmldoc->GetRootNode();


  // Check that it is the proper DOCTYPE
  if( rootNode->GetNodeName() != TString( "Combination" ) ){
    std::cout << "Error: Driver DOCTYPE not equal to 'Combination'" << std::endl;
    throw hf_exc();
  }

  // Loop over the Combination's attributes
  std::string OutputFilePrefix;

  TListIter attribIt = rootNode->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

      // Get the Name, Val of this node
      TString attrName    = curAttr->GetName();
      std::string attrVal = curAttr->GetValue();

      if( attrName == TString( "" ) ) {
	std::cout << " Error: Attribute for 'Combination' with no name found" << std::endl;
	throw hf_exc();
      }

      else if( attrName == TString( "OutputFilePrefix" ) ) {
	OutputFilePrefix = string(curAttr->GetValue());
	std::cout << "output file prefix is : " << OutputFilePrefix << endl;
      }

      / *
      else if( attrName == TString( "InputFile" ) ) {
        channel.InputFile = attrVal ;
      }
      * /

      else {
	std::cout << " Error: Unknown attribute for 'Combination' encountered: " 
		  << attrName << std::endl;
	throw hf_exc();
      }

      // node = node->GetNextNode();

  }

  TXMLNode* node = NULL;

  // Get the list of channel XML files to combine
  // Do this first so we can quickly exit
  // if no channels are found
  std::vector< std::string > xml_channel_files;
  node = rootNode->GetChildren();
  while( node != 0 ) {
    if( node->GetNodeName() == TString( "Input" ) ) {
      xml_channel_files.push_back(node->GetText());
    }
    node = node->GetNextNode();
  }

  // If no channel xml files are found, exit
  if(xml_channel_files.empty()){
    cerr << "no input channels found" << endl;
    exit(1);
  }
  else {
    std::cout << "Found Channels: ";
    for( unsigned int i=0; i < xml_channel_files.size(); ++i )   std::cout << " " << xml_channel_files.at(i);
    std::cout << std::endl;
  }

  // Get the list of functions
  // These apply to all measurements
  std::vector< std::string > preprocessFunctions;
  node = rootNode->GetChildren();
  while( node != 0 ) {
    if( node->GetNodeName() == TString( "Function" ) ) {
      preprocessFunctions.push_back( ParseFunctionConfig( node ) ); 
    }
    node = node->GetNextNode();
  }

  std::cout << std::endl;

  // Fill the list of measurements
  //std::vector< HistFactory::Measurement > measurement_list;
  node = rootNode->GetChildren();
  while( node != 0 ) {

    if( node->GetNodeName() == TString( "" ) ) {
      std::cout << "Error: Node found in Measurement Driver XML with no name" << std::endl;
      throw hf_exc();
    }

    else if( node->GetNodeName() == TString( "Measurement" ) ) {
      HistFactory::Measurement measurement = CreateMeasurementFromDriverNode( node );
      // Set the prefix (obtained above)
      measurement.OutputFilePrefix = OutputFilePrefix;
      measurement_list.push_back( measurement );
    }

    else if( node->GetNodeName() == TString( "Function" ) ) {
      // Already processed these (directly above)
      ;
    }

    else if( node->GetNodeName() == TString( "Input" ) ) {
      // Already processed these (directly above)
      ;
    }

    else if( IsAcceptableNode( node ) ) { ; }
    
    else {
      std::cout << "Error: Unknown node found in Measurement Driver XML: "
		<< node->GetNodeName() << std::endl;
      throw hf_exc();
    }

    node = node->GetNextNode();

  }

  std::cout << "Done Processing Measurements" << std::endl;

  if( measurement_list.size() == 0 ) {
    std::cout << "Error: No Measurements found in XML Driver File" << std::endl;
    throw hf_exc();
  }
  else {
    std::cout << "Found Measurements: ";
    for( unsigned int i=0; i < measurement_list.size(); ++i )   std::cout << " " << measurement_list.at(i).GetName();
    std::cout << std::endl;
  }


  // Add the preprocessed functions to each measurement
  for( unsigned int i = 0; i < measurement_list.size(); ++i) {
    measurement_list.at(i).preprocessFunctions = preprocessFunctions;
  }

  // Create an instance of the class
  // that collects histograms
  HistCollector collector;

  // Fill the list of channels
  // std::vector< HistFactory::Channel > channel_list;
  for( unsigned int i = 0; i < xml_channel_files.size(); ++i ) {
    std::string channel_xml = xml_channel_files.at(i);
    std::cout << "Parsing Channel: " << channel_xml << std::endl;
    HistFactory::Channel channel =  ParseChannelXMLFile( channel_xml );

    // Get the histograms for the channel
    collector.CollectHistograms( channel );

    channel_list.push_back( channel );
  }

  // Finally, add the channels to the measurements:
  for( unsigned int i = 0; i < measurement_list.size(); ++i) {

    HistFactory::Measurement& measurement = measurement_list.at(i);

    for( unsigned int j = 0; j < channel_list.size(); ++j ) {
      measurement.channels.push_back( channel_list.at(j) );
    }

  }

}  
*/

  // At this point, we have fully processed
  // the XML.  Thus, we are done.

  // Remember, the vectors:
  //  - measurement_list
  //  - channel_list 
  // are filled by reference from this
  // function's argument list
  // Cheers.


  // --------------------------------------------------------------- //
  // --------------------------------------------------------------- //

  /*
  // At this point, we have all the information we need
  // from the xml files.
  
  // We will make the measurements 1-by-1
  // This part will be migrated to the
  // MakeModelAndMeasurements function,
  // but is here for now.

  for(unsigned int i = 0; i < measurement_list.size(); ++i) {

    HistFactory::Measurement measurement = measurement_list.at(i);

    // Add the channels to this measurement
    for( unsigned int chanItr = 0; chanItr < channel_list.size(); ++chanItr ) {
      measurement.channels.push_back( channel_list.at( chanItr ) );
    }

    
    // Create the workspaces for the channels
    vector<RooWorkspace*> channel_workspaces;
    vector<string>        channel_names;
    TFile* outFile = new TFile(outputFileName.c_str(), "recreate");
    HistoToWorkspaceFactory factory(outputFileNamePrefix, rowTitle, systToFix, nominalLumi, lumiError, lowBin, highBin , outFile);


    // Loop over channels and make the individual 
    // channel fits:


    // read the xml for each channel and combine

    for( unsigned int chanItr = 0; chanItr < channel_list.size(); ++chanItr ) {

      HistFactory::Channel channel = channel_list.at( chanItr );

      string ch_name=channel.Name;
      channel_names.push_back(ch_name);

      RooWorkspace* ws = factory.MakeSingleChannelModel( channel );
      channel_workspaces.push_back(ws);

      // set poi in ModelConfig
      ModelConfig* proto_config = (ModelConfig *) ws->obj("ModelConfig");

      std::cout << "Setting Parameter of Interest as :" << measurement.POI << endl;
      RooRealVar* poi = (RooRealVar*) ws->var( (measurement.POI).c_str() );
      RooArgSet * params= new RooArgSet;
      if(poi){
	params->add(*poi);
      }
      proto_config->SetParametersOfInterest(*params);


      // Gamma/Uniform Constraints:
      // turn some Gaussian constraints into Gamma/Uniform/LogNorm constraints, rename model newSimPdf
      if( measurement.gammaSyst.size()>0 || measurement.uniformSyst.size()>0 || measurement.logNormSyst.size()>0) {
	factory.EditSyst( ws, ("model_"+ch_name).c_str(), gammaSyst, uniformSyst, logNormSyst);
	proto_config->SetPdf( *ws->pdf("newSimPdf") );
      }

      // fill out ModelConfig and export
      RooAbsData* expData = ws->data("expData");
      if(poi){
	proto_config->GuessObsAndNuisance(*expData);
      }
      ws->writeToFile( (outputFileNamePrefix+"_"+ch_name+"_"+rowTitle+"_model.root").c_str() );

      // do fit unless exportOnly requested
      if(!exportOnly){
	if(!poi){
	  cout <<"can't do fit for this channel, no parameter of interest"<<endl;
	} else{
	  factory.FitModel(ws, ch_name, "newSimPdf", "expData", false);
	}
      }
      fprintf(factory.pFile, " & " );
    }


    // Now, combine the channels
    RooWorkspace* ws=factory.MakeCombinedModel(channel_names, channel_workspaces);
    // Gamma/Uniform Constraints:
    // turn some Gaussian constraints into Gamma/Uniform/logNormal constraints, rename model newSimPdf
    if(gammaSyst.size()>0 || uniformSyst.size()>0 || logNormSyst.size()>0) 
      factory.EditSyst(ws, "simPdf", gammaSyst, uniformSyst, logNormSyst);
    //
    // set parameter of interest according to the configuration
    //
    ModelConfig * combined_config = (ModelConfig *) ws->obj("ModelConfig");
    cout << "Setting Parameter of Interest as :" << measurement.POI << endl;
    RooRealVar* poi = (RooRealVar*) ws->var( (measurement.POI).c_str() );
    //RooRealVar* poi = (RooRealVar*) ws->var((POI+"_comb").c_str());
    RooArgSet * params= new RooArgSet;
    cout << poi << endl;
    if(poi){
      params->add(*poi);
    }
    combined_config->SetParametersOfInterest(*params);
    ws->Print();

    // Set new PDF if there are gamma/uniform constraint terms
    if(gammaSyst.size()>0 || uniformSyst.size()>0 || logNormSyst.size()>0) 
      combined_config->SetPdf(*ws->pdf("newSimPdf"));

    RooAbsData* simData = ws->data("simData");
    combined_config->GuessObsAndNuisance(*simData);
    //	  ws->writeToFile(("results/model_combined_edited.root").c_str());
    ws->writeToFile( (outputFileNamePrefix+"_combined_"+rowTitle+"_model.root").c_str() );

    // TO DO:
    // Totally factorize the statistical test in "fit Model" to a different area
    if(!exportOnly){
      if(!poi){
	cout <<"can't do fit for this channel, no parameter of interest"<<endl;
      } else{
	factory.FitModel(ws, "combined", "simPdf", "simData", false);
      }
    }


  } // End Loop over measurement_list

  // Done
  */



HistFactory::Measurement ConfigParser::CreateMeasurementFromDriverNode( TXMLNode* node ) {


  HistFactory::Measurement measurement;

  // Set the default values:
  measurement.SetLumi( 1.0 );
  measurement.SetLumiRelErr( .10 );
  measurement.SetBinLow( 0 );
  measurement.SetBinHigh( 1 );
  measurement.SetExportOnly( false );

  std::cout << "Creating new measurement: " << std::endl;

  // First, get the attributes of the node
  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    if( curAttr->GetName() == TString( "" ) ) {
      std::cout << "Found XML attribute in Measurement with no name"  << std::endl;
      // ADD Output Here
      throw hf_exc();
    }
    else if( curAttr->GetName() == TString( "Name" ) ) {
      //rowTitle=curAttr->GetValue();
      measurement.SetName(  curAttr->GetValue() );
      //measurement.OutputFileName = outputFileNamePrefix+"_"+rowTitle+".root";
    }
    else if( curAttr->GetName() == TString( "Lumi" ) ) {
      measurement.SetLumi( atof(curAttr->GetValue()) );
    }
    else if( curAttr->GetName() == TString( "LumiRelErr" ) ) {
      measurement.SetLumiRelErr( atof(curAttr->GetValue()) );
    }
    else if( curAttr->GetName() == TString( "BinLow" ) ) {
      measurement.SetBinLow( atoi(curAttr->GetValue()) );
    }
    else if( curAttr->GetName() == TString( "BinHigh" ) ) {
      measurement.SetBinHigh( atoi(curAttr->GetValue()) );
    }
    else if( curAttr->GetName() == TString( "Mode" ) ) {
      cout <<"\n INFO: Mode attribute is deprecated, will ignore\n"<<endl;
    }
    else if( curAttr->GetName() == TString( "ExportOnly" ) ) {
      measurement.SetExportOnly( CheckTrueFalse(curAttr->GetValue(),"Measurement") );
    }

    else {
      std::cout << "Found unknown XML attribute in Measurement: " << curAttr->GetName()
		<< std::endl;
      throw hf_exc();
    }

  } // End Loop over attributes



  // Then, get the properties of the children nodes
  TXMLNode* child = node->GetChildren();
  while( child != 0 ) {
  
    if( child->GetNodeName() == TString( "" ) ) {
      std::cout << "Found XML child node of Measurement with no name"  << std::endl;
      throw hf_exc();
    }

    else if( child->GetNodeName() == TString( "POI" ) ) {
      if( child->GetText() == NULL ) {
	std::cout << "Error: node: " << child->GetName() 
		  << " has no text." << std::endl;
	throw hf_exc();
      }
      measurement.SetPOI( child->GetText() );
    }

    else if( child->GetNodeName() == TString( "ParamSetting" ) ) {
      TListIter paramIt = child->GetAttributes();
      TXMLAttr* curParam = 0;
      while( ( curParam = dynamic_cast< TXMLAttr* >( paramIt() ) ) != 0 ) {
	if( curParam->GetName() == TString( "Const" ) ) {
	  if(curParam->GetValue()==TString("True")){
	    // Fix here...?
	    if( child->GetText() == NULL ) {
	      std::cout << "Error: node: " << child->GetName() 
			<< " has no text." << std::endl;
	      throw hf_exc();
	    }
	    AddSubStrings( measurement.GetConstantParams(), child->GetText() );
	  }
	}
      }
    }

    else if( child->GetNodeName() == TString( "ConstraintTerm" ) ) {
      vector<string> syst; 
      string type = ""; 
      double rel = 0;

      map<string,double> gammaSyst;
      map<string,double> uniformSyst;
      map<string,double> logNormSyst;

      // Get the list of parameters in this tag:
      if( child->GetText() == NULL ) {
	std::cout << "Error: node: " << child->GetName() 
		  << " has no text." << std::endl;
	throw hf_exc();
      }
      AddSubStrings(syst, child->GetText());

      // Now, loop over this tag's attributes
      attribIt = child->GetAttributes();
      curAttr = 0;
      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

	if( curAttr->GetName() == TString( "" ) ) {
	  std::cout << "Error: Found tag attribute with no name in ConstraintTerm" << std::endl;
	  throw hf_exc();
	}

	else if( curAttr->GetName() == TString( "Type" ) ) {
	  type = curAttr->GetValue();
	}

	else if( curAttr->GetName() == TString( "RelativeUncertainty" ) ) {
	  rel = atof(curAttr->GetValue());
	}

	else {
	  std::cout << "Found tag attribute with unknown name in ConstraintTerm: "
		    << curAttr->GetName() << std::endl;
	  throw hf_exc();
	}

      } // End Loop over tag attributes


      // Now, fill the maps, depending on the type:

      // Check that the type is in the correct form:
      if( ! (type=="Gamma"     || type=="Uniform" || 
	     type=="LogNormal" || type=="NoConstraint") ) {
	std::cout << "Error: Encountered unknown type for ConstraintTerm: " << type << std::endl;
	throw hf_exc();
      }


      if (type=="Gamma" && rel!=0) {
	for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) {
	  // Fix Here...?
	  measurement.GetGammaSyst()[(*it).c_str()] = rel;
	}
      }
	
      if (type=="Uniform" && rel!=0) {
	for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) {
	  // Fix Here...?
	  measurement.GetUniformSyst()[(*it).c_str()] = rel;
	}
      }
	
      if (type=="LogNormal" && rel!=0) {
	for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) {
	  // Fix Here...?
	  measurement.GetLogNormSyst()[(*it).c_str()] = rel;
	}
      }
	
      if (type=="NoConstraint") {
	for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) {
	  // Fix Here...?
	  measurement.GetNoSyst()[(*it).c_str()] = 1.0; // MB : dummy value
	}
      }

    } // End adding of Constraint terms


    else if( IsAcceptableNode( child ) ) { ; }

    else {
    std::cout << "Found XML child of Measurement with unknown name: " << child->GetNodeName()
		<< std::endl;
      throw hf_exc();
    }

    child = child->GetNextNode();

  }

  measurement.PrintTree();
  std::cout << std::endl;

  return measurement;

}





// Deprecated
/*
HistFactory::Channel ConfigParser::ReadXmlConfig( string filen, Double_t lumi ) {

  

  //  Open an xml file, 
  //  parse it, and return
  //  a list of channels


  TString lumiStr;
  lumiStr+=lumi;
  lumiStr.ReplaceAll(' ', TString());

  std::cout << "Parsing file: " << filen ;

  TDOMParser xmlparser;
  string channelName;
  string inputFileName;
  string histoName;
  string histoPathName;

  // reading in the file and parse by DOM
  Int_t parseError = xmlparser.ParseFile( filen.c_str() );
  if( parseError ) { 
    std::cout << "Loading of xml document \"" << filen
	      << "\" failed" << std::endl;
  } 


  HistFactory::Measurement measurement;

  TXMLDocument* xmldoc = xmlparser.GetXMLDocument();
  TXMLNode* rootNode = xmldoc->GetRootNode();

  // not assuming that combination is the only option
  // single channel is also ok

  if( rootNode->GetNodeName() == TString( "" ) ){

    std::cout << "Error: Encounterd XML with no DOCTYPE" << std::endl;
    throw hf_exc();

  }


  else if( rootNode->GetNodeName() == TString( "Channel" ) ){

    HistFactory::Channel channel = ParseChannelXMLFile( rootNode );
    measurement.channels.push_back( channel );

  }

  else if( rootNode->GetNodeName() == TString( "Combination" ) ){

    ConfigureMeasurementFromDriverXML( measurement );

    std::cout << "Stuff" << std::endl;

  }
  // above two are the only options
  else {
    std::cout << "Found XML file with unknown DOCTYPE: " << rootNode->GetNodeName()
	      << std::endl;
    throw hf_exc();
  }


}
*/


HistFactory::Channel ConfigParser::ParseChannelXMLFile( string filen ) {

  /*
  TString lumiStr;
  lumiStr+=lumi;
  lumiStr.ReplaceAll(' ', TString());
  */

  std::cout << "Parsing file: " << filen ;

  TDOMParser xmlparser;

  // reading in the file and parse by DOM
  Int_t parseError = xmlparser.ParseFile( filen.c_str() );
  if( parseError ) { 
    std::cout << "Loading of xml document \"" << filen
	      << "\" failed" << std::endl;
  } 


  TXMLDocument* xmldoc = xmlparser.GetXMLDocument();
  TXMLNode* rootNode = xmldoc->GetRootNode();

  // Check that is is a CHANNEL based on the DOCTYPE

  if( rootNode->GetNodeName() != TString( "Channel" ) ){
    std::cout << "Error: In parsing a Channel XML, " 
	      << "Encounterd XML with DOCTYPE: " << rootNode->GetNodeName() 
	      << std::endl;
    std::cout << " DOCTYPE for channels must be 'Channel' "
	      << " Check that your XML is properly written" << std::endl;
    throw hf_exc();
  }

  // Now, create the channel, 
  // configure it based on the XML
  // and return it

  HistFactory::Channel channel;

  // Set the default values:
  
  // Walk through the root node and
  // get its attributes
  TListIter attribIt = rootNode->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ) {
      std::cout << " Error: Attribute for 'Channel' with no name found" << std::endl;
      throw hf_exc();
    }

    else if( attrName == TString( "Name" ) ) {
      channel.SetName( attrVal );
      std::cout << " : creating a channel named " << channel.GetName() << std::endl;
    }

    else if( attrName == TString( "InputFile" ) ) {
      std::cout << "Setting InputFile for this channel: " << attrVal << std::endl;
      channel.SetInputFile( attrVal );
      // Set the current (cached) value
      m_currentInputFile = attrVal;
    }

    else if( curAttr->GetName() == TString( "HistoPath" ) ) {
      std::cout << "Setting HistoPath for this channel: " << attrVal << std::endl;
      // Set the current (cached) value
      channel.SetHistoPath( attrVal );
      m_currentHistoPath = attrVal;
    }

    else if( curAttr->GetName() == TString( "HistoName" ) ) {
      // Changed This:
      std::cout << "Use of HistoName in Channel is deprecated" << std::endl;
      std::cout << "This will be ignored" << std::endl;
    }
      
    else {
      std::cout << " Error: Unknown attribute for 'Channel' encountered: " 
		<< attrName << std::endl;
      throw hf_exc();
    }


  } // End loop over the channel tag's attributes
    

  // Check that the channel was properly initiated:
  
  if( channel.GetName() == "" ) {
    std::cout << "Error: Channel created with no name" << std::endl;
    throw hf_exc();
  }

  m_currentChannel = channel.GetName();

  // Loop over the children nodes in the XML file
  // and configure the channel based on them

  TXMLNode* node = rootNode->GetChildren();

  while( node != 0 ) {

    if( node->GetNodeName() == TString( "" ) ) {
      std::cout << "Error: Encountered node in Channel with no name" << std::endl;
      throw hf_exc();
    }

    else if( node->GetNodeName() == TString( "Data" ) ) {
      channel.SetData( CreateDataElement(node) );
    }

    else if( node->GetNodeName() == TString( "StatErrorConfig" ) ) {
      channel.SetStatErrorConfig( CreateStatErrorConfigElement(node) );
    }

    else if( node->GetNodeName() == TString( "Sample" ) ) {
      channel.GetSamples().push_back( CreateSampleElement(node) );
    }

    else if( IsAcceptableNode( node ) ) { ; }

    else {
      std::cout << "Error: Encountered node in Channel with unknown name: " << node->GetNodeName() << std::endl;
      throw hf_exc();
    }

    node = node->GetNextNode();

  } // End loop over tags in this channel


  std::cout << "Created Channel: " << std::endl;
  channel.Print();

  return channel;

}




    /*

    // Setup default values:
    EstimateSummary::ConstraintType StatConstraintType = EstimateSummary::Gaussian; //"Gaussian";
    Double_t RelErrorThreshold = .05;
    node = rootNode->GetChildren();
    while( node != 0 ) {
      if( node->GetNodeName() == TString( "StatErrorConfig" ) ) {
	
	// Loop over the node's attributes
        attribIt = node->GetAttributes();
        curAttr = 0;
        while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

          if( curAttr->GetName() == TString( "RelErrorThreshold" ) ) {
            RelErrorThreshold = atof(curAttr->GetValue()) ;
          }

          if( curAttr->GetName() == TString( "ConstraintType" ) ) {
	    // Allowable Values:  Gaussian
	    if( curAttr->GetValue()==TString("Gaussian") || curAttr->GetValue()==TString("Gauss")  )    StatConstraintType = EstimateSummary::Gaussian;
	    else if( curAttr->GetValue()==TString("Poisson") || curAttr->GetValue()==TString("Pois")  ) StatConstraintType = EstimateSummary::Poisson;
	    else cout << "Invalid Stat Constraint Type: " << curAttr->GetValue() << endl;
          }
	} // End: Loop Over Attributes

      } // End: Loop Over Nodes
	node = node->GetNextNode();
    }


    node = rootNode->GetChildren();
    while( node != 0 ) {

      if( node->GetNodeName() == TString( "Sample" ) ) {
	inputFileName_cache=inputFileName;
	histoName_cache=histoName;
	histoPathName_cache=histoPathName;
	EstimateSummary sample_channel;
	sample_channel.channel=channelName;

	// Set the Stat Paramaters
	// (These are uniform over a channel, but
	// they are only used if a particular sample
	// requests to use them)
	sample_channel.StatConstraintType=StatConstraintType;
	sample_channel.RelErrorThreshold=RelErrorThreshold;

	// For each sample, include a possible
	// external histogram for the stat error:
	/ *
	string statErrorName="";
	string statErrorPath="";
	string statErrorFile="";
	* /
	attribIt = node->GetAttributes();
	curAttr = 0;
	while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	  if( curAttr->GetName() == TString( "Name" ) ) {
	    // name of the smaple
	    sample_channel.name=curAttr->GetValue();
	  }
	  if( curAttr->GetName() == TString( "InputFile" ) ) {
	    inputFileName = curAttr->GetValue() ;
	  }
	  if( curAttr->GetName() == TString( "HistoName" ) ) {
	    histoName = curAttr->GetValue() ;
	  }
	  if( curAttr->GetName() == TString( "HistoPath" ) ) {
	    histoPathName=curAttr->GetValue();
	  }
	  if( curAttr->GetName() == TString( "NormalizeByTheory" ) ) {
	    if((curAttr->GetValue()==TString("False"))){
	      sample_channel.normName=lumiStr;
	    }
	  }
	  / * 
	  if( curAttr->GetName() == TString( "IncludeStatError" ) ) { 
	    if((curAttr->GetValue()==TString("True"))){
	      sample_channel.IncludeStatError = true; // Added McStat
	    }
	  }
	  if( curAttr->GetName() == TString( "ShapeFactorName" ) ) {
	    sample_channel.shapeFactorName = curAttr->GetValue();
	  }
	  * /
	} // (Casting to TH1F* here...
	cout << "Getting nominal histogram: " << histoPathName << "/" << histoName << " in file " << inputFileName << endl;
	//sample_channel.nominal = (TH1F*) GetHisto(inputFileName, histoPathName, histoName);
	sample_channel.nominal = GetHisto(inputFileName, histoPathName, histoName);
	// Set the rel Error hist later,
	// if necessary
	//sample_channel.relStatError = NULL;

	TXMLNode* sys = node->GetChildren();
	AddSystematic(sample_channel, sys, inputFileName, histoPathName, histoName);
	summary.push_back(sample_channel);
	inputFileName=inputFileName_cache;
	histoName=histoName_cache;
	histoPathName=histoPathName_cache;
	//sample_channel.print();
      }
      node = node->GetNextNode();
    }

    */

/*
void HistFactory::AddSystematic( EstimateSummary & sample_channel, TXMLNode* node, string inputFileName, string histoPathName, string histoName){

  while( node != 0 ) {

    if( node->GetNodeName() == TString( "NormFactor" ) ){
      TListIter attribIt = node->GetAttributes();
      EstimateSummary::NormFactor norm;
      TXMLAttr* curAttr = 0;
      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	if( curAttr->GetName() == TString( "Name" ) ) {
	  norm.name = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "Val" ) ) {
	  norm.val = atof(curAttr->GetValue()) ;
	}
	if( curAttr->GetName() == TString( "Low" ) ) {
	  norm.low = atof(curAttr->GetValue()) ;
	}
	if( curAttr->GetName() == TString( "High" ) ) {
	  norm.high = atof(curAttr->GetValue()) ;
	}
	if( curAttr->GetName() == TString( "Const" ) ) {
	  norm.constant =  (curAttr->GetValue()==TString("True"));
	}
      }

      sample_channel.normFactor.push_back(norm);
    }

    if( node->GetNodeName() == TString( "HistoSys" ) ){
      TListIter attribIt = node->GetAttributes();
      TXMLAttr* curAttr = 0;
      string Name, histoPathHigh, histoPathLow, 
	histoNameLow, histoNameHigh, inputFileHigh, inputFileLow;
      inputFileLow=inputFileName; inputFileHigh=inputFileName;
      histoPathLow=histoPathName; histoPathHigh=histoPathName;
      histoNameLow=histoName; histoNameHigh=histoName;
      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	if( curAttr->GetName() == TString( "Name" ) ) {
	  Name = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "InputFileHigh" ) ) {
	  inputFileHigh = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoPathHigh" ) ) {
	  histoPathHigh = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoNameHigh" ) ) {
	  histoNameHigh = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "InputFileLow" ) ) {
	  inputFileLow = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoPathLow" ) ) {
	  histoPathLow = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoNameLow" ) ) {
	  histoNameLow = curAttr->GetValue() ;
	}
      }
      //sample_channel.AddSyst(Name, 
      //		     (TH1F*) GetHisto( inputFileLow, histoPathLow, histoNameLow),
      //		     (TH1F*) GetHisto( inputFileHigh, histoPathHigh, histoNameHigh));
      sample_channel.AddSyst(Name, 
			     GetHisto( inputFileLow, histoPathLow, histoNameLow),
			     GetHisto( inputFileHigh, histoPathHigh, histoNameHigh));
    }

    if( node->GetNodeName() == TString( "OverallSys" ) ){
      TListIter attribIt = node->GetAttributes();
      TXMLAttr* curAttr = 0;
      string Name;
      Double_t Low=0, High=0;
      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	if( curAttr->GetName() == TString( "Name" ) ) {
	  Name = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "High" ) ) {
	  High = atof(curAttr->GetValue()) ;
	}
	if( curAttr->GetName() == TString( "Low" ) ) {
	  Low = atof(curAttr->GetValue()) ;
	}
      }
      sample_channel.overallSyst[Name] = UncertPair(Low, High); 
    }


    if( node->GetNodeName() == TString( "StatError" ) ){
      TListIter attribIt = node->GetAttributes();
      TXMLAttr* curAttr = 0;
      bool   statErrorActivate = false;
      string statHistName = "";
      string statHistPath = "";
      string statHistFile = inputFileName;

      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	if( curAttr->GetName() == TString( "Activate" ) ) {
	  if( curAttr->GetValue() == TString("True") ) statErrorActivate = true;
	}
	if( curAttr->GetName() == TString( "InputFile" ) ) {
	  statHistFile = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoName" ) ) {
	  statHistName = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoPath" ) ) {
	  statHistPath = curAttr->GetValue() ;
	}
      }
      if( statErrorActivate) {
	sample_channel.IncludeStatError = true;
	// Get an external histogram if necessary
	if( statHistName != "" ) {
	  cout << "Getting rel StatError histogram: " << statHistPath << "/" << statHistName << " in file " << statHistFile << endl;
	  sample_channel.relStatError = (TH1*) GetHisto(statHistFile, statHistPath, statHistName);
	} else {
	  sample_channel.relStatError = NULL;
	}
      }
    } // END: StatError Node
    
    if( node->GetNodeName() == TString( "ShapeFactor" ) ){
      TListIter attribIt = node->GetAttributes();
      TXMLAttr* curAttr = 0;
      string Name="";
      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	if( curAttr->GetName() == TString( "Name" ) ) {
	  Name = curAttr->GetValue() ;
	}
      }
      sample_channel.shapeFactorName=Name; 
    } // END: ShapeFactor Node
    
    if( node->GetNodeName() == TString( "ShapeSys" ) ){
      TListIter attribIt = node->GetAttributes();
      TXMLAttr* curAttr = 0;
      string Name="";
      string HistoName;
      string HistoPath;
      string HistoFile = inputFileName;
      EstimateSummary::ConstraintType ConstraintType = EstimateSummary::Gaussian; //"Gaussian";

      while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
	if( curAttr->GetName() == TString( "Name" ) ) {
	  Name = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoName" ) ) {
	 HistoName = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoPath" ) ) {
	  HistoPath = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoFile" ) ) {
	  HistoFile = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "ConstraintType" ) ) {
	  if( curAttr->GetValue()==TString("Gaussian")     || curAttr->GetValue()==TString("Gauss") ) ConstraintType = EstimateSummary::Gaussian;
	  else if( curAttr->GetValue()==TString("Poisson") || curAttr->GetValue()==TString("Pois") )  ConstraintType = EstimateSummary::Poisson;
	}
      }
      // Now, set the EstimateSummary accordingly
      EstimateSummary::ShapeSys Sys;
      Sys.name = Name;
      Sys.hist = NULL;
      cout << "Getting rel ShapeSys constraint histogram: " << HistoPath << "/" << HistoName << " in file " << HistoFile << endl;
      Sys.hist = (TH1*) GetHisto(HistoFile, HistoPath, HistoName);
      if( Sys.hist == NULL ) {
	cout << "Failed to get histogram: " << HistoPath << "/" <<HistoName << " in file " << HistoFile << endl;
      }
      Sys.constraint = ConstraintType;
      sample_channel.shapeSysts.push_back( Sys );
    } // END: ShapeFactor Node


    node=node->GetNextNode();
  }


}
*/


HistFactory::Data ConfigParser::CreateDataElement( TXMLNode* node ) {

  std::cout << "Creating Data Element" << std::endl;

    HistFactory::Data data;

    // Set the default values
    data.SetInputFile( m_currentInputFile );
    data.SetHistoPath( m_currentHistoPath );
    //data.HistoName = m_currentHistoName;

    // Now, set the attributes
    TListIter attribIt = node->GetAttributes();
    TXMLAttr* curAttr = 0;
    while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

      // Get the Name, Val of this node
      TString attrName    = curAttr->GetName();
      std::string attrVal = curAttr->GetValue();

      if( attrName == TString( "" ) ) {
	std::cout << " Error: Attribute for 'Data' with no name found" << std::endl;
	throw hf_exc();
      }

      else if( attrName == TString( "InputFile" ) ) {
	data.SetInputFile( attrVal );
      }

      else if( attrName == TString( "HistoName" ) ) {
	data.SetHistoName( attrVal );
      }

      else if( attrName == TString( "HistoPath" ) ) {
	data.SetHistoPath( attrVal );
      }

    else if( IsAcceptableNode( node ) ) { ; }

      else {
	std::cout << " Error: Unknown attribute for 'Data' encountered: " << attrName << std::endl;
	throw hf_exc();
      }

    }

    // Check the properties of the data node:
    if( data.GetInputFile() == "" ) {
      std::cout << "Error: Data Node has no InputFile" << std::endl;
      throw hf_exc();
    }
    if( data.GetHistoName() == "" ) {
      std::cout << "Error: Data Node has no HistoName" << std::endl;
      throw hf_exc();
    }

    std::cout << "Created Data Node with"
	      << " InputFile: " << data.GetInputFile()
	      << " HistoName: " << data.GetHistoName()
	      << " HistoPath: " << data.GetHistoPath()
	      << std::endl;


    // data.hist = GetHisto(data.FileName, data.HistoPath, data.HistoName);

    return data;
}



HistFactory::StatErrorConfig ConfigParser::CreateStatErrorConfigElement( TXMLNode* node ) {

  std::cout << "Creating StatErrorConfig Element" << std::endl;
  
  HistFactory::StatErrorConfig config;

  // Setup default values:
  config.SetConstraintType( Constraint::Gaussian );
  config.SetRelErrorThreshold( 0.05 ); // 5%


  // Loop over the node's attributes
  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();
    
    if( attrName == TString( "RelErrorThreshold" ) ) {
      config.SetRelErrorThreshold( atof(attrVal.c_str()) );
    }
    
    if( attrName == TString( "ConstraintType" ) ) {
      // Allowable Values:  Gaussian

      if( attrVal == "" ) {
	std::cout << "Error: Bad Value for StatErrorConfig Constraint Type Found" << std::endl;
	throw hf_exc();
      }

      else if( attrVal=="Gaussian" || attrVal=="Gauss"  ) {  
	config.SetConstraintType( Constraint::Gaussian );
      }

      else if( attrVal=="Poisson" || attrVal=="Pois"  ) {
	config.SetConstraintType( Constraint::Poisson );
      }

      else if( IsAcceptableNode( node ) ) { ; }

      else {
	cout << "Invalid Stat Constraint Type: " << curAttr->GetValue() << endl;
	throw hf_exc();
      }

    }

  } // End: Loop Over Attributes

  std::cout << "Created StatErrorConfig Element with" 
	    << " Constraint type: " << config.GetConstraintType()
	    << " RelError Threshold: " << config.GetRelErrorThreshold()
	    << std::endl;

  return config;

}


HistFactory::Sample ConfigParser::CreateSampleElement( TXMLNode* node ) {

  std::cout << "Creating Sample Element" << std::endl;

  HistFactory::Sample sample;

  // Set the default values
  sample.SetInputFile( m_currentInputFile );
  sample.SetHistoPath( m_currentHistoPath );
  sample.SetChannelName( m_currentChannel );
  sample.SetNormalizeByTheory( true );
  //sample.HistoName = m_currentHistoName;

  // Now, set the attributes

  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ) {
      std::cout << " Error: Attribute for 'Sample' with no name found" << std::endl;
      throw hf_exc();
    }

    else if( attrName == TString( "Name" ) ) {
      sample.SetName( attrVal );
    }

    else if( attrName == TString( "InputFile" ) ) {
      sample.SetInputFile( attrVal );
    }

    else if( attrName == TString( "HistoName" ) ) {
      sample.SetHistoName( attrVal );
    }

    else if( attrName == TString( "HistoPath" ) ) {
      sample.SetHistoPath( attrVal );
    }

    else if( attrName == TString( "NormalizeByTheory" ) ) {
      sample.SetNormalizeByTheory( CheckTrueFalse(attrVal,"Sample") );
      /*
	if( attrVal == "" ) {
	std::cout << "Error: Attribute 'NormalizeByTheory' in Sample has no value" << std::endl;
	throw hf_exc();
	}
	else if ( attrVal == "True"  || attrVal == "true"  )   sample.NormalizeByTheory = true;
	else if ( attrVal == "False" || attrVal == "false" )   sample.NormalizeByTheory = false;
	else {
	std::cout << "Error: Attribute 'NormalizeByTheory' in Sample has unknown value: " << attrVal <<  std::endl;
	std::cout << "Value must be 'True' or 'False' " <<  std::endl;
	throw hf_exc();
	}
      */
    }

    else {
      std::cout << " Error: Unknown attribute for 'Sample' encountered: " << attrName << std::endl;
      throw hf_exc();
    }

  }

  // Quickly check the properties of the Sample Node
  if( sample.GetName() == "" ) {
    std::cout << "Error: Sample Node has no Name" << std::endl;
    throw hf_exc();
  }
  if( sample.GetInputFile() == "" ) {
    std::cout << "Error: Sample Node has no InputFile" << std::endl;
    throw hf_exc();
  }
  if( sample.GetHistoName() == "" ) {
    std::cout << "Error: Sample Node has no HistoName" << std::endl;
    throw hf_exc();
  }


  // Now, loop over the children and add the systematics

  TXMLNode* child = node->GetChildren();

  while( child != 0 ) {
      
    if( child->GetNodeName() == TString( "" ) ) {
      std::cout << "Error: Encountered node in Sample with no name" << std::endl;
      throw hf_exc();
    }

    else if( child->GetNodeName() == TString( "NormFactor" ) ) {
      sample.GetNormFactorList().push_back( MakeNormFactor( child ) );
    }

    else if( child->GetNodeName() == TString( "OverallSys" ) ) {
      sample.GetOverallSysList().push_back( MakeOverallSys( child ) );
    }

    else if( child->GetNodeName() == TString( "HistoSys" ) ) {
      sample.GetHistoSysList().push_back( MakeHistoSys( child ) );
    }

    else if( child->GetNodeName() == TString( "HistoFactor" ) ) {
      std::cout << "WARNING: HistoFactor not yet supported" << std::endl;
      //sample.GetHistoFactorList().push_back( MakeHistoFactor( child ) );
    }

    else if( child->GetNodeName() == TString( "ShapeSys" ) ) {
      sample.GetShapeSysList().push_back( MakeShapeSys( child ) );
    }

    else if( child->GetNodeName() == TString( "ShapeFactor" ) ) {
      sample.GetShapeFactorList().push_back( MakeShapeFactor( child ) );
    }

    else if( child->GetNodeName() == TString( "StatError" ) ) {
      sample.SetStatError( ActivateStatError(child) );
    }

    else if( IsAcceptableNode( child ) ) { ; }

    else {
      std::cout << "Error: Encountered node in Sample with unknown name: " << child->GetNodeName() << std::endl;
      throw hf_exc();
    }

    child=child->GetNextNode();
  }


  std::cout << "Created Sample Node with"
	    << " Name: " << sample.GetName()
	    << " InputFile: " << sample.GetInputFile()
	    << " HistoName: " << sample.GetHistoName()
	    << " HistoPath: " << sample.GetHistoPath()
	    << std::endl;


  // sample.hist = GetHisto(sample.FileName, sample.HistoPath, sample.HistoName);

  return sample;
}


HistFactory::NormFactor ConfigParser::MakeNormFactor( TXMLNode* node ) {

  std::cout << "Making NormFactor:" << std::endl;
  
  HistFactory::NormFactor norm;

  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ){
      std::cout << "Error: Encountered Element in NormFactor with no name" << std::endl;
      throw hf_exc();
    }

    else if( curAttr->GetName() == TString( "Name" ) ) {
      norm.SetName( attrVal );
    }
    else if( curAttr->GetName() == TString( "Val" ) ) {
      norm.SetVal( atof(attrVal.c_str()) );
    }
    else if( curAttr->GetName() == TString( "Low" ) ) {
      norm.SetLow( atof(attrVal.c_str()) );
    }
    else if( curAttr->GetName() == TString( "High" ) ) {
      norm.SetHigh( atof(attrVal.c_str()) );
    }
    else if( curAttr->GetName() == TString( "Const" ) ) {
      norm.SetConst( CheckTrueFalse(attrVal,"NormFactor") );
    }

    else {
      std::cout << "Error: Encountered Element in NormFactor with unknown name: " 
		<< attrName << std::endl;
      throw hf_exc();
    }

  } // End loop over properties

  if( norm.GetName() == "" ) {
    std::cout << "Error: NormFactor Node has no Name" << std::endl;
    throw hf_exc();
  }

  norm.Print();

  return norm;

}

HistFactory::HistoFactor ConfigParser::MakeHistoFactor( TXMLNode* /*node*/ ) {

  std::cout << "Making HistoFactor" << std::endl;

  HistFactory::HistoFactor dummy;

  dummy.SetInputFileLow( m_currentInputFile );
  dummy.SetHistoPathLow( m_currentHistoPath );

  dummy.SetInputFileHigh( m_currentInputFile );
  dummy.SetHistoPathHigh( m_currentHistoPath );

  std::cout << "Made HistoFactor" << std::endl;

  return dummy;

}


HistFactory::HistoSys ConfigParser::MakeHistoSys( TXMLNode* node ) {

  std::cout << "Making HistoSys:" << std::endl;

  HistFactory::HistoSys histoSys;

  histoSys.SetInputFileLow( m_currentInputFile );
  histoSys.SetHistoPathLow( m_currentHistoPath );

  histoSys.SetInputFileHigh( m_currentInputFile );
  histoSys.SetHistoPathHigh( m_currentHistoPath );


  // MUST SET DEFAULT VALUES!!!!J!LH:OIH:OIHO:H

  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  /*
  string Name, histoPathHigh, histoPathLow, 
    histoNameLow, histoNameHigh, inputFileHigh, inputFileLow;
  inputFileLow=inputFileName; inputFileHigh=inputFileName;
  histoPathLow=histoPathName; histoPathHigh=histoPathName;
  histoNameLow=histoName; histoNameHigh=histoName;
  */

  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();
	
    if( attrName == TString( "" ) ){
      std::cout << "Error: Encountered Element in HistoSys with no name" << std::endl;
      throw hf_exc();
    }

    else if( curAttr->GetName() == TString( "Name" ) ) {
      histoSys.SetName( attrVal );
    }

    else if( curAttr->GetName() == TString( "InputFileHigh" ) ) {
      histoSys.SetInputFileHigh( attrVal );
    }
    else if( curAttr->GetName() == TString( "HistoPathHigh" ) ) {
      histoSys.SetHistoPathHigh( attrVal );
    }
    else if( curAttr->GetName() == TString( "HistoNameHigh" ) ) {
      histoSys.SetHistoNameHigh( attrVal );
    }

    else if( curAttr->GetName() == TString( "InputFileLow" ) ) {
      histoSys.SetInputFileLow( attrVal );
    }
    else if( curAttr->GetName() == TString( "HistoPathLow" ) ) {
      histoSys.SetHistoPathLow( attrVal );
    }
    else if( curAttr->GetName() == TString( "HistoNameLow" ) ) {
      histoSys.SetHistoNameLow( attrVal );
    }

    else {
      std::cout << "Error: Encountered Element in HistoSys with unknown name: " 
		<< attrName << std::endl;
      throw hf_exc();
    }

  } // End loop over properties


  if( histoSys.GetName() == "" ) {
    std::cout << "Error: HistoSys Node has no Name" << std::endl;
    throw hf_exc();
  }
  if( histoSys.GetInputFileHigh() == "" ) {
    std::cout << "Error: HistoSysSample Node has no InputFileHigh" << std::endl;
    throw hf_exc();
  }
  if( histoSys.GetInputFileLow() == "" ) {
    std::cout << "Error: HistoSysSample Node has no InputFileLow" << std::endl;
    throw hf_exc();
  }
  if( histoSys.GetHistoNameHigh() == "" ) {
    std::cout << "Error: HistoSysSample Node has no HistoNameHigh" << std::endl;
    throw hf_exc();
  }
  if( histoSys.GetHistoNameLow() == "" ) {
    std::cout << "Error: HistoSysSample Node has no HistoNameLow" << std::endl;
    throw hf_exc();
  }


  histoSys.Print();

  return histoSys;

}


HistFactory::OverallSys ConfigParser::MakeOverallSys( TXMLNode* node ) {

  std::cout << "Making OverallSys:" << std::endl;

  HistFactory::OverallSys overallSys;

  // MUST SET DEFAULT VALUES!!!!J!LH:OIH:OIHO:H
  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ){
      std::cout << "Error: Encountered Element in OverallSys with no name" << std::endl;
      throw hf_exc();
    }

    else if( attrName == TString( "Name" ) ) {
      overallSys.SetName( attrVal );
    }
    else if( attrName == TString( "High" ) ) {
      overallSys.SetHigh( atof(attrVal.c_str()) );
    }
    else if( attrName == TString( "Low" ) ) {
      overallSys.SetLow( atof(attrVal.c_str()) );
    }

    else {
      std::cout << "Error: Encountered Element in OverallSys with unknown name: " 
		<< attrName << std::endl;
      throw hf_exc();
    }

  }

  if( overallSys.GetName() == "" ) {
    std::cout << "Error: Encountered OverallSys with no name" << std::endl;
    throw hf_exc();
  }
  

  overallSys.Print();

  return overallSys;

}


HistFactory::ShapeFactor ConfigParser::MakeShapeFactor( TXMLNode* node ) {

  std::cout << "Making ShapeFactor" << std::endl;

  HistFactory::ShapeFactor shapeFactor;

  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  string Name="";
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ){
      std::cout << "Error: Encountered Element in ShapeFactor with no name" << std::endl;
      throw hf_exc();
    }

    else if( attrName == TString( "Name" ) ) {
      shapeFactor.SetName( attrVal );
    }

    else {
      std::cout << "Error: Encountered Element in ShapeFactor with unknown name: " 
		<< attrName << std::endl;
      throw hf_exc();
    }

  }

  if( shapeFactor.GetName() == "" ) {
    std::cout << "Error: Encountered ShapeFactor with no name" << std::endl;
    throw hf_exc();
  }

  shapeFactor.Print();

  return shapeFactor;

}


HistFactory::ShapeSys ConfigParser::MakeShapeSys( TXMLNode* node ) {

  std::cout << "Making ShapeSys" << std::endl;

  HistFactory::ShapeSys shapeSys;
  shapeSys.SetConstraintType( Constraint::Gaussian );

  shapeSys.SetInputFile( m_currentInputFile );
  shapeSys.SetHistoPath( m_currentHistoPath );


  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  //EstimateSummary::ConstraintType ConstraintType = EstimateSummary::Gaussian; //"Gaussian";

  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {


    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ){
      std::cout << "Error: Encountered Element in ShapeSys with no name" << std::endl;
      throw hf_exc();
    }

    else if( attrName == TString( "Name" ) ) {
      shapeSys.SetName( attrVal );
    }

    else if( attrName == TString( "HistoName" ) ) {
      shapeSys.SetHistoName( attrVal );
    }

    else if( attrName == TString( "HistoPath" ) ) {
      shapeSys.SetHistoPath( attrVal );
    }

    else if( attrName == TString( "InputFile" ) ) {
      shapeSys.SetInputFile( attrVal );
    }

    else if( attrName == TString( "ConstraintType" ) ) {
      if( attrVal=="" ) {
	std::cout << "Error: ShapeSys Constraint type is empty" << std::endl;
	throw hf_exc();
      }
      else if( attrVal=="Gaussian" || attrVal=="Gauss" ) {
	shapeSys.SetConstraintType( Constraint::Gaussian );
      }
      else if( attrVal=="Poisson"  || attrVal=="Pois"  ) {
	shapeSys.SetConstraintType( Constraint::Poisson );
      }
      else {
	cout << "Error: Encountered unknown ShapeSys Constraint type: " << attrVal << endl;
	throw hf_exc();
      }
    }

    else {
      std::cout << "Error: Encountered Element in ShapeSys with unknown name: " 
		<< attrName << std::endl;
      throw hf_exc();
    }

  } // End loop over attributes


  if( shapeSys.GetName() == "" ) {
    std::cout << "Error: Encountered ShapeSys with no Name" << std::endl;
    throw hf_exc();
  }
  if( shapeSys.GetInputFile() == "" ) {
    std::cout << "Error: Encountered ShapeSys with no InputFile" << std::endl;
    throw hf_exc();
  }
  if( shapeSys.GetHistoName() == "" ) {
    std::cout << "Error: Encountered ShapeSys with no HistoName" << std::endl;
    throw hf_exc();
  }

  shapeSys.Print();

  return shapeSys;

}

  /*

  // Now, set the EstimateSummary accordingly
  EstimateSummary::ShapeSys Sys;
  Sys.name = Name;
  Sys.hist = NULL;
  cout << "Getting rel ShapeSys constraint histogram: " << HistoPath << "/" << HistoName << " in file " << HistoFile << endl;
  Sys.hist = (TH1*) GetHisto(HistoFile, HistoPath, HistoName);
  if( Sys.hist == NULL ) {
    cout << "Failed to get histogram: " << HistoPath << "/" <<HistoName << " in file " << HistoFile << endl;
  }
  Sys.constraint = ConstraintType;
  */


HistFactory::StatError ConfigParser::ActivateStatError( TXMLNode* node ) {
	
  std::cout << "Activating StatError" << std::endl;

  // Have to figiure this out:
  // Must use this channel's constraint type
  // as defined by the StatErrorConfig

  HistFactory::StatError statError;
  statError.Activate( false );
  statError.SetUseHisto( false );


  // Loop over the node's attributes
  TListIter attribIt = node->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {

    // Get the Name, Val of this node
    TString attrName    = curAttr->GetName();
    std::string attrVal = curAttr->GetValue();

    if( attrName == TString( "" ) ){
      std::cout << "Error: Encountered Element in ActivateStatError with no name" << std::endl;
      throw hf_exc();
    }
    
    else if( attrName == TString( "Activate" ) ) {
      statError.Activate( CheckTrueFalse(attrVal,"ActivateStatError") );
    }

    else if( attrName == TString( "HistoName" ) ) {
      statError.SetHistoName( attrVal );
    }

    else if( attrName == TString( "HistoPath" ) ) {
      statError.SetHistoPath( attrVal );
    }

    else if( attrName == TString( "InputFile" ) ) {
      statError.SetInputFile( attrVal );
    }
    
    else {
      std::cout << "Error: Encountered Element in ActivateStatError with unknown name: " 
		<< attrName << std::endl;
      throw hf_exc();
    }

  } // End: Loop Over Attributes

  // Based on the input, determine
  // if we should use a histogram or not:
  if( statError.GetHistoName() != "" ) {
    statError.SetUseHisto( true );

    // Check that a file has been set
    // (Possibly using the default)
    if( statError.GetInputFile() == "" ) {
      statError.SetInputFile( m_currentInputFile );
    }

  }

  /*
  if( statError.Activate ) {
    if( statError.UseHisto ) {
    }
    else {
      statError.InputFile = "";
      statError.HistoName = "";
      statError.HistoPath = "";
    }
    }
  */

  statError.Print();

  return statError;

}


RooStats::HistFactory::PreprocessFunction ConfigParser::ParseFunctionConfig( TXMLNode* functionNode ){

  std::cout << "Parsing FunctionConfig" << std::endl;

  RooStats::HistFactory::PreprocessFunction func;

  //std::string name, expression, dependents;
  TListIter attribIt = functionNode->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
    if( curAttr->GetName() == TString( "Name" ) ) {
      func.SetName( curAttr->GetValue() );
      //name = curAttr->GetValue() ;
    }
    if( curAttr->GetName() == TString( "Expression" ) ) {
      func.SetExpression( curAttr->GetValue() );
    }
    if( curAttr->GetName() == TString( "Dependents" ) ) {
      func.SetDependents( curAttr->GetValue() );
    }    
  }
  
  std::string command = "expr::"+func.GetName()+"('"+func.GetExpression()+"',{"+func.GetDependents()+"})";
  func.SetCommand( command );
  //  cout << "will pre-process this line " << ret <<endl;
  return func;

}


bool ConfigParser::IsAcceptableNode( TXMLNode* node ) {

  if( node->GetNodeName() == TString( "text" ) ) {    
    return true;
  }

  if( node->GetNodeName() == TString( "comment" ) ) {
    return true;
  }

  return false;

}


bool ConfigParser::CheckTrueFalse( std::string attrVal, std::string NodeTitle ) {

  if( attrVal == "" ) {
    std::cout << "Error: In " << NodeTitle
	      << " Expected either 'True' or 'False' but found empty" << std::endl;
    throw hf_exc();
  }
  else if ( attrVal == "True"  || attrVal == "true"  )   return true;
  else if ( attrVal == "False" || attrVal == "false" )   return false;
  else {
    std::cout << "Error: In " << NodeTitle
	      << " Expected either 'True' or 'False' but found: " << attrVal <<  std::endl;
    throw hf_exc();
  }

  return false;
  
}

//ConfigParser
