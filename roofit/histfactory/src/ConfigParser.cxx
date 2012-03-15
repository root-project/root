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

#include "ConfigParser.h"
#include "Helper.h"

using namespace RooStats;
using namespace HistFactory;

std::string HistFactory::ParseFunctionConfig( TXMLNode* functionNode ){
  std::string name, expression, dependents;
  TListIter attribIt = functionNode->GetAttributes();
  TXMLAttr* curAttr = 0;
  while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
    if( curAttr->GetName() == TString( "Name" ) ) {
      name = curAttr->GetValue() ;
    }
    if( curAttr->GetName() == TString( "Expression" ) ) {
      expression = curAttr->GetValue() ;
    }
    if( curAttr->GetName() == TString( "Dependents" ) ) {
      dependents = curAttr->GetValue() ;
    }    
  }
  std::string ret = "expr::"+name+"('"+expression+"',{"+dependents+"})";
  //  cout << "will pre-process this line " << ret <<endl;
  return ret;
}

void HistFactory::ReadXmlConfig( string filen, vector<EstimateSummary> & summary, Double_t lumi ){

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

  TXMLDocument* xmldoc = xmlparser.GetXMLDocument();
  TXMLNode* rootNode = xmldoc->GetRootNode();

  // not assuming that combination is the only option
  // single channel is also ok
  if( rootNode->GetNodeName() == TString( "Channel" ) ){

    // Walk through the node received and instanciate/configure the object
    TListIter attribIt = rootNode->GetAttributes();
    TXMLAttr* curAttr = 0;
    while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
      if( curAttr->GetName() == TString( "Name" ) ) {
        // name of the channel
        channelName = curAttr->GetValue() ;
        std::cout << " : creating a channel named " << channelName<< std::endl;
      }
      if( curAttr->GetName() == TString( "InputFile" ) ) {
        inputFileName = curAttr->GetValue() ;
      }
      if( curAttr->GetName() == TString( "HistoName" ) ) {
        histoName = curAttr->GetValue() ;
      }
      if( curAttr->GetName() == TString( "HistoPath" ) ) {
        histoPathName = curAttr->GetValue() ;
      }
    }
    
    string inputFileName_cache, histoName_cache, histoPathName_cache;
    TXMLNode* node = rootNode->GetChildren();

    node = rootNode->GetChildren();
    while( node != 0 ) {
      if( node->GetNodeName() == TString( "Data" ) ) {
        inputFileName_cache=inputFileName;
        histoName_cache=histoName;
        histoPathName_cache=histoPathName;
        EstimateSummary data_channel;
        data_channel.channel=channelName;
        data_channel.name="Data";

        attribIt = node->GetAttributes();
        curAttr = 0;
        while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
          if( curAttr->GetName() == TString( "InputFile" ) ) {
            inputFileName = curAttr->GetValue() ;
          }
          if( curAttr->GetName() == TString( "HistoName" ) ) {
            histoName = curAttr->GetValue() ;
          }
          if( curAttr->GetName() == TString( "HistoPath" ) ) {
            histoPathName=curAttr->GetValue();
          }
        }
	cout << "Getting nominal data histogram: " << histoPathName << "/" << histoName << " in file " << inputFileName << endl;
        //data_channel.nominal = (TH1F*) GetHisto(inputFileName, histoPathName, histoName);
	data_channel.nominal = GetHisto(inputFileName, histoPathName, histoName);
        summary.push_back(data_channel);
        inputFileName=inputFileName_cache;
        histoName=histoName_cache;
        histoPathName=histoPathName_cache;
        //data_channel.print();
      }
      node = node->GetNextNode();
    }

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
	/*
	string statErrorName="";
	string statErrorPath="";
	string statErrorFile="";
	*/
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
	  /*
	  if( curAttr->GetName() == TString( "IncludeStatError" ) ) { 
	    if((curAttr->GetValue()==TString("True"))){
	      sample_channel.IncludeStatError = true; // Added McStat
	    }
	  }
	  if( curAttr->GetName() == TString( "ShapeFactorName" ) ) {
	    sample_channel.shapeFactorName = curAttr->GetValue();
	  }
	  */
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

  }
  // above two are the only options
  else {
    std::cout << "Did not find 'Channel' at the root level" 
	      << std::endl;
  }
}


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
	if( curAttr->GetName() == TString( "HistoFileHigh" ) ) {
	  inputFileHigh = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoPathHigh" ) ) {
	  histoPathHigh = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoNameHigh" ) ) {
	  histoNameHigh = curAttr->GetValue() ;
	}
	if( curAttr->GetName() == TString( "HistoFileLow" ) ) {
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

