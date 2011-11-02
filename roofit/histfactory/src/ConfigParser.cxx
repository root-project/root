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
        data_channel.nominal = (TH1F*) GetHisto(inputFileName, histoPathName, histoName);
        summary.push_back(data_channel);
        inputFileName=inputFileName_cache;
        histoName=histoName_cache;
        histoPathName=histoPathName_cache;
        //data_channel.print();
      }
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
        }
        sample_channel.nominal = (TH1F*) GetHisto(inputFileName, histoPathName, histoName);
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
      sample_channel.AddSyst(Name, 
                    (TH1F*) GetHisto( inputFileLow, histoPathLow, histoNameLow),
                    (TH1F*) GetHisto( inputFileHigh, histoPathHigh, histoNameHigh));
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
    node=node->GetNextNode();
  }
}

