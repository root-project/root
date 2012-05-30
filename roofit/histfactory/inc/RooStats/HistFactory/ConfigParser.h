// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOSTATS_CONFIGPARSER_h
#define ROOSTATS_CONFIGPARSER_h


#include <cstdlib>
#include <string>
#include <TXMLNode.h>

#include "TList.h"
#include "TFile.h"
#include "TXMLAttr.h"

//#include "RooStats/HistFactory/EstimateSummary.h"


#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Sample.h"


//using namespace std; 

// KC: Should make this a class and have it do some of what is done in MakeModelAndMeasurements

namespace RooStats{
   namespace HistFactory {

     class ConfigParser {

     public:

       // The "main" method
       std::vector< RooStats::HistFactory::Measurement > GetMeasurementsFromXML(std::string input); 


       // Another alternet method
       // void FillMeasurementsAndChannelsFromXML(std::string input, 
       //				       std::vector< RooStats::HistFactory::Measurement >&,
       //				       std::vector< RooStats::HistFactory::Channel >&);
       

       RooStats::HistFactory::Measurement CreateMeasurementFromDriverNode( TXMLNode* node );
       RooStats::HistFactory::Channel ParseChannelXMLFile( std::string filen );


       // Helpers used to process a channel
       HistFactory::Data CreateDataElement( TXMLNode* node );
       HistFactory::Sample CreateSampleElement( TXMLNode* node );
       HistFactory::StatErrorConfig CreateStatErrorConfigElement( TXMLNode* node );

       // Helpers used when processing a Sample
       HistFactory::NormFactor  MakeNormFactor( TXMLNode* node );
       HistFactory::HistoSys    MakeHistoSys( TXMLNode* node );
       HistFactory::HistoFactor MakeHistoFactor( TXMLNode* node );
       HistFactory::OverallSys  MakeOverallSys( TXMLNode* node );
       HistFactory::ShapeFactor MakeShapeFactor( TXMLNode* node );
       HistFactory::ShapeSys    MakeShapeSys( TXMLNode* node );
       HistFactory::StatError   ActivateStatError( TXMLNode* node );
       HistFactory::PreprocessFunction ParseFunctionConfig( TXMLNode* functionNode );

       // To be deprecated
       /*
       typedef std::pair<double,double> UncertPair;
       void AddSystematic( RooStats::HistFactory::EstimateSummary &, TXMLNode*, std::string, std::string,std::string);
       void ReadXmlConfig( std::string, std::vector<RooStats::HistFactory::Channel>& , Double_t );
       */

     protected:


       bool CheckTrueFalse( std::string val, std::string Name );
       bool IsAcceptableNode( TXMLNode* functionNode );

       // To facilitate writing xml, when not
       // specified, files and paths default
       // to these cached values
       std::string m_currentInputFile;
       std::string m_currentChannel;
       // std::string m_currentHistoName;
       std::string m_currentHistoPath;

     };
   }
}

#endif
