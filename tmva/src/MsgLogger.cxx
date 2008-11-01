// @(#)root/tmva $Id$
// Author: Attila Krasznahorkay

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MsgLogger                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Author:                                                                        *
 *      Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch> - CERN, Switzerland   *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

// Local include(s):
#include "TMVA/MsgLogger.h"
#include "TMVA/Config.h"

// STL include(s):
#include <iomanip>

#include <stdlib.h>

// this is the hardcoded prefix
#define PREFIX "--- "
// this is the hardcoded suffix
#define SUFFIX ": "

// ROOT include(s):

ClassImp(TMVA::MsgLogger)

// this is the hard-coded maximum length of the source names
UInt_t TMVA::MsgLogger::fgMaxSourceSize = 15;


//_______________________________________________________________________
TMVA::MsgLogger::MsgLogger( const TObject* source, EMsgType minType )
   : fObjSource( source ), 
     fStrSource( "" ), 
     fPrefix( PREFIX ), 
     fSuffix( SUFFIX ), 
     fActiveType( kINFO ), 
     fMinType( minType )
{
   // constructor
   InitMaps();
}

//_______________________________________________________________________
TMVA::MsgLogger::MsgLogger( const std::string& source, EMsgType minType )
   : fObjSource( 0 ),
     fStrSource( source ), 
     fPrefix( PREFIX ), 
     fSuffix( SUFFIX ), 
     fActiveType( kINFO ), 
     fMinType( minType )
{
   // constructor
   InitMaps();
}

//_______________________________________________________________________
TMVA::MsgLogger::MsgLogger( EMsgType minType )
   : fObjSource( 0 ), 
     fStrSource( "Unknown" ), 
     fPrefix( PREFIX ), 
     fSuffix( SUFFIX ), 
     fActiveType( kINFO ), 
     fMinType( minType )
{
   // constructor
   InitMaps();
}

//_______________________________________________________________________
TMVA::MsgLogger::MsgLogger( const MsgLogger& parent ) :
   //   basic_ios< MsgLogger::char_type, MsgLogger::traits_type >( new MsgLogger::__stringbuf_type() ),
   std::basic_ios< MsgLogger::char_type, MsgLogger::traits_type >(),
   std::ostringstream(),
   TObject(),
   fPrefix( PREFIX ), 
   fSuffix( SUFFIX )
{
   // copy constructor
   InitMaps();
   *this = parent;
}

//_______________________________________________________________________
TMVA::MsgLogger::~MsgLogger() 
{
   // destructor
}

//_______________________________________________________________________
TMVA::MsgLogger& TMVA::MsgLogger::operator= ( const MsgLogger& parent ) 
{
   // assingment operator
   if (&parent != this) {
      fObjSource  = parent.fObjSource;
      fStrSource  = parent.fStrSource;
      fActiveType = parent.fActiveType;
      fMinType    = parent.fMinType;
   }

   return *this;
}

//_______________________________________________________________________
std::string TMVA::MsgLogger::GetFormattedSource() const
{
   // make sure the source name is no longer than fgMaxSourceSize:
   std::string source_name;
   if (fObjSource) source_name = fObjSource->GetName();
   else            source_name = fStrSource;

   if (source_name.size() > fgMaxSourceSize) {
      source_name = source_name.substr( 0, fgMaxSourceSize - 3 );
      source_name += "...";
   }
   
   return source_name;
}

//_______________________________________________________________________
std::string TMVA::MsgLogger::GetPrintedSource() const
{ 
   // the full logger prefix
   std::string source_name = GetFormattedSource();
   if (source_name.size() < fgMaxSourceSize) 
      for (std::string::size_type i=source_name.size(); i<fgMaxSourceSize; i++) source_name.push_back( ' ' );

   return fPrefix + source_name + fSuffix; 
}

//_______________________________________________________________________
void TMVA::MsgLogger::Send() 
{
   // activates the logger writer

   // make sure the source name is no longer than fgMaxSourceSize:
   std::string source_name = GetFormattedSource();

   std::string message = this->str();
   std::string::size_type previous_pos = 0, current_pos = 0;

   // slice the message into lines:
   while (kTRUE) {
      current_pos = message.find( '\n', previous_pos );
      std::string line = message.substr( previous_pos, current_pos - previous_pos );

      std::ostringstream message_to_send;
      // must call the modifiers like this, otherwise g++ get's confused with the operators...
      message_to_send.setf( std::ios::adjustfield, std::ios::left );
      message_to_send.width( fgMaxSourceSize );
      message_to_send << source_name << fSuffix << line;
      this->WriteMsg( fActiveType, message_to_send.str() );

      if (current_pos == message.npos) break;
      previous_pos = current_pos + 1;
   }

   // reset the stream buffer:
   this->str( "" );
   return;
}

//_______________________________________________________________________
void TMVA::MsgLogger::WriteMsg( EMsgType type, const std::string& line ) const 
{
   // putting the output string, the message type, and the color
   // switcher together into a single string

   if (type < fMinType) return;
   std::map<EMsgType, std::string>::const_iterator stype;
   if ((stype = fTypeMap.find( type )) == fTypeMap.end()) return;
   if (!gConfig().IsSilent()) {
      if (gConfig().UseColor()) {
         // no text for INFO
         if (type == kINFO) std::cout << fPrefix << line << std::endl; // no color for info
         else               std::cout << fColorMap.find( type )->second << fPrefix << "<" 
                                      << stype->second << "> " << line  << "\033[0m" << std::endl;
      } 
      else {
         if (type == kINFO) std::cout << fPrefix << line << std::endl;
         else               std::cout << fPrefix << "<" << stype->second << "> " << line << std::endl;
      }
   }
   // take decision to stop if fatal error
   if (type == kFATAL) { 
      if (!gConfig().IsSilent()) std::cout << "***> abort program execution" << std::endl;
      exit(1);
   }
}

//_______________________________________________________________________
TMVA::MsgLogger& TMVA::MsgLogger::Endmsg( MsgLogger& logger ) 
{
   // end line
   logger.Send();
   return logger;
}

//_______________________________________________________________________
void TMVA::MsgLogger::InitMaps()
{
   // fill maps that assign a string and a color to echo message level
   fTypeMap[kVERBOSE]  = std::string("VERBOSE");
   fTypeMap[kDEBUG]    = std::string("DEBUG");
   fTypeMap[kINFO]     = std::string("INFO");
   fTypeMap[kWARNING]  = std::string("WARNING");
   fTypeMap[kERROR]    = std::string("ERROR");
   fTypeMap[kFATAL]    = std::string("FATAL");
   fTypeMap[kSILENT]   = std::string("SILENT");

   fColorMap[kVERBOSE] = std::string("\033[1;34m");
   fColorMap[kDEBUG]   = std::string("\033[34m");
   fColorMap[kINFO]    = std::string("");
   fColorMap[kWARNING] = std::string("\033[1;31m");
   fColorMap[kERROR]   = std::string("\033[31m");
   fColorMap[kFATAL]   = std::string("\033[37;41;1m");
   fColorMap[kSILENT]  = std::string("\033[30m");
}
