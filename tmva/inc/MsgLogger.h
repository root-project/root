// @(#)root/tmva $Id$
// Author: Attila Krasznahorkay

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MsgLogger                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      TMVA output logger class producing nicely formatted log messages          *
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

#ifndef ROOT_TMVA_MsgLogger
#define ROOT_TMVA_MsgLogger

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MsgLogger                                                            //
//                                                                      //
// ostringstream derivative to redirect and format output               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// STL include(s):
#include <string>
#include <sstream>
#include <iostream>
#include <map>

// ROOT include(s)
#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

// Local include(s):

namespace TMVA {

   class MsgLogger : public std::ostringstream, public TObject {

   public:

      MsgLogger( const TObject* source, EMsgType minType = kINFO );
      MsgLogger( const std::string& source, EMsgType minType = kINFO );
      MsgLogger( EMsgType minType = kINFO );
      MsgLogger( const MsgLogger& parent );
      ~MsgLogger();

      // Accessors
      void        SetSource ( const std::string& source ) { fStrSource = source; }
      EMsgType    GetMinType()                      const { return fMinType; }
      void        SetMinType( EMsgType minType )          { fMinType = minType; }
      std::string GetSource()          const              { return fStrSource; }
      std::string GetPrintedSource()   const;
      std::string GetFormattedSource() const;

      static UInt_t GetMaxSourceSize()                    { return (UInt_t)fgMaxSourceSize; }

      // Needed for copying
      MsgLogger& operator= ( const MsgLogger& parent );

      // Stream modifier(s)
      static MsgLogger& Endmsg( MsgLogger& logger );
      
      // Accept stream modifiers
      MsgLogger& operator<< ( MsgLogger& ( *_f )( MsgLogger& ) );
      MsgLogger& operator<< ( std::ostream& ( *_f )( std::ostream& ) );
      MsgLogger& operator<< ( std::ios& ( *_f )( std::ios& ) );
      
      // Accept message type specification
      MsgLogger& operator<< ( EMsgType type );
      
      // For all the "conventional" inputs
      template <class T> MsgLogger& operator<< ( T arg ) {
         *(std::ostringstream*)this << arg;
         return *this;
      }

      // Temporaly disables all the loggers (Caution! Use with care !)
      static void  InhibitOutput();
      static void  EnableOutput();

   private:

      // private utility routines
      void Send();
      void InitMaps();
      void WriteMsg( EMsgType type, const std::string& line ) const;

      const TObject*           fObjSource;        // the source TObject (used for name)
      std::string              fStrSource;        // alternative string source
      static const std::string fgPrefix;          // the prefix of the source name
      static const std::string fgSuffix;          // suffix following source name
      EMsgType                 fActiveType;       // active type
      static UInt_t            fgMaxSourceSize;   // maximum length of source name
      static Bool_t            fgOutputSupressed; // disable the output globaly (used by generic booster)
      static Bool_t            fgInhibitOutput;   // flag to suppress all output
      static Int_t             fgInstanceCounter; // counts open MsgLogger instances

      static std::map<EMsgType, std::string>* fgTypeMap;   // matches output types with strings
      static std::map<EMsgType, std::string>* fgColorMap;  // matches output types with terminal colors
      EMsgType                                fMinType;    // minimum type for output

      ClassDef(MsgLogger,0) // Ostringstream derivative to redirect and format logging output
   }; // class MsgLogger

   inline MsgLogger& MsgLogger::operator<< ( MsgLogger& (*_f)( MsgLogger& ) )
   {
      return (_f)(*this);
   }

   inline MsgLogger& MsgLogger::operator<< ( std::ostream& (*_f)( std::ostream& ) )
   {
      (_f)(*this);
      return *this;
   }

   inline MsgLogger& MsgLogger::operator<< ( std::ios& ( *_f )( std::ios& ) )
   {
      (_f)(*this);
      return *this;
   }

   inline MsgLogger& MsgLogger::operator<< ( EMsgType type )
   {
      fActiveType = type;
      return *this;
   }

   // Although the proper definition of "Endl" as a function pointer
   // would be nicer C++-wise, it introduces some "unused variable"
   // warnings so let's use the #define definition after all...
   //   static MsgLogger& ( *Endl )( MsgLogger& ) = &MsgLogger::Endmsg;
#define Endl MsgLogger::Endmsg

}

#endif // TMVA_MsgLogger
