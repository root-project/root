// @(#)root/tmva $Id: MsgLogger.h,v 1.6 2006/10/17 21:22:29 andreas.hoecker Exp $
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
 * Authors (alphabetical):                                                        *
 *      Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch> - CERN, Switzerland   *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
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
// ostreamstream derivative to redirect and format output               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// STL include(s):
#include <string>
#include <sstream>
#include <map>

// ROOT include(s)
#include "TObject.h"
#include "TString.h"

// Local include(s):

namespace TMVA {

   // define outside of class to facilite access
   enum MsgType { 
      kVERBOSE = 1, 
      kDEBUG   = 2,
      kINFO    = 3,
      kWARNING = 4,
      kERROR   = 5,
      kFATAL   = 6,
      kALWAYS  = 7
   };

   class MsgLogger : public std::ostringstream, public TObject {

   public:

      MsgLogger( const TObject* source, MsgType minType = kINFO );
      MsgLogger( const std::string& source, MsgType minType = kINFO );
      MsgLogger( MsgType minType = kINFO );
      MsgLogger( const MsgLogger& parent );
      ~MsgLogger();

      // Accessors
      void        SetSource ( const std::string& source ) { fStrSource = source; }
      MsgType     GetMinType()                      const { return fMinType; }
      void        SetMinType( MsgType minType )           { fMinType = minType; }
      UInt_t      GetMaxSourceSize() const                { return (UInt_t)fMaxSourceSize; }
      std::string GetPrintedSource() const;
      std::string GetFormattedSource() const;
      
      // Needed for copying
      MsgLogger& operator= ( const MsgLogger& parent );

      // Stream modifier(s)
      static MsgLogger& endmsg( MsgLogger& logger );
      
      // Accept stream modifiers
      MsgLogger& operator<< ( MsgLogger& ( *_f )( MsgLogger& ) );
      MsgLogger& operator<< ( std::ostream& ( *_f )( std::ostream& ) );
      MsgLogger& operator<< ( std::ios& ( *_f )( std::ios& ) );
      
      // Accept message type specification
      MsgLogger& operator<< ( MsgType type );
      
      // For all the "conventional" inputs
      template <class T> MsgLogger& operator<< ( T arg ) {
         *(std::ostringstream*)this << arg;
         return *this;
      }

   private:

      // private utility routines
      void Send();
      void InitMaps();
      void WriteMsg( MsgType type, const std::string& line ) const;

      const TObject*                 fObjSource;     // the source TObject (used for name)
      std::string                    fStrSource;     // alternative string source
      const std::string              fPrefix;        // the prefix of the source name
      const std::string              fSuffix;        // suffix following source name
      MsgType                        fActiveType;    // active type
      const std::string::size_type   fMaxSourceSize; // maximum length of source name

      std::map<MsgType, std::string> fTypeMap;       // matches output types with strings
      std::map<MsgType, std::string> fColorMap;      // matches output types with terminal colors
      MsgType                        fMinType;       // minimum type for output

      ClassDef(MsgLogger,0) // ostringstream derivative to redirect and format logging output  
         ;
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

   inline MsgLogger& MsgLogger::operator<< ( MsgType type ) 
   {
      fActiveType = type;
      return *this;
   }

   // Although the proper definition of "Endl" as a function pointer
   // would be nicer C++-wise, it introduces some "unused variable"
   // warnings so let's use the #define definition after all...
   //   static MsgLogger& ( *Endl )( MsgLogger& ) = &MsgLogger::endmsg;
#define Endl MsgLogger::endmsg

}

#endif // TMVA_MsgLogger
