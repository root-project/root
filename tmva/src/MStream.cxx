// @(#)root/tmva $Id: MStream.cxx,v 1.2 2006/08/30 22:19:58 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MStream                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
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

#include "TString.h"
#include "TMVA/MStream.h"

ClassImp(TMVA::MStream)

TMVA::MStream::MStream( OutputLevel ol, const TString prefix )
   : ostream( std::cout.rdbuf() ),
     m_outputLevel( ol ),
     m_prefix( prefix )
{
   // default constructor
   InitMap();
}

TMVA::MStream::MStream( ostream& os, OutputLevel ol, const TString prefix ) 
   : ostream( os.rdbuf() ),
     m_outputLevel( ol ),
     m_prefix( prefix )
{
   // constructor for ostream other than standard output
   InitMap();
}

TMVA::MStream::~MStream() 
{}

void TMVA::MStream::InitMap()
{
   m_outputText[ERROR]   = "<FATAL ERROR> ";
   m_outputText[WARNING] = "<WARNING> ";
   m_outputText[INFO]    = "";
   m_outputText[VERBOSE] = "<verbose> ";
   m_outputText[DEBUG]   = "<DEBUG> ";
}

ostream& TMVA::MStream::operator << (OutputLevel ol) 
{                  
   // if the output level is smaller or equal the requested level, print
   if (ol >= m_outputLevel) *this << m_prefix.Data() << m_outputText[ol];
   else                     *this << "";

   return *this;
}
