// @(#)root/vmc:$Id$
// Author: Ivana Hrivnacova, 29/04/2014

/*************************************************************************
 * Copyright (C) 2014, Rene Brun and Fons Rademakers.                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************

#ifndef ROOT_TMCtls
#define ROOT_TMCtls

// Thread Local Storage typedefs
//
// According to Geant4 tls.hh and G4Threading.hh

// Always build with thread support but keep a possibility to introduce
// a build option
#define VMC_MULTITHREADED 1

#if ( defined (VMC_MULTITHREADED) )
/*
  #if ( ( defined(__MACH__) && defined(__clang__) && defined(__x86_64__) ) || \
        ( defined(__MACH__) && defined(__GNUC__) && __GNUC__>=4 && __GNUC_MINOR__>=7 ) || \
        defined(__linux__) || defined(_AIX) ) && ( !defined(__CINT__) )
*/
  #if ( defined(__linux__) ) && ( !defined(__CINT__) )
      //  Multi-threaded build: for POSIX systems
      #include <pthread.h>
      #define TMCThreadLocal __thread
  #else
      //#  error "No Thread Local Storage (TLS) technology supported for this platform. Use sequential build !"
      #define TMCThreadLocal
  #endif
#else
  #define TMCThreadLocal
#endif

#endif //ROOT_TMCtls
