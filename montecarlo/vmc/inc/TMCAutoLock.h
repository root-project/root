// @(#)root/vmc:$Id$
// Author: Ivana Hrivnacova, 24/03/2017

/*************************************************************************
 * Copyright (C) 2014, Rene Brun and Fons Rademakers.                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMCAUTOLOCK_HH
#define TMCAUTOLOCK_HH

//------------------------------------------------
// The Geant4 Virtual Monte Carlo package
// Copyright (C) 2013, 2014 Ivana Hrivnacova
// All rights reserved.
//
// For the licensing terms see geant4_vmc/LICENSE.
// Contact: root-vmc@cern.ch
//-------------------------------------------------

/// \file TMCAutoLock.h
/// \brief Definition of the TMCTemplateAutoLock and TMCImpMutexAutoLock classes
///
/// \author I. Hrivnacova; IPN Orsay

//
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
//
// $Id$
//
// ---------------------------------------------------------------
// GEANT 4 class header file
//
// Class Description:
//
// This class provides a mechanism to create a mutex and locks/unlocks it.
// Can be used by applications to implement in a portable way a mutexing logic.
// Usage Example:
//
//      #include "G4Threading.hh"
//      #include "G4AutoLock.hh"
//      /* somehwere */
//      G4Mutex aMutex = G4MUTEX_INITIALIZER;
//      /*
//       somewhere else:
//       The G4AutoLock instance will automatically unlock the mutex when it
//       goes out of scope, lock and unlock method are anyway available for
//       explicit handling of mutex lock. */
//      G4AutoLock l(&aMutex);
//      ProtectedCode();
//      l.unlock(); //explicit unlock
//      UnprotectedCode();
//      l.lock();   //explicit lock
//
// Note that G4AutoLock is defined also for a sequential Geant4 build,
// but has no effect.

// ---------------------------------------------------------------
// Author: Andrea Dotti (15 Feb 2013): First Implementation
// ---------------------------------------------------------------

#ifndef _WIN32
#define TMCMULTITHREADED 1
#endif

#if defined(TMCMULTITHREADED)

#include <pthread.h>
typedef pthread_mutex_t TMCMutex;
#define TMCMUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define TMCMUTEXLOCK pthread_mutex_lock
#define TMCMUTEXUNLOCK pthread_mutex_unlock
typedef int (*TMCthread_lock)(TMCMutex *);
typedef int (*TMCthread_unlock)(TMCMutex *);
#else
typedef int TMCMutex;
int fake_mutex_lock_unlock(TMCMutex *);
#define TMCMUTEX_INITIALIZER 1
#define TMCMUTEXLOCK fake_mutex_lock_unlock
#define TMCMUTEXUNLOCK fake_mutex_lock_unlock
typedef int (*TMCthread_lock)(TMCMutex *);
typedef int (*TMCthread_unlock)(TMCMutex *);
#endif

/// \brief Template classe which provides a mechanism to create a mutex and
/// locks/unlocks it.
///
/// Extracted from G4AutoLock implementation for Linux
/// Note: Note that G4TemplateAutoLock by itself is not thread-safe and
///       cannot be shared among threads due to the locked switch

template <class M, typename L, typename U>
class TMCTemplateAutoLock {
public:
   TMCTemplateAutoLock(M *mtx, L l, U u) : locked(false), _m(mtx), _l(l), _u(u) { lock(); }

   virtual ~TMCTemplateAutoLock() { unlock(); }

   inline void unlock()
   {
      if (!locked) return;
      _u(_m);
      locked = false;
   }

   inline void lock()
   {
      if (locked) return;
      _l(_m);
      locked = true;
   }

private:
   // Disable copy and assignement operators
   //
   TMCTemplateAutoLock(const TMCTemplateAutoLock &rhs);
   TMCTemplateAutoLock &operator=(const TMCTemplateAutoLock &rhs);

private:
   bool locked;
   M *_m;
   L _l;
   U _u;
};

/// \brief Realization of TMCTemplateAutoLock with TMCMutex
///
/// Extracted from G4AutoLock implementation for Linux

struct TMCImpMutexAutoLock : public TMCTemplateAutoLock<TMCMutex, TMCthread_lock, TMCthread_unlock> {
   TMCImpMutexAutoLock(TMCMutex *mtx)
      : TMCTemplateAutoLock<TMCMutex, TMCthread_lock, TMCthread_unlock>(mtx, &TMCMUTEXLOCK, &TMCMUTEXUNLOCK)
   {
   }
};
typedef TMCImpMutexAutoLock TMCAutoLock;

#endif // TMCAUTOLOCK_HH
