// @(#)root/vmc:$Id$
// Author: Ivana Hrivnacova; 24/02/2003

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2003, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMCVerbose
#define ROOT_TMCVerbose

//
// Class TMCVerbose
// ----------------
// Class for printing a detailed infomation from MC application.
// Defined levels:
//  0  no output
//  1  info up to event level
//  2  info up to tracking level
//  3  detailed info for each step

#include <TObject.h>

class TVirtualMCStack;

class TMCVerbose : public TObject {
public:
   TMCVerbose(Int_t level);
   TMCVerbose();
   virtual ~TMCVerbose();

   // methods
   virtual void InitMC();
   virtual void RunMC(Int_t nofEvents);
   virtual void FinishRun();

   virtual void ConstructGeometry();
   virtual void ConstructOpGeometry();
   virtual void InitGeometry();
   virtual void AddParticles();
   virtual void AddIons();
   virtual void GeneratePrimaries();
   virtual void BeginEvent();
   virtual void BeginPrimary();
   virtual void PreTrack();
   virtual void Stepping();
   virtual void PostTrack();
   virtual void FinishPrimary();
   virtual void FinishEvent();

   // set methods
   void SetLevel(Int_t level);

   // get methods
   Int_t GetLevel() const;

private:
   // methods
   void PrintBanner() const;
   void PrintTrackInfo() const;
   void PrintStepHeader() const;

   // data members
   Int_t fLevel;      ///< Verbose level
   Int_t fStepNumber; ///< Current step number

   ClassDef(TMCVerbose, 1) // Verbose class for MC application
};

// inline functions

inline void TMCVerbose::SetLevel(Int_t level)
{
   fLevel = level;
}

inline Int_t TMCVerbose::GetLevel() const
{
   return fLevel;
}

#endif // ROOT_TMCVerbose
