// @(#)root/vmc:$Name:  $:$Id: TMCVerbose.h,v 1.1 2003/02/26 13:31:32 brun Exp $
// Author: Ivana Hrivnacova; 24/02/2003

#ifndef ROOT_TMCVerbose
#define ROOT_TMCVerbose

//
// Class TMCVerbose
// ----------------
// Class for printing detailed info from MC application.
// Defined levels:
//  0  no output
//  1  info up to event level 
//  2  info up to tracking level
//  3  detailed info for each step

#include <TObject.h>

class TVirtualMCStack;

class TMCVerbose : public TObject
{
  public:
    TMCVerbose(Int_t level);
    TMCVerbose();
    virtual ~TMCVerbose();
  
    // methods
    virtual void InitMC();
    virtual void RunMC(Int_t nofEvents);
    virtual void FinishRun();
 
    virtual void ConstructGeometry();
    virtual void InitGeometry();
    virtual void GeneratePrimaries();
    virtual void BeginEvent();
    virtual void BeginPrimary();
    virtual void PreTrack();
    virtual void Stepping();
    virtual void PostTrack();
    virtual void FinishPrimary();
    virtual void FinishEvent();
    
    // set methods
    void  SetLevel(Int_t level);

  private:
    // methods
    void PrintBanner() const;
    void PrintTrackInfo() const;
    void PrintStepHeader() const;
  
    // data members
    Int_t  fLevel;      // verbose level
    Int_t  fStepNumber; // current step number

  ClassDef(TMCVerbose,1)  //Verbose class for MCapplication
};

// inline functions

inline void  TMCVerbose::SetLevel(Int_t level)
{ fLevel = level; }

#endif //ROOT_TMCVerbose

