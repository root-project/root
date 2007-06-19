// @(#)root/tmva $\Id$
// Author: Andraes Hoecker

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealingFitter                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Fitter using a Simulated Annealing Algorithm
//_______________________________________________________________________

#include "TMVA/SimulatedAnnealingFitter.h"
#include "TMVA/SimulatedAnnealing.h"
#include "TMVA/Timer.h"
#include "TMVA/Interval.h"

ClassImp(TMVA::SimulatedAnnealingFitter)

//_______________________________________________________________________
TMVA::SimulatedAnnealingFitter::SimulatedAnnealingFitter( IFitterTarget& target, 
                                                          const TString& name, 
                                                          const std::vector<Interval*>& ranges, 
                                                          const TString& theOption ) 
   : TMVA::FitterBase( target, name, ranges, theOption )
{
   // constructor

   // default parameters settings for Simulated Annealing algorithm
   DeclareOptions();
   ParseOptions();
}            

//_______________________________________________________________________
void TMVA::SimulatedAnnealingFitter::DeclareOptions() 
{
   // declare SA options

   // MaxCalls                 <int>      maximum number of calls for simulated annealing
   // TemperatureGradient      <float>    temperature gradient for simulated annealing
   // UseAdaptiveTemperature   <bool>     use of adaptive temperature for simulated annealing
   // InitialTemperature       <float>    initial temperature for simulated annealing
   // MinTemperature           <float>    minimum temperature for simulated annealing 
   // Eps                      <int>      number of epochs for simulated annealing
   // NFunLoops                <int>      number of loops for simulated annealing      
   // NEps                     <int>      number of epochs for simulated annealing

   // default settings
   fMaxCalls               = 50000;
   fTemperatureGradient    = 0.7;
   fUseAdaptiveTemperature = kTRUE;
   fInitialTemperature     = 100000;
   fMinTemperature         = 500;
   fEps                    = 1e-04;
   fNFunLoops              = 5;
   fNEps                   = 4; // needs to be at least 2 !
   DeclareOptionRef(fMaxCalls,               "MaxCalls",               "Maximum number of minimisation calls");
   DeclareOptionRef(fTemperatureGradient,    "TemperatureGradient",    "Temperature gradient"); 
   DeclareOptionRef(fUseAdaptiveTemperature, "UseAdaptiveTemperature", "Use adaptive termperature");  
   DeclareOptionRef(fInitialTemperature,     "InitialTemperature",     "Initial temperature");  
   DeclareOptionRef(fMinTemperature,         "MinTemperature",         "Mimimum temperature");
   DeclareOptionRef(fEps,                    "Eps",                    "Epsilon");  
   DeclareOptionRef(fNFunLoops,              "NFunLoops",              "Number of function loops");  
   DeclareOptionRef(fNEps,                   "NEps",                   "Number of epsilons");                   
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealingFitter::SetParameters( Int_t    naxCalls,               
                                                    Int_t    nFunLoops,              
                                                    Int_t    nEps,                   
                                                    Bool_t   useAdaptiveTemperature, 
                                                    Double_t temperatureGradient,    
                                                    Double_t initialTemperature,     
                                                    Double_t minTemperature,         
                                                    Double_t eps )
{
   // set SA configuration parameters
   fMaxCalls               = naxCalls;             
   fNFunLoops              = nFunLoops;            
   fNEps                   = nEps;                 
   fUseAdaptiveTemperature = useAdaptiveTemperature;
   fTemperatureGradient    = temperatureGradient;  
   fInitialTemperature     = initialTemperature;   
   fMinTemperature         = minTemperature;       
   fEps                    = eps;                  
}

//_______________________________________________________________________
Double_t TMVA::SimulatedAnnealingFitter::Run( std::vector<Double_t>& pars )
{
   // Execute fitting
   fLogger << kINFO << "<SimulatedAnnealingFitter> Optimisation, please be patient ... " << Endl;

   SimulatedAnnealing sa( GetFitterTarget(), fRanges );

   // set driving parameters
   sa.SetMaxCalls    ( fMaxCalls );              
   sa.SetTempGrad    ( fTemperatureGradient );   
   sa.SetUseAdaptTemp( fUseAdaptiveTemperature );
   sa.SetInitTemp    ( fInitialTemperature );    
   sa.SetMinTemp     ( fMinTemperature );
   sa.SetNumFunLoops ( fNFunLoops );                   
   sa.SetAccuracy    ( fEps );             
   sa.SetNEps        ( fNEps );                  

   // minimise
   Double_t fcn = sa.Minimize( pars );

   return fcn;
}
