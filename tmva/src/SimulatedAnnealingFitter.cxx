// @(#)root/tmva $Id$ 
// Author: Andraes Hoecker, Kamil Kraszewski, Maciej Kruk

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
 *      Krzysztof Danielowski <danielow@cern.ch>       - IFJ & AGH, Poland        *
 *      Andreas Hoecker       <Andreas.Hocker@cern.ch> - CERN, Switzerland        *
 *      Kamil Kraszewski      <kalq@cern.ch>           - IFJ & UJ, Poland         *
 *      Maciej Kruk           <mkruk@cern.ch>          - IFJ & AGH, Poland        *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      IFJ-Krakow, Poland                                                        *
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
   fMaxCalls                = 100000;
   fInitialTemperature      = 1e+6;
   fMinTemperature          = 1e-6;
   fEps                     = 1e-10;
   fTemperatureScale        = 1.0;
   fAdaptiveSpeed           = 1.0;
   fTemperatureAdaptiveStep = 0.009875;
   fKernelTemperatureS      = "IncAdaptive";
   fUseDefaultScale         = kFALSE;
   fUseDefaultTemperature   = kFALSE;

   DeclareOptionRef(fMaxCalls,               "MaxCalls",              "Maximum number of minimisation calls");
   DeclareOptionRef(fInitialTemperature,     "InitialTemp",           "Initial temperature");  
   DeclareOptionRef(fMinTemperature,         "MinTemp",               "Mimimum temperature");
   DeclareOptionRef(fEps,                    "Eps",                   "Epsilon");  
   DeclareOptionRef(fTemperatureScale,       "TempScale",             "Temperature scale");
   DeclareOptionRef(fAdaptiveSpeed,          "AdaptiveSpeed",         "Adaptive speed");
   DeclareOptionRef(fTemperatureAdaptiveStep,"TempAdaptiveStep",      "Step made in each generation temperature adaptive");
   DeclareOptionRef(fUseDefaultScale,        "UseDefaultScale",       "Use default temperature scale for temperature minimisation algorithm");
   DeclareOptionRef(fUseDefaultTemperature,  "UseDefaultTemp",        "Use default initial temperature");

   DeclareOptionRef(fKernelTemperatureS,     "KernelTemp",            "Temperature minimisation algorithm");
   AddPreDefVal(TString("IncAdaptive"));
   AddPreDefVal(TString("DecAdaptive"));
   AddPreDefVal(TString("Sqrt"));
   AddPreDefVal(TString("Log"));
   AddPreDefVal(TString("Sin"));
   AddPreDefVal(TString("Homo"));
   AddPreDefVal(TString("Geo"));
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealingFitter::SetParameters( Int_t    maxCalls,
                                                    Double_t initialTemperature,
                                                    Double_t minTemperature,
                                                    Double_t eps,
                                                    TString  kernelTemperatureS,
                                                    Double_t temperatureScale,
                                                    Double_t temperatureAdaptiveStep,
                                                    Bool_t   useDefaultScale,
                                                    Bool_t   useDefaultTemperature)
{
   // set SA configuration parameters
   fMaxCalls                 = maxCalls;
   fInitialTemperature       = initialTemperature;
   fMinTemperature           = minTemperature; 
   fEps                      = eps;
   fKernelTemperatureS       = kernelTemperatureS;
   fTemperatureScale         = temperatureScale;
   fTemperatureAdaptiveStep = temperatureAdaptiveStep;
   fUseDefaultScale          = useDefaultScale;
   fUseDefaultTemperature    = useDefaultTemperature;
}

//_______________________________________________________________________
Double_t TMVA::SimulatedAnnealingFitter::Run( std::vector<Double_t>& pars )
{
   // Execute fitting
   Log() << kINFO << "<SimulatedAnnealingFitter> Optimisation, please be patient ... " << Endl;
   Log() << kINFO << "(progress timing may be inaccurate for SA)" << Endl;

   SimulatedAnnealing sa( GetFitterTarget(), fRanges );

   // set driving parameters
   sa.SetOptions( fMaxCalls, fInitialTemperature, fMinTemperature, fEps, fKernelTemperatureS,
                  fTemperatureScale, fAdaptiveSpeed, fTemperatureAdaptiveStep, 
                  fUseDefaultScale, fUseDefaultTemperature );
   // minimise
   Double_t fcn = sa.Minimize( pars );

   return fcn;
}
