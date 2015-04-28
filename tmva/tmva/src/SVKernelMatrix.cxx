// @(#)root/tmva $Id$
// Author: Andrzej Zemla

/**********************************************************************************
* Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
* Package: TMVA                                                                  *
* Class  : SVKernelMatrix                                                        *
* Web    : http://tmva.sourceforge.net                                           *
*                                                                                *
* Description:                                                                   *
*      Implementation                                                            *
*                                                                                *
* Authors (alphabetical):                                                        *
*      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
*      Andrzej Zemla  <azemla@cern.ch>        - IFJ PAN, Krakow, Poland          *
*      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *
*                                                                                *
* Copyright (c) 2005:                                                            *
*      CERN, Switzerland                                                         *
*      MPI-K Heidelberg, Germany                                                 *
*      PAN, Krakow, Poland                                                       *
*                                                                                *
* Redistribution and use in source and binary forms, with or without             *
* modification, are permitted according to the terms listed in LICENSE           *
* (http://tmva.sourceforge.net/LICENSE)                                          *
**********************************************************************************/

#include "TMVA/SVKernelMatrix.h"
#include "TMVA/SVKernelFunction.h"
#include "TMVA/SVEvent.h"
#include <iostream>
#include <stdexcept>
#include "TMVA/MsgLogger.h"

//_______________________________________________________________________
TMVA::SVKernelMatrix::SVKernelMatrix()
   : fSize(0),
     fKernelFunction(0),
     fSVKernelMatrix(0),
     fLogger( new MsgLogger("ResultsRegression", kINFO) )
{
   // constructor
}

//_______________________________________________________________________
TMVA::SVKernelMatrix::SVKernelMatrix( std::vector<TMVA::SVEvent*>* inputVectors, SVKernelFunction* kernelFunction )
   : fSize(inputVectors->size()),
     fKernelFunction(kernelFunction),
     fLogger( new MsgLogger("SVKernelMatrix", kINFO) )
{
   // constructor
   fSVKernelMatrix = new Float_t*[fSize];
   try{
      for (UInt_t i = 0; i < fSize; i++) fSVKernelMatrix[i] = new Float_t[i+1];
   }catch(...){
      Log() << kFATAL << "Input data too large. Not enough memory to allocate memory for Support Vector Kernel Matrix. Please reduce the number of input events or use a different method."<<Endl;
   }
   for (UInt_t i = 0; i < fSize; i++) {
      fSVKernelMatrix[i][i] = 2*fKernelFunction->Evaluate((*inputVectors)[i], (*inputVectors)[i]);
      for (UInt_t j = 0; j <=i; j++) {
         fSVKernelMatrix[i][j] = fKernelFunction->Evaluate((*inputVectors)[i], (*inputVectors)[j]);
      }
   }
}

//_______________________________________________________________________
TMVA::SVKernelMatrix::~SVKernelMatrix()
{
   // destructor
   for (UInt_t i = fSize -1; i > 0; i--) {
      delete[] fSVKernelMatrix[i];
      fSVKernelMatrix[i] = 0;
   }
   delete[] fSVKernelMatrix;
   fSVKernelMatrix = 0;
}

//_______________________________________________________________________
Float_t* TMVA::SVKernelMatrix::GetLine( UInt_t line )
{
   // returns a row of the kernel matrix

   Float_t* fLine = NULL;
   if (line >= fSize) {
      return NULL;
   }
   else {
      fLine = new Float_t[fSize];
      for( UInt_t i = 0; i <line; i++)
         fLine[i] = fSVKernelMatrix[line][i];
      for( UInt_t i = line; i < fSize; i++)
         fLine[i] = fSVKernelMatrix[i][line];
      return fLine;
   }
}

//_______________________________________________________________________
Float_t TMVA::SVKernelMatrix::GetElement(UInt_t i, UInt_t j)
{ 
   // returns an element of the kernel matrix

   if (i > j) return fSVKernelMatrix[i][j]; 
   else       return fSVKernelMatrix[j][i]; // it's symmetric, ;)
}
