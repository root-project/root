// Author: Danilo Piparo CERN, Enrico Guiraud  4/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                    *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CLING__

#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;

#pragma link C++ class ROOT::RDataFrame-;
#pragma link C++ class ROOT::RDF::RInterface<ROOT::Detail::RDF::RFilterBase>-;
#pragma link C++ class ROOT::RDF::RInterface<ROOT::Detail::RDF::RCustomColumnBase>-;
#pragma link C++ namespace ROOT::Internal::RDF;
#pragma link C++ namespace ROOT::Detail::RDF;
#pragma link C++ namespace ROOT::RDF;
#pragma link C++ class ROOT::Detail::RDF::RFilterBase-;
#pragma link C++ class ROOT::Detail::RDF::RJittedFilter-;
#pragma link C++ class ROOT::Detail::RDF::RCustomColumnBase-;
#pragma link C++ class ROOT::Detail::RDF::RJittedCustomColumn-;
#pragma link C++ class ROOT::Internal::RDF::CountHelper-;
#pragma link C++ class ROOT::Detail::RDF::RRangeBase-;
#pragma link C++ class ROOT::Detail::RDF::RLoopManager-;
#pragma link C++ class ROOT::RDF::TH1DModel-;
#pragma link C++ class ROOT::RDF::TH2DModel-;
#pragma link C++ class ROOT::RDF::TH3DModel-;
#pragma link C++ class ROOT::RDF::TProfile1DModel-;
#pragma link C++ class ROOT::RDF::TProfile2DModel-;
#pragma link C++ class ROOT::Internal::RDF::RIgnoreErrorLevelRAII-;
#pragma link C++ class ROOT::Internal::RDF::FillHelper-;
#pragma link C++ class ROOT::RDF::RTrivialDS-;
#pragma link C++ class ROOT::RDF::RRootDS-;
#pragma link C++ class ROOT::RDF::RCsvDS-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<int>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<unsigned int>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<char>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<unsigned char>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<float>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<double>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<Long64_t>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<ULong64_t>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<int>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<unsigned int>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<char>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<unsigned char>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<float>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<double>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<Long64_t>>-;
#pragma link C++ class ROOT::Internal::RDF::TColumnValue<std::vector<ULong64_t>>-;

#endif


