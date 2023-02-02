/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFitHS3_HistFactoryJSONTool_h
#define RooFitHS3_HistFactoryJSONTool_h

#include <iostream>
#include <string>

namespace RooStats {
namespace HistFactory {

class Measurement;

class JSONTool {
public:
   JSONTool(RooStats::HistFactory::Measurement &m) : _measurement(m) {}

   void PrintJSON(std::ostream &os = std::cout);
   void PrintJSON(std::string const &filename);
   void PrintYAML(std::ostream &os = std::cout);
   void PrintYAML(std::string const &filename);

private:
   RooStats::HistFactory::Measurement &_measurement;
};

} // namespace HistFactory
} // namespace RooStats

#endif
