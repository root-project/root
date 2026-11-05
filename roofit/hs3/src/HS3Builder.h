/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_HS3Builder_h
#define RooFit_HS3Builder_h

#include "HS3Document.h"

#include <RooArgSet.h>

#include <memory>
#include <string>
#include <vector>

class RooDataHist;
class RooJSONFactoryWSTool;
class RooRealVar;

namespace RooFit {
namespace JSONIO {
class Importer;
}

namespace HS3 {

/// Forward the warnings to the message service, and throw if there are errors.
void report(Diagnostics const &diagnostics);

/// Create the variables for the axes of binned data.
RooArgSet createObservables(std::vector<BinnedAxis> const &axes);

/// The axis of binned data for a variable.
BinnedAxis toBinnedAxis(RooRealVar const &var);

/// Create a RooDataHist with the contents and errors. If there are no errors, it doesn't store squared weights.
std::unique_ptr<RooDataHist> createDataHist(std::string const &name, RooArgSet const &vars,
                                            std::vector<double> const &contents, std::vector<double> const *errors);

/// Create the RooFit objects of a HistFactory distribution in the workspace of the tool.
void buildHistFactoryDist(RooJSONFactoryWSTool &tool, std::string const &name, HistFactoryDist const &dist);

/// Called by the exporter of a HistFactory distribution, which is then written by the tool.
void setExportedObject(RooJSONFactoryWSTool &tool, HistFactoryDist dist);

/// Whether this is the importer that RooFit registers for HistFactory distributions, as opposed to the one of a user.
bool isBuiltinHistFactoryImporter(JSONIO::Importer const &importer);

} // namespace HS3
} // namespace RooFit

#endif
