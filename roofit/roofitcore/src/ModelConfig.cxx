/*
 * Project: RooFit
 * Authors:
 *   Kyle Cranmer,
 *   Lorenzo Moneta,
 *   Gregory Schott,
 *   Wouter Verkerke,
 *   Sven Kreiss
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooStats::ModelConfig
    \ingroup Roostats

ModelConfig is a simple class that holds configuration information specifying how a model
should be used in the context of various RooStats tools.  A single model can be used
in different ways, and this class should carry all that is needed to specify how it should be used.
ModelConfig requires a workspace to be set.

A ModelConfig holds sets of parameters of the likelihood function that have different interpretations:
- **Parameter of interest** Parameters that are measured (*i.e.* fitted).
- **Nuisance parameters** Parameters that are fitted, but their post-fit value is not interesting. Often,
they might be constrained because external knowledge about them exists, *e.g.* from external measurements.
- **Constraint parameters** No direct use in RooFit/RooStats. Can be used by the user for bookkeeping.
- **Observables** Parameters that have been measured externally, *i.e.* they exist in a dataset. These are not fitted,
but read during fitting from the entries of a dataset.
- **Conditional observables** Observables that are not integrated when the normalisation of the PDF is calculated.
See *e.g.* `rf306_condpereventerrors` in the RooFit tutorials.
- **Global observables** Observables that to the fit look like "constant" values, *i.e.* they are not being
fitted and they are not loaded from a dataset, but some knowledge exists that allows to set them to a
specific value. Examples:
-- A signal efficiency measured in a Monte Carlo study.
-- When constraining a parameter \f$ b \f$, the target value (\f$ b_0 \f$) that this parameter is constrained to:
\f[
  \mathrm{Constraint}_b = \mathrm{Gauss}(b_0 \, | \, b, 0.2)
\f]
- **External constraints** Include given external constraints to likelihood by multiplying them with the original
likelihood.
*/

#include <RooFit/ModelConfig.h>

#include <RooFitResult.h>
#include <RooMsgService.h>
#include <RooRealVar.h>

#include <sstream>


namespace {

void removeConstantParameters(RooAbsCollection &coll)
{
   RooArgSet constSet;
   for (auto const *myarg : static_range_cast<RooRealVar *>(coll)) {
      if (myarg->isConstant())
         constSet.add(*myarg);
   }
   coll.remove(constSet);
}

} // namespace

using std::ostream;

namespace RooStats {

////////////////////////////////////////////////////////////////////////////////
/// Makes sensible guesses of observables, parameters of interest
/// and nuisance parameters if one or multiple have been set by the creator of this ModelConfig.
///
/// Defaults:
/// - Observables: determined from data,
/// - Global observables: explicit obs  -  obs from data  -  constant observables
/// - Parameters of interest: empty,
/// - Nuisance parameters: all parameters except parameters of interest
///
/// We use nullptr to mean not set, so we don't want to fill
/// with empty RooArgSets.

void ModelConfig::GuessObsAndNuisance(const RooArgSet &obsSet, bool printModelConfig)
{

   // observables
   if (!GetObservables()) {
      SetObservables(*std::unique_ptr<RooArgSet>{GetPdf()->getObservables(obsSet)});
   }
   // global observables
   if (!GetGlobalObservables()) {
      RooArgSet co(*GetObservables());
      co.remove(*std::unique_ptr<RooArgSet>{GetPdf()->getObservables(obsSet)});
      removeConstantParameters(co);
      if (!co.empty())
         SetGlobalObservables(co);

      // TODO BUG This does not work as observables with the same name are already in the workspace.
      /*
      RooArgSet o(*GetObservables());
      o.remove(co);
      SetObservables(o);
      */
   }

   // parameters
   //   if (!GetParametersOfInterest()) {
   //      SetParametersOfInterest(RooArgSet());
   //   }
   if (!GetNuisanceParameters()) {
      RooArgSet params;
      GetPdf()->getParameters(&obsSet, params);
      RooArgSet p(params);
      p.remove(*GetParametersOfInterest());
      removeConstantParameters(p);
      if (!p.empty())
         SetNuisanceParameters(p);
   }

   // print Modelconfig as an info message

   if (printModelConfig) {
      std::ostream &oldstream = RooPrintable::defaultPrintStream(&ccoutI(InputArguments));
      Print();
      RooPrintable::defaultPrintStream(&oldstream);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// print contents of Model on the default print stream
/// It can be changed using RooPrintable

void ModelConfig::Print(Option_t *) const
{
   ostream &os = RooPrintable::defaultPrintStream();

   os << std::endl << "=== Using the following for " << GetName() << " ===" << std::endl;

   // args
   if (GetObservables()) {
      os << "Observables:             ";
      GetObservables()->Print("");
   }
   if (GetParametersOfInterest()) {
      os << "Parameters of Interest:  ";
      GetParametersOfInterest()->Print("");
   }
   if (GetNuisanceParameters()) {
      os << "Nuisance Parameters:     ";
      GetNuisanceParameters()->Print("");
   }
   if (GetGlobalObservables()) {
      os << "Global Observables:      ";
      GetGlobalObservables()->Print("");
   }
   if (GetConstraintParameters()) {
      os << "Constraint Parameters:   ";
      GetConstraintParameters()->Print("");
   }
   if (GetConditionalObservables()) {
      os << "Conditional Observables: ";
      GetConditionalObservables()->Print("");
   }
   if (GetProtoData()) {
      os << "Proto Data:              ";
      GetProtoData()->Print("");
   }

   // pdfs
   if (GetPdf()) {
      os << "PDF:                     ";
      GetPdf()->Print("");
   }
   if (GetPriorPdf()) {
      os << "Prior PDF:               ";
      GetPriorPdf()->Print("");
   }

   // snapshot
   const RooArgSet *snapshot = GetSnapshot();
   if (snapshot) {
      os << "Snapshot:                " << std::endl;
      snapshot->Print("v");
      delete snapshot;
   }

   os << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// If a workspace already exists in this ModelConfig, RooWorkspace::merge(ws) will be called
/// on the existing workspace.

void ModelConfig::SetWS(RooWorkspace &ws)
{
   if (!fRefWS) {
      fRefWS = &ws;
   } else {
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
      GetWS()->merge(ws);
      RooMsgService::instance().setGlobalKillBelow(level);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get workspace.

RooWorkspace *ModelConfig::GetWS() const
{
   if (!fRefWS) {
      coutE(ObjectHandling) << "workspace not set" << std::endl;
      return nullptr;
   }
   return fRefWS;
}

////////////////////////////////////////////////////////////////////////////////
/// save snapshot in the workspace
/// and use values passed with the set

void ModelConfig::SetSnapshot(const RooArgSet &set)
{
   if (!GetWS())
      return;

   fSnapshotName = GetName();
   if (!fSnapshotName.empty())
      fSnapshotName += "_";
   fSnapshotName += set.GetName();
   if (!fSnapshotName.empty())
      fSnapshotName += "_";
   fSnapshotName += "snapshot";
   GetWS()->saveSnapshot(fSnapshotName, set, true); // import also the given parameter values
   DefineSetInWS(fSnapshotName.c_str(), set);
}

////////////////////////////////////////////////////////////////////////////////
/// Load the snapshot from ws and return the corresponding set with the snapshot values.
/// User must delete returned RooArgSet.

const RooArgSet *ModelConfig::GetSnapshot() const
{
   if (!GetWS())
      return nullptr;
   if (!fSnapshotName.length())
      return nullptr;
   // calling loadSnapshot will also copy the current parameter values in the workspaces
   // since we do not want to change the model parameters - we restore the previous ones
   if (!GetWS()->set(fSnapshotName))
      return nullptr;
   RooArgSet snapshotVars(*GetWS()->set(fSnapshotName));
   if (snapshotVars.empty())
      return nullptr;
   // make my snapshot which will contain a copy of the snapshot variables
   RooArgSet tempSnapshot;
   snapshotVars.snapshot(tempSnapshot);
   // load snapshot value from the workspace
   if (!(GetWS()->loadSnapshot(fSnapshotName.c_str())))
      return nullptr;
   // by doing this snapshotVars will have the snapshot values - make the snapshot to return
   const RooArgSet *modelSnapshot = dynamic_cast<const RooArgSet *>(snapshotVars.snapshot());
   // restore now the variables of snapshot in ws to their original values
   // need to const cast since assign is not const (but in reality in just assign values and does not change the set)
   // and anyway the set is const
   snapshotVars.assignFast(tempSnapshot);
   return modelSnapshot;
}

////////////////////////////////////////////////////////////////////////////////
/// load the snapshot from ws if it exists

void ModelConfig::LoadSnapshot() const
{
   if (!GetWS())
      return;
   GetWS()->loadSnapshot(fSnapshotName.c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// helper functions to avoid code duplication

void ModelConfig::DefineSetInWS(const char *name, const RooArgSet &set)
{
   if (!GetWS())
      return;

   const RooArgSet *prevSet = GetWS()->set(name);
   if (prevSet) {
      // be careful not to remove passed set in case it is the same updated
      if (prevSet != &set)
         GetWS()->removeSet(name);
   }

   // suppress warning when we re-define a previously defined set (when set == prevSet )
   // and set is not removed in that case
   RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   GetWS()->defineSet(name, set, true);

   RooMsgService::instance().setGlobalKillBelow(level);
}

////////////////////////////////////////////////////////////////////////////////
/// internal function to import Pdf in WS

void ModelConfig::ImportPdfInWS(const RooAbsPdf &pdf)
{
   if (!GetWS())
      return;

   if (!GetWS()->pdf(pdf.GetName())) {
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
      GetWS()->import(pdf, RooFit::RecycleConflictNodes());
      RooMsgService::instance().setGlobalKillBelow(level);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// internal function to import data in WS

void ModelConfig::ImportDataInWS(RooAbsData &data)
{
   if (!GetWS())
      return;

   if (!GetWS()->data(data.GetName())) {
      RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
      GetWS()->import(data);
      RooMsgService::instance().setGlobalKillBelow(level);
   }
}

////////////////////////////////////////////////////////////////////////////////

bool ModelConfig::SetHasOnlyParameters(const RooArgSet &set, const char *errorMsgPrefix)
{

   RooArgSet nonparams;
   for (auto const *arg : set) {
      if (!arg->isFundamental()) {
         nonparams.add(*arg);
      }
   }

   if (errorMsgPrefix && !nonparams.empty()) {
      std::cout << errorMsgPrefix << " ERROR: specified set contains non-parameters: " << nonparams << std::endl;
   }
   return (nonparams.empty());
}

/// Specify the external constraints.
void ModelConfig::SetExternalConstraints(const RooArgSet &set)
{
   fExtConstraintsName = std::string(GetName()) + "_ExternalConstraints";
   DefineSetInWS(fExtConstraintsName.c_str(), set);
}

/// Specify the conditional observables.
void ModelConfig::SetConditionalObservables(const RooArgSet &set)
{
   if (!SetHasOnlyParameters(set, "ModelConfig::SetConditionalObservables"))
      return;
   fConditionalObsName = std::string(GetName()) + "_ConditionalObservables";
   DefineSetInWS(fConditionalObsName.c_str(), set);
}

/// Specify the global observables.
void ModelConfig::SetGlobalObservables(const RooArgSet &set)
{

   if (!SetHasOnlyParameters(set, "ModelConfig::SetGlobalObservables"))
      return;

   // make global observables constant
   for (auto *arg : set) {
      arg->setAttribute("Constant", true);
   }

   fGlobalObsName = std::string(GetName()) + "_GlobalObservables";
   DefineSetInWS(fGlobalObsName.c_str(), set);
}

namespace {

std::unique_ptr<RooLinkedList>
finalizeCmdList(ModelConfig const &modelConfig, RooLinkedList const &cmdList, std::vector<RooCmdArg> &cmdArgs)
{
   auto addCmdArg = [&](RooCmdArg const &cmdArg) {
      if (cmdList.FindObject(cmdArg.GetName())) {
         std::stringstream ss;
         ss << "Illegal command argument \"" << cmdArg.GetName()
            << "\" passed to ModelConfig::createNLL(). This option is retrieved from the ModelConfig itself.";
         const std::string errorMsg = ss.str();
         oocoutE(&modelConfig, InputArguments) << errorMsg << std::endl;
         throw std::runtime_error(errorMsg);
      }
      cmdArgs.push_back(cmdArg);
   };

   if (auto args = modelConfig.GetConditionalObservables()) {
      addCmdArg(RooFit::ConditionalObservables(*args));
   }

   if (auto args = modelConfig.GetGlobalObservables()) {
      addCmdArg(RooFit::GlobalObservables(*args));
   }

   if (auto args = modelConfig.GetExternalConstraints()) {
      addCmdArg(RooFit::ExternalConstraints(*args));
   }

   auto finalCmdList = std::make_unique<RooLinkedList>(cmdList);
   for (RooCmdArg &arg : cmdArgs) {
      finalCmdList->Add(&arg);
   }

   return finalCmdList;
}

} // namespace

/** @fn RooStats::ModelConfig::createNLL()
 *
 * Wrapper around RooAbsPdf::createNLL(), where
 * the pdf and some configuration options are retrieved from the ModelConfig.
 *
 * The options taken from the ModelConfig are:
 *
 *   * ConditionalObservables()
 *   * GlobalObservables()
 *   * ExternalConstraints()
 *
 * Except for the options above, you can still pass all the other command
 * arguments supported by RooAbsPdf::createNLL().
 */

std::unique_ptr<RooAbsReal> ModelConfig::createNLLImpl(RooAbsData &data, const RooLinkedList &cmdList) const
{
   std::vector<RooCmdArg> cmdArgs;
   auto finalCmdList = finalizeCmdList(*this, cmdList, cmdArgs);
   return std::unique_ptr<RooAbsReal>{GetPdf()->createNLL(data, *finalCmdList)};
}

/** @fn RooStats::ModelConfig::fitTo()
 *
 * Wrapper around RooAbsPdf::fitTo(), where
 * the pdf and some configuration options are retrieved from the ModelConfig.
 *
 * See ModelConfig::createNLL() for more information.
 */
std::unique_ptr<RooFitResult> ModelConfig::fitToImpl(RooAbsData &data, const RooLinkedList &cmdList) const
{
   std::vector<RooCmdArg> cmdArgs;
   auto finalCmdList = finalizeCmdList(*this, cmdList, cmdArgs);
   return std::unique_ptr<RooFitResult>{GetPdf()->fitTo(data, *finalCmdList)};
}

} // end namespace RooStats
