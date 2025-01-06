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

#ifndef RooFit_ModelConfig_h
#define RooFit_ModelConfig_h

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooGlobalFunc.h>
#include <RooWorkspaceHandle.h>

#include <TRef.h>

#include <string>

class RooFitResult;

// ModelConfig kept in the RooStats namespace for backwards compatibility.
namespace RooStats {

class ModelConfig final : public TNamed, public RooWorkspaceHandle {

public:
   ModelConfig(RooWorkspace *ws = nullptr)
   {
      if (ws)
         SetWS(*ws);
   }

   ModelConfig(const char *name, RooWorkspace *ws = nullptr) : TNamed(name, name)
   {
      if (ws)
         SetWS(*ws);
   }

   ModelConfig(const char *name, const char *title, RooWorkspace *ws = nullptr) : TNamed(name, title)
   {
      if (ws)
         SetWS(*ws);
   }

   /// clone
   ModelConfig *Clone(const char *name = "") const override
   {
      ModelConfig *mc = new ModelConfig(*this);
      if (strcmp(name, "") == 0) {
         mc->SetName(this->GetName());
      } else {
         mc->SetName(name);
      }
      return mc;
   }

   /// Set a workspace that owns all the necessary components for the analysis.
   void SetWS(RooWorkspace &ws) override;
   //// alias for SetWS(...)
   virtual void SetWorkspace(RooWorkspace &ws) { SetWS(ws); }

   /// Remove the existing reference to a workspace and replace it with this new one.
   void ReplaceWS(RooWorkspace *ws) override
   {
      fRefWS = nullptr;
      SetWS(*ws);
   }

   /// Set the proto DataSet, add to the workspace if not already there
   virtual void SetProtoData(RooAbsData &data)
   {
      ImportDataInWS(data);
      SetProtoData(data.GetName());
   }

   /// Set the Pdf, add to the workspace if not already there
   virtual void SetPdf(const RooAbsPdf &pdf)
   {
      ImportPdfInWS(pdf);
      SetPdf(pdf.GetName());
   }

   /// Set the Prior Pdf, add to the workspace if not already there
   virtual void SetPriorPdf(const RooAbsPdf &pdf)
   {
      ImportPdfInWS(pdf);
      SetPriorPdf(pdf.GetName());
   }

   /// Specify parameters of the PDF.
   virtual void SetParameters(const RooArgSet &set)
   {
      if (!SetHasOnlyParameters(set, "ModelConfig::SetParameters"))
         return;
      fPOIName = std::string(GetName()) + "_POI";
      DefineSetInWS(fPOIName.c_str(), set);
   }

   /// Specify parameters of interest.
   virtual void SetParametersOfInterest(const RooArgSet &set)
   {
      if (!SetHasOnlyParameters(set, "ModelConfig::SetParametersOfInterest"))
         return;
      SetParameters(set);
   }

   /// Specify parameters
   /// using a list of comma-separated list of arguments already in the workspace.
   virtual void SetParameters(const char *argList)
   {
      if (!GetWS())
         return;
      SetParameters(GetWS()->argSet(argList));
   }

   /// Specify parameters of interest
   /// using a comma-separated list of arguments already in the workspace.
   virtual void SetParametersOfInterest(const char *argList) { SetParameters(argList); }

   /// Specify the nuisance parameters (parameters that are not POI).
   virtual void SetNuisanceParameters(const RooArgSet &set)
   {
      if (!SetHasOnlyParameters(set, "ModelConfig::SetNuisanceParameters"))
         return;
      fNuisParamsName = std::string(GetName()) + "_NuisParams";
      DefineSetInWS(fNuisParamsName.c_str(), set);
   }

   /// Specify the nuisance parameters
   /// using a comma-separated list of arguments already in the workspace.
   virtual void SetNuisanceParameters(const char *argList)
   {
      if (!GetWS())
         return;
      SetNuisanceParameters(GetWS()->argSet(argList));
   }

   /// Specify the constraint parameters
   virtual void SetConstraintParameters(const RooArgSet &set)
   {
      if (!SetHasOnlyParameters(set, "ModelConfig::SetConstrainedParameters"))
         return;
      fConstrParamsName = std::string(GetName()) + "_ConstrainedParams";
      DefineSetInWS(fConstrParamsName.c_str(), set);
   }
   /// Specify the constraint parameters
   /// through a comma-separated list of arguments already in the workspace.
   virtual void SetConstraintParameters(const char *argList)
   {
      if (!GetWS())
         return;
      SetConstraintParameters(GetWS()->argSet(argList));
   }

   /// Specify the observables.
   virtual void SetObservables(const RooArgSet &set)
   {
      if (!SetHasOnlyParameters(set, "ModelConfig::SetObservables"))
         return;
      fObservablesName = std::string(GetName()) + "_Observables";
      DefineSetInWS(fObservablesName.c_str(), set);
   }
   /// specify the observables
   /// through a comma-separated list of arguments already in the workspace.
   virtual void SetObservables(const char *argList)
   {
      if (!GetWS())
         return;
      SetObservables(GetWS()->argSet(argList));
   }

   virtual void SetConditionalObservables(const RooArgSet &set);
   /// Specify the conditional observables
   /// through a comma-separated list of arguments already in the workspace.
   virtual void SetConditionalObservables(const char *argList)
   {
      if (!GetWS())
         return;
      SetConditionalObservables(GetWS()->argSet(argList));
   }

   virtual void SetGlobalObservables(const RooArgSet &set);
   /// Specify the global observables
   /// through a comma-separated list of arguments already in the workspace.
   virtual void SetGlobalObservables(const char *argList)
   {
      if (!GetWS())
         return;
      SetGlobalObservables(GetWS()->argSet(argList));
   }

   void SetExternalConstraints(const RooArgSet &set);
   /// Specify the external constraints
   /// through a comma-separated list of arguments already in the workspace.
   virtual void SetExternalConstraints(const char *argList)
   {
      if (!GetWS())
         return;
      SetExternalConstraints(GetWS()->argSet(argList));
   }

   /// Set parameter values for a particular hypothesis if using a common PDF
   /// by saving a snapshot in the workspace.
   virtual void SetSnapshot(const RooArgSet &set);

   /// Specify the name of the PDF in the workspace to be used.
   virtual void SetPdf(const char *name)
   {
      if (!GetWS())
         return;

      if (GetWS()->pdf(name)) {
         fPdfName = name;
      } else {
         std::stringstream ss;
         ss << "pdf " << name << " does not exist in workspace";
         const std::string errorMsg = ss.str();
         coutE(ObjectHandling) << errorMsg << std::endl;
         throw std::runtime_error(errorMsg);
      }
   }

   /// Specify the name of the PDF in the workspace to be used.
   virtual void SetPriorPdf(const char *name)
   {
      if (!GetWS())
         return;

      if (GetWS()->pdf(name)) {
         fPriorPdfName = name;
      } else {
         std::stringstream ss;
         ss << "pdf " << name << " does not exist in workspace";
         const std::string errorMsg = ss.str();
         coutE(ObjectHandling) << errorMsg << std::endl;
         throw std::runtime_error(errorMsg);
      }
   }

   /// Specify the name of the dataset in the workspace to be used.
   virtual void SetProtoData(const char *name)
   {
      if (!GetWS())
         return;

      if (GetWS()->data(name)) {
         fProtoDataName = name;
      } else {
         std::stringstream ss;
         ss << "dataset " << name << " does not exist in workspace";
         const std::string errorMsg = ss.str();
         coutE(ObjectHandling) << errorMsg << std::endl;
         throw std::runtime_error(errorMsg);
      }
   }

   /* getter methods */

   /// get model PDF (return nullptr if pdf has not been specified or does not exist)
   RooAbsPdf *GetPdf() const { return (GetWS()) ? GetWS()->pdf(fPdfName) : nullptr; }

   /// get RooArgSet containing the parameter of interest (return nullptr if not existing)
   const RooArgSet *GetParametersOfInterest() const { return (GetWS()) ? GetWS()->set(fPOIName) : nullptr; }

   /// get RooArgSet containing the nuisance parameters (return nullptr if not existing)
   const RooArgSet *GetNuisanceParameters() const { return (GetWS()) ? GetWS()->set(fNuisParamsName) : nullptr; }

   /// get RooArgSet containing the constraint parameters (return nullptr if not existing)
   const RooArgSet *GetConstraintParameters() const { return (GetWS()) ? GetWS()->set(fConstrParamsName) : nullptr; }

   /// get parameters prior pdf  (return nullptr if not existing)
   RooAbsPdf *GetPriorPdf() const { return (GetWS()) ? GetWS()->pdf(fPriorPdfName) : nullptr; }

   /// get RooArgSet for observables  (return nullptr if not existing)
   const RooArgSet *GetObservables() const { return (GetWS()) ? GetWS()->set(fObservablesName) : nullptr; }

   /// get RooArgSet for conditional observables  (return nullptr if not existing)
   const RooArgSet *GetConditionalObservables() const
   {
      return (GetWS()) ? GetWS()->set(fConditionalObsName) : nullptr;
   }

   /// get RooArgSet for global observables  (return nullptr if not existing)
   const RooArgSet *GetGlobalObservables() const { return (GetWS()) ? GetWS()->set(fGlobalObsName) : nullptr; }

   /// get RooArgSet for global observables  (return nullptr if not existing)
   const RooArgSet *GetExternalConstraints() const { return (GetWS()) ? GetWS()->set(fExtConstraintsName) : nullptr; }

   /// get Proto data set (return nullptr if not existing)
   RooAbsData *GetProtoData() const { return (GetWS()) ? GetWS()->data(fProtoDataName) : nullptr; }

   /// get RooArgSet for parameters for a particular hypothesis  (return nullptr if not existing)
   const RooArgSet *GetSnapshot() const;

   void LoadSnapshot() const;

   RooWorkspace *GetWS() const override;
   /// alias for GetWS()
   RooWorkspace *GetWorkspace() const { return GetWS(); }

   void GuessObsAndNuisance(const RooAbsData &data, bool printModelConfig = true);

   /// overload the print method
   void Print(Option_t *option = "") const override;

   template <typename... CmdArgs_t>
   std::unique_ptr<RooAbsReal> createNLL(RooAbsData &data, CmdArgs_t const &...cmdArgs) const
   {
      return createNLLImpl(data, *RooFit::Detail::createCmdList(&cmdArgs...));
   }

   template <typename... CmdArgs_t>
   std::unique_ptr<RooFitResult> fitTo(RooAbsData &data, CmdArgs_t const &...cmdArgs) const
   {
      return fitToImpl(data, *RooFit::Detail::createCmdList(&cmdArgs...));
   }

protected:
   /// helper function to check that content of a given set is exclusively parameters
   bool SetHasOnlyParameters(const RooArgSet &set, const char *errorMsgPrefix = nullptr);

   /// helper functions to define a set in the WS
   void DefineSetInWS(const char *name, const RooArgSet &set);

   /// internal function to import Pdf in WS
   void ImportPdfInWS(const RooAbsPdf &pdf);

   /// internal function to import data in WS
   void ImportDataInWS(RooAbsData &data);

   TRef fRefWS; ///< WS reference used in the file

   std::string fWSName; ///< name of the WS

   std::string fPdfName;  ///< name of  PDF in workspace
   std::string fDataName; ///< name of data set in workspace
   std::string fPOIName;  ///< name for RooArgSet specifying parameters of interest

   std::string fNuisParamsName;   ///< name for RooArgSet specifying nuisance parameters
   std::string fConstrParamsName; ///< name for RooArgSet specifying constrained parameters
   std::string fPriorPdfName;     ///< name for RooAbsPdf specifying a prior on the parameters

   std::string fConditionalObsName; ///< name for RooArgSet specifying conditional observables
   std::string fGlobalObsName;      ///< name for RooArgSet specifying global observables
   std::string fExtConstraintsName; ///< name for RooArgSet specifying external constraints
   std::string fProtoDataName;      ///< name for RooArgSet specifying dataset that should be used as proto-data

   std::string fSnapshotName; ///< name for RooArgSet that specifies a particular hypothesis

   std::string fObservablesName; ///< name for RooArgSet specifying observable parameters.

private:
   std::unique_ptr<RooAbsReal> createNLLImpl(RooAbsData &data, const RooLinkedList &cmdList) const;
   std::unique_ptr<RooFitResult> fitToImpl(RooAbsData &data, const RooLinkedList &cmdList) const;

   ClassDefOverride(ModelConfig,
                    6); ///< A class that holds configuration information for a model using a workspace as a store
};

} // end namespace RooStats

namespace RooFit {
using ModelConfig = RooStats::ModelConfig;
}

#endif
