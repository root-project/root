// @(#)root/roostats:$Id$
// Author: George Lewis, Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef HISTFACTORY_MEASUREMENT_H
#define HISTFACTORY_MEASUREMENT_H

#include <TNamed.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class TFile;
class TH1;

class RooWorkspace;

namespace RooStats::HistFactory {

namespace Constraint {

enum Type {
   Gaussian,
   Poisson
};
std::string Name(Type type);
Type GetType(const std::string &Name);

} // namespace Constraint

/** \class OverallSys
 * \ingroup HistFactory
 * Configuration for a constrained overall systematic to scale sample normalisations.
 */
class OverallSys {

public:
   void SetName(const std::string &Name) { fName = Name; }
   const std::string &GetName() const { return fName; }

   void SetLow(double Low) { fLow = Low; }
   void SetHigh(double High) { fHigh = High; }
   double GetLow() const { return fLow; }
   double GetHigh() const { return fHigh; }

   void Print(std::ostream & = std::cout) const;
   void PrintXML(std::ostream &) const;

protected:
   std::string fName;
   double fLow = 0.0;
   double fHigh = 0.0;
};

/** \class NormFactor
 * \ingroup HistFactory
 * Configuration for an \a un- constrained overall systematic to scale sample normalisations.
 */
class NormFactor {

public:
   void SetName(const std::string &Name) { fName = Name; }
   std::string GetName() const { return fName; }

   void SetVal(double Val) { fVal = Val; }
   double GetVal() const { return fVal; }

   void SetLow(double Low) { fLow = Low; }
   void SetHigh(double High) { fHigh = High; }
   double GetLow() const { return fLow; }
   double GetHigh() const { return fHigh; }

   void Print(std::ostream & = std::cout) const;
   void PrintXML(std::ostream &) const;

protected:
   std::string fName;
   double fVal = 1.0;
   double fLow = 1.0;
   double fHigh = 1.0;
};

/** ////////////////////////////////////////////////////////////////////////////////////////////
 * \class HistogramUncertaintyBase
 * \ingroup HistFactory
 * Base class to store the up and down variations for histogram uncertainties.
 * Use the derived classes for actual models.
 */
class HistogramUncertaintyBase {

public:
   HistogramUncertaintyBase();
   HistogramUncertaintyBase(const std::string &Name);
   HistogramUncertaintyBase(const HistogramUncertaintyBase &oth);
   HistogramUncertaintyBase(HistogramUncertaintyBase &&);

   virtual ~HistogramUncertaintyBase();

   // Need deep copies because the class owns its histograms.
   HistogramUncertaintyBase &operator=(const HistogramUncertaintyBase &oth);
   HistogramUncertaintyBase &operator=(HistogramUncertaintyBase &&);

   virtual void Print(std::ostream & = std::cout) const;
   virtual void PrintXML(std::ostream &) const = 0;
   virtual void writeToFile(const std::string &FileName, const std::string &DirName);

   void SetHistoLow(TH1 *Low);
   void SetHistoHigh(TH1 *High);

   const TH1 *GetHistoLow() const { return fhLow.get(); }
   const TH1 *GetHistoHigh() const { return fhHigh.get(); }

   void SetName(const std::string &Name) { fName = Name; }
   const std::string &GetName() const { return fName; }

   void SetInputFileLow(const std::string &InputFileLow) { fInputFileLow = InputFileLow; }
   void SetInputFileHigh(const std::string &InputFileHigh) { fInputFileHigh = InputFileHigh; }

   const std::string &GetInputFileLow() const { return fInputFileLow; }
   const std::string &GetInputFileHigh() const { return fInputFileHigh; }

   void SetHistoNameLow(const std::string &HistoNameLow) { fHistoNameLow = HistoNameLow; }
   void SetHistoNameHigh(const std::string &HistoNameHigh) { fHistoNameHigh = HistoNameHigh; }

   const std::string &GetHistoNameLow() const { return fHistoNameLow; }
   const std::string &GetHistoNameHigh() const { return fHistoNameHigh; }

   void SetHistoPathLow(const std::string &HistoPathLow) { fHistoPathLow = HistoPathLow; }
   void SetHistoPathHigh(const std::string &HistoPathHigh) { fHistoPathHigh = HistoPathHigh; }

   const std::string &GetHistoPathLow() const { return fHistoPathLow; }
   const std::string &GetHistoPathHigh() const { return fHistoPathHigh; }

protected:
   std::string fName;

   std::string fInputFileLow;
   std::string fHistoNameLow;
   std::string fHistoPathLow;

   std::string fInputFileHigh;
   std::string fHistoNameHigh;
   std::string fHistoPathHigh;

   // The Low and High Histograms
   std::unique_ptr<TH1> fhLow;
   std::unique_ptr<TH1> fhHigh;
};

/** \class HistoSys
 * \ingroup HistFactory
 * Configuration for a constrained, coherent shape variation of affected samples.
 */
class HistoSys final : public HistogramUncertaintyBase {
public:
   void PrintXML(std::ostream &) const override;
};

/** \class HistoFactor
 * \ingroup HistFactory
 * Configuration for an *un*constrained, coherent shape variation of affected samples.
 */
class HistoFactor final : public HistogramUncertaintyBase {
public:
   void PrintXML(std::ostream &) const override;
};

/** \class ShapeSys
 * \ingroup HistFactory
 * Constrained bin-by-bin variation of affected histogram.
 */
class ShapeSys final : public HistogramUncertaintyBase {

public:
   ShapeSys() : fConstraintType(Constraint::Gaussian) {}
   ShapeSys(const ShapeSys &other) : HistogramUncertaintyBase(other), fConstraintType(other.fConstraintType) {}
   ShapeSys &operator=(const ShapeSys &oth)
   {
      if (this == &oth)
         return *this;
      HistogramUncertaintyBase::operator=(oth);
      fConstraintType = oth.fConstraintType;
      return *this;
   }
   ShapeSys &operator=(ShapeSys &&) = default;

   void SetInputFile(const std::string &InputFile) { fInputFileHigh = InputFile; }
   std::string GetInputFile() const { return fInputFileHigh; }

   void SetHistoName(const std::string &HistoName) { fHistoNameHigh = HistoName; }
   std::string GetHistoName() const { return fHistoNameHigh; }

   void SetHistoPath(const std::string &HistoPath) { fHistoPathHigh = HistoPath; }
   std::string GetHistoPath() const { return fHistoPathHigh; }

   void Print(std::ostream & = std::cout) const override;
   void PrintXML(std::ostream &) const override;
   void writeToFile(const std::string &FileName, const std::string &DirName) override;

   const TH1 *GetErrorHist() const { return fhHigh.get(); }
   void SetErrorHist(TH1 *hError);

   void SetConstraintType(Constraint::Type ConstrType) { fConstraintType = ConstrType; }
   Constraint::Type GetConstraintType() const { return fConstraintType; }

protected:
   Constraint::Type fConstraintType;
};

/** \class ShapeFactor
 * \ingroup HistFactory
 * *Un*constrained bin-by-bin variation of affected histogram.
 */
class ShapeFactor : public HistogramUncertaintyBase {

public:
   void Print(std::ostream & = std::cout) const override;
   void PrintXML(std::ostream &) const override;
   void writeToFile(const std::string &FileName, const std::string &DirName) override;

   void SetInitialShape(TH1 *shape);
   const TH1 *GetInitialShape() const { return fhHigh.get(); }

   void SetConstant(bool constant) { fConstant = constant; }
   bool IsConstant() const { return fConstant; }

   bool HasInitialShape() const { return fHasInitialShape; }

   void SetInputFile(const std::string &InputFile)
   {
      fInputFileHigh = InputFile;
      fHasInitialShape = true;
   }
   const std::string &GetInputFile() const { return fInputFileHigh; }

   void SetHistoName(const std::string &HistoName)
   {
      fHistoNameHigh = HistoName;
      fHasInitialShape = true;
   }
   const std::string &GetHistoName() const { return fHistoNameHigh; }

   void SetHistoPath(const std::string &HistoPath)
   {
      fHistoPathHigh = HistoPath;
      fHasInitialShape = true;
   }
   const std::string &GetHistoPath() const { return fHistoPathHigh; }

protected:
   bool fConstant = false;

   // A histogram representing
   // the initial shape
   bool fHasInitialShape = false;
};

/** \class StatError
 * \ingroup HistFactory
 * Statistical error of Monte Carlo predictions.
 */
class StatError : public HistogramUncertaintyBase {

public:
   void Print(std::ostream & = std::cout) const override;
   void PrintXML(std::ostream &) const override;
   void writeToFile(const std::string &FileName, const std::string &DirName) override;

   void Activate(bool IsActive = true) { fActivate = IsActive; }
   bool GetActivate() const { return fActivate; }

   void SetUseHisto(bool UseHisto = true) { fUseHisto = UseHisto; }
   bool GetUseHisto() const { return fUseHisto; }

   void SetInputFile(const std::string &InputFile) { fInputFileHigh = InputFile; }
   const std::string &GetInputFile() const { return fInputFileHigh; }

   void SetHistoName(const std::string &HistoName) { fHistoNameHigh = HistoName; }
   const std::string &GetHistoName() const { return fHistoNameHigh; }

   void SetHistoPath(const std::string &HistoPath) { fHistoPathHigh = HistoPath; }
   const std::string &GetHistoPath() const { return fHistoPathHigh; }

   const TH1 *GetErrorHist() const { return fhHigh.get(); }
   void SetErrorHist(TH1 *Error);

protected:
   bool fActivate = false;
   bool fUseHisto = false; // Use an external histogram for the errors
};

/** \class StatErrorConfig
 * \ingroup HistFactory
 * Configuration to automatically assign nuisance parameters for the statistical
 * error of the Monte Carlo simulations.
 * The default is to assign a Poisson uncertainty to a bin when its statistical uncertainty
 * is larger than 5% of the bin content.
 */
class StatErrorConfig {

public:
   void Print(std::ostream & = std::cout) const;
   void PrintXML(std::ostream &) const;

   void SetRelErrorThreshold(double Threshold) { fRelErrorThreshold = Threshold; }
   double GetRelErrorThreshold() const { return fRelErrorThreshold; }

   void SetConstraintType(Constraint::Type ConstrType) { fConstraintType = ConstrType; }
   Constraint::Type GetConstraintType() const { return fConstraintType; }

protected:
   double fRelErrorThreshold = 0.05;
   Constraint::Type fConstraintType = Constraint::Poisson;
};

// Internal class wrapping an histogram and managing its content.
// conveninet for dealing with histogram pointers in the
// HistFactory class
class HistRef {

public:
   HistRef(TH1 *h = nullptr);

   HistRef(const HistRef &other);

   HistRef(HistRef &&other);

   ~HistRef();

   HistRef &operator=(const HistRef &other);
   HistRef &operator=(HistRef &&other);

   TH1 *GetObject() const;

   void SetObject(TH1 *h);

   void operator=(TH1 *h);

   TH1 *ReleaseObject();

private:
   static TH1 *CopyObject(const TH1 *h);
   std::unique_ptr<TH1> fHist; ///< pointer to contained histogram
};

class Asimov {

public:
   Asimov(std::string Name = "") : fName(Name) {}

   void ConfigureWorkspace(RooWorkspace *);

   std::string GetName() { return fName; }
   void SetName(const std::string &name) { fName = name; }

   void SetFixedParam(const std::string &param, bool constant = true) { fParamsToFix[param] = constant; }
   void SetParamValue(const std::string &param, double value) { fParamValsToSet[param] = value; }

   std::map<std::string, bool> &GetParamsToFix() { return fParamsToFix; }
   std::map<std::string, double> &GetParamsToSet() { return fParamValsToSet; }

protected:
   std::string fName;

   std::map<std::string, bool> fParamsToFix;
   std::map<std::string, double> fParamValsToSet;
};

class PreprocessFunction {
public:
   PreprocessFunction() {}

   PreprocessFunction(std::string const &name, std::string const &expression, std::string const &dependents);

   void Print(std::ostream & = std::cout) const;
   void PrintXML(std::ostream &) const;

   void SetName(const std::string &name) { fName = name; }
   std::string const &GetName() const { return fName; }

   void SetExpression(const std::string &expression) { fExpression = expression; }
   std::string const &GetExpression() const { return fExpression; }

   void SetDependents(const std::string &dependents) { fDependents = dependents; }
   std::string const &GetDependents() const { return fDependents; }

   std::string GetCommand() const;

private:
   std::string fName;
   std::string fExpression;
   std::string fDependents;
};

class Data {

public:
   // friend class Channel;

   Data() {}
   /// constructor from name, file and path. Name of the histogram should not include the path
   Data(std::string HistoName, std::string InputFile, std::string HistoPath = "");

   std::string const &GetName() const { return fName; }
   void SetName(const std::string &name) { fName = name; }

   void SetInputFile(const std::string &InputFile) { fInputFile = InputFile; }
   std::string const &GetInputFile() const { return fInputFile; }

   void SetHistoName(const std::string &HistoName) { fHistoName = HistoName; }
   std::string const &GetHistoName() const { return fHistoName; }

   void SetHistoPath(const std::string &HistoPath) { fHistoPath = HistoPath; }
   std::string const &GetHistoPath() const { return fHistoPath; }

   void Print(std::ostream & = std::cout);
   void PrintXML(std::ostream &) const;
   void writeToFile(std::string FileName, std::string DirName);

   TH1 *GetHisto();
   const TH1 *GetHisto() const;
   void SetHisto(TH1 *Hist);

protected:
   std::string fName;

   std::string fInputFile;
   std::string fHistoName;
   std::string fHistoPath;

   // The Data Histogram
   HistRef fhData;
};

class Sample {

public:
   Sample();
   ~Sample();
   Sample(std::string Name);
   Sample(const Sample &other);
   Sample &operator=(const Sample &other);
   /// constructor from name, file and path. Name of the histogram should not include the path
   Sample(std::string Name, std::string HistoName, std::string InputFile, std::string HistoPath = "");

   void Print(std::ostream & = std::cout) const;
   void PrintXML(std::ofstream &xml) const;
   void writeToFile(std::string FileName, std::string DirName);

   const TH1 *GetHisto() const;
   // set histogram for this sample
   void SetHisto(TH1 *histo);
   void SetValue(double Val);

   // Some helper functions
   // Note that histogram name should not include the path of the histogram in the file.
   // This has to be given separately

   void ActivateStatError();
   void ActivateStatError(std::string HistoName, std::string InputFile, std::string HistoPath = "");

   void AddOverallSys(std::string Name, double Low, double High);
   void AddOverallSys(const OverallSys &Sys);

   void AddNormFactor(std::string const &Name, double Val, double Low, double High);
   void AddNormFactor(const NormFactor &Factor);

   void AddHistoSys(std::string Name, std::string HistoNameLow, std::string HistoFileLow, std::string HistoPathLow,
                    std::string HistoNameHigh, std::string HistoFileHigh, std::string HistoPathHigh);
   void AddHistoSys(const HistoSys &Sys);

   void AddHistoFactor(std::string Name, std::string HistoNameLow, std::string HistoFileLow, std::string HistoPathLow,
                       std::string HistoNameHigh, std::string HistoFileHigh, std::string HistoPathHigh);
   void AddHistoFactor(const HistoFactor &Factor);

   void AddShapeFactor(std::string Name);
   void AddShapeFactor(const ShapeFactor &Factor);

   void AddShapeSys(std::string Name, Constraint::Type ConstraintType, std::string HistoName, std::string HistoFile,
                    std::string HistoPath = "");
   void AddShapeSys(const ShapeSys &Sys);

   /// defines whether the normalization scale with luminosity
   void SetNormalizeByTheory(bool norm) { fNormalizeByTheory = norm; }
   /// does the normalization scale with luminosity
   bool GetNormalizeByTheory() const { return fNormalizeByTheory; }

   /// get name of sample
   std::string GetName() const { return fName; }
   /// set name of sample
   void SetName(const std::string &Name) { fName = Name; }

   /// get input ROOT file
   std::string GetInputFile() const { return fInputFile; }
   /// set input ROOT file
   void SetInputFile(const std::string &InputFile) { fInputFile = InputFile; }

   /// get histogram name
   std::string GetHistoName() const { return fHistoName; }
   /// set histogram name
   void SetHistoName(const std::string &HistoName) { fHistoName = HistoName; }

   /// get histogram path
   std::string GetHistoPath() const { return fHistoPath; }
   /// set histogram path
   void SetHistoPath(const std::string &HistoPath) { fHistoPath = HistoPath; }

   /// get name of associated channel
   std::string GetChannelName() const { return fChannelName; }
   /// set name of associated channel
   void SetChannelName(const std::string &ChannelName) { fChannelName = ChannelName; }

   std::vector<RooStats::HistFactory::OverallSys> &GetOverallSysList() { return fOverallSysList; }
   std::vector<RooStats::HistFactory::NormFactor> &GetNormFactorList() { return fNormFactorList; }
   std::vector<RooStats::HistFactory::HistoSys> &GetHistoSysList() { return fHistoSysList; }
   std::vector<RooStats::HistFactory::HistoFactor> &GetHistoFactorList() { return fHistoFactorList; }
   std::vector<RooStats::HistFactory::ShapeSys> &GetShapeSysList() { return fShapeSysList; }
   std::vector<RooStats::HistFactory::ShapeFactor> &GetShapeFactorList() { return fShapeFactorList; }

   const std::vector<RooStats::HistFactory::OverallSys> &GetOverallSysList() const { return fOverallSysList; }
   const std::vector<RooStats::HistFactory::NormFactor> &GetNormFactorList() const { return fNormFactorList; }
   const std::vector<RooStats::HistFactory::HistoSys> &GetHistoSysList() const { return fHistoSysList; }
   const std::vector<RooStats::HistFactory::HistoFactor> &GetHistoFactorList() const { return fHistoFactorList; }
   const std::vector<RooStats::HistFactory::ShapeSys> &GetShapeSysList() const { return fShapeSysList; }
   const std::vector<RooStats::HistFactory::ShapeFactor> &GetShapeFactorList() const { return fShapeFactorList; }

   bool HasStatError() const { return fStatErrorActivate; }
   RooStats::HistFactory::StatError &GetStatError() { return fStatError; }
   const RooStats::HistFactory::StatError &GetStatError() const { return fStatError; }
   void SetStatError(RooStats::HistFactory::StatError Error) { fStatError = std::move(Error); }

protected:
   std::string fName;
   std::string fInputFile;
   std::string fHistoName;
   std::string fHistoPath;

   /// The Name of the parent channel
   std::string fChannelName;

   //
   // Systematics
   //

   std::vector<RooStats::HistFactory::OverallSys> fOverallSysList;
   std::vector<RooStats::HistFactory::NormFactor> fNormFactorList;

   std::vector<RooStats::HistFactory::HistoSys> fHistoSysList;
   std::vector<RooStats::HistFactory::HistoFactor> fHistoFactorList;

   std::vector<RooStats::HistFactory::ShapeSys> fShapeSysList;
   std::vector<RooStats::HistFactory::ShapeFactor> fShapeFactorList;

   /// Properties
   RooStats::HistFactory::StatError fStatError;

   bool fNormalizeByTheory = false;
   bool fStatErrorActivate = false;

   /// The Nominal Shape
   HistRef fhNominal;
   std::unique_ptr<TH1> fhCountingHist;
};

class Channel {

public:
   friend class Measurement;

   Channel() = default;
   Channel(std::string Name, std::string InputFile = "");

   /// set name of channel
   void SetName(const std::string &Name) { fName = Name; }
   /// get name of channel
   std::string GetName() const { return fName; }
   /// set name of input file containing histograms
   void SetInputFile(const std::string &file) { fInputFile = file; }
   /// get name of input file
   std::string GetInputFile() const { return fInputFile; }
   /// set path for histograms in input file
   void SetHistoPath(const std::string &file) { fHistoPath = file; }
   /// get path to histograms in input file
   std::string GetHistoPath() const { return fHistoPath; }

   /// set data object
   void SetData(const RooStats::HistFactory::Data &data) { fData = data; }
   void SetData(std::string HistoName, std::string InputFile, std::string HistoPath = "");
   void SetData(double Val);
   void SetData(TH1 *hData);
   /// get data object
   RooStats::HistFactory::Data &GetData() { return fData; }
   const RooStats::HistFactory::Data &GetData() const { return fData; }

   /// add additional data object
   void AddAdditionalData(const RooStats::HistFactory::Data &data) { fAdditionalData.push_back(data); }
   /// retrieve vector of additional data objects
   std::vector<RooStats::HistFactory::Data> &GetAdditionalData() { return fAdditionalData; }

   void SetStatErrorConfig(double RelErrorThreshold, Constraint::Type ConstraintType);
   void SetStatErrorConfig(double RelErrorThreshold, std::string ConstraintType);
   /// define treatment of statistical uncertainties
   void SetStatErrorConfig(RooStats::HistFactory::StatErrorConfig Config) { fStatErrorConfig = Config; }
   /// get information about threshold for statistical uncertainties and constraint term
   HistFactory::StatErrorConfig &GetStatErrorConfig() { return fStatErrorConfig; }
   const HistFactory::StatErrorConfig &GetStatErrorConfig() const { return fStatErrorConfig; }

   void AddSample(RooStats::HistFactory::Sample sample);
   /// get vector of samples for this channel
   std::vector<RooStats::HistFactory::Sample> &GetSamples() { return fSamples; }
   const std::vector<RooStats::HistFactory::Sample> &GetSamples() const { return fSamples; }

   void Print(std::ostream & = std::cout);
   void PrintXML(std::string const &directory, std::string const &prefix = "") const;

   void CollectHistograms();
   bool CheckHistograms() const;

protected:
   std::string fName;
   std::string fInputFile;
   std::string fHistoPath;

   HistFactory::Data fData;

   /// One can add additional datasets
   /// These are simply added to the xml under a different name
   std::vector<RooStats::HistFactory::Data> fAdditionalData;

   HistFactory::StatErrorConfig fStatErrorConfig;

   std::vector<RooStats::HistFactory::Sample> fSamples;

   TH1 *GetHistogram(std::string InputFile, std::string HistoPath, std::string HistoName,
                     std::map<std::string, std::unique_ptr<TFile>> &lsof);
};

extern Channel BadChannel;

class Measurement : public TNamed {

public:
   Measurement() = default;
   /// Constructor specifying name and title of measurement
   Measurement(const char *Name, const char *Title = "") : TNamed(Name, Title) {}

   ///  set output prefix
   void SetOutputFilePrefix(const std::string &prefix) { fOutputFilePrefix = prefix; }
   /// retrieve prefix for output files
   std::string GetOutputFilePrefix() { return fOutputFilePrefix; }

   /// insert PoI at beginning of vector of PoIs
   void SetPOI(const std::string &POI) { fPOI.insert(fPOI.begin(), POI); }
   /// append parameter to vector of PoIs
   void AddPOI(const std::string &POI) { fPOI.push_back(POI); }
   /// get name of PoI at given index
   std::string GetPOI(unsigned int i = 0) { return fPOI.at(i); }
   /// get vector of PoI names
   std::vector<std::string> &GetPOIList() { return fPOI; }

   /// Add a parameter to be set as constant
   /// (Similar to ParamSetting method below)
   void AddConstantParam(const std::string &param);
   /// empty vector of constant parameters
   void ClearConstantParams() { fConstantParams.clear(); }
   /// get vector of all constant parameters
   std::vector<std::string> &GetConstantParams() { return fConstantParams; }

   /// Set a parameter to a specific value
   /// (And optionally fix it)
   void SetParamValue(const std::string &param, double value);
   /// get map: parameter name <--> parameter value
   std::map<std::string, double> &GetParamValues() { return fParamValues; }
   /// clear map of parameter values
   void ClearParamValues() { fParamValues.clear(); }

   void AddPreprocessFunction(std::string name, std::string expression, std::string dependencies);
   /// add a preprocess function object
   void AddFunctionObject(const RooStats::HistFactory::PreprocessFunction function)
   {
      fFunctionObjects.push_back(function);
   }
   void SetFunctionObjects(std::vector<RooStats::HistFactory::PreprocessFunction> objects)
   {
      fFunctionObjects = objects;
   }
   /// get vector of defined function objects
   std::vector<RooStats::HistFactory::PreprocessFunction> &GetFunctionObjects() { return fFunctionObjects; }
   const std::vector<RooStats::HistFactory::PreprocessFunction> &GetFunctionObjects() const { return fFunctionObjects; }
   std::vector<std::string> GetPreprocessFunctions() const;

   /// get vector of defined Asimov Datasets
   std::vector<RooStats::HistFactory::Asimov> &GetAsimovDatasets() { return fAsimovDatasets; }
   /// add an Asimov Dataset
   void AddAsimovDataset(RooStats::HistFactory::Asimov dataset) { fAsimovDatasets.push_back(dataset); }

   /// set integrated luminosity used to normalise histograms (if NormalizeByTheory is true for this sample)
   void SetLumi(double Lumi) { fLumi = Lumi; }
   /// set relative uncertainty on luminosity
   void SetLumiRelErr(double RelErr) { fLumiRelErr = RelErr; }
   /// retrieve integrated luminosity
   double GetLumi() { return fLumi; }
   /// retrieve relative uncertainty on luminosity
   double GetLumiRelErr() { return fLumiRelErr; }

   void SetBinLow(int BinLow) { fBinLow = BinLow; }
   void SetBinHigh(int BinHigh) { fBinHigh = BinHigh; }
   int GetBinLow() { return fBinLow; }
   int GetBinHigh() { return fBinHigh; }

   void PrintTree(std::ostream & = std::cout); /// Print to a stream
   void PrintXML(std::string Directory = "", std::string NewOutputPrefix = "");

   std::vector<RooStats::HistFactory::Channel> &GetChannels() { return fChannels; }
   const std::vector<RooStats::HistFactory::Channel> &GetChannels() const { return fChannels; }
   RooStats::HistFactory::Channel &GetChannel(std::string);
   /// Add a completely configured channel.
   void AddChannel(RooStats::HistFactory::Channel chan) { fChannels.push_back(chan); }

   bool HasChannel(std::string);
   void writeToFile(TFile *file);

   void CollectHistograms();

   void AddGammaSyst(std::string syst, double uncert);
   void AddLogNormSyst(std::string syst, double uncert);
   void AddUniformSyst(std::string syst);
   void AddNoSyst(std::string syst);

   std::map<std::string, double> &GetGammaSyst() { return fGammaSyst; }
   std::map<std::string, double> &GetUniformSyst() { return fUniformSyst; }
   std::map<std::string, double> &GetLogNormSyst() { return fLogNormSyst; }
   std::map<std::string, double> &GetNoSyst() { return fNoSyst; }

   std::map<std::string, double> const &GetGammaSyst() const { return fGammaSyst; }
   std::map<std::string, double> const &GetUniformSyst() const { return fUniformSyst; }
   std::map<std::string, double> const &GetLogNormSyst() const { return fLogNormSyst; }
   std::map<std::string, double> const &GetNoSyst() const { return fNoSyst; }

   std::string GetInterpolationScheme() { return fInterpolationScheme; }

private:
   /// Configurables of this measurement
   std::string fOutputFilePrefix;
   std::vector<std::string> fPOI;
   double fLumi = 1.0;
   double fLumiRelErr = 0.1;
   int fBinLow = 0;
   int fBinHigh = 1;
   bool fExportOnly = true;
   std::string fInterpolationScheme;

   /// Channels that make up this measurement
   std::vector<RooStats::HistFactory::Channel> fChannels;

   /// List of Parameters to be set constant
   std::vector<std::string> fConstantParams;

   /// Map of parameter names to initial values to be set
   std::map<std::string, double> fParamValues;

   /// List of Preprocess Function objects
   std::vector<RooStats::HistFactory::PreprocessFunction> fFunctionObjects;

   /// List of Asimov datasets to generate
   std::vector<RooStats::HistFactory::Asimov> fAsimovDatasets;

   /// List of Alternate constraint terms
   std::map<std::string, double> fGammaSyst;
   std::map<std::string, double> fUniformSyst;
   std::map<std::string, double> fLogNormSyst;
   std::map<std::string, double> fNoSyst;

   std::string GetDirPath(TDirectory *dir);

   ClassDefOverride(RooStats::HistFactory::Measurement, 3);
};

} // namespace RooStats::HistFactory

#endif
