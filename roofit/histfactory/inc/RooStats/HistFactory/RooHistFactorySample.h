/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooStats_HistFactory_RooHistFactorySample_h
#define RooStats_HistFactory_RooHistFactorySample_h

#include <RooAbsReal.h>
#include <RooListProxy.h>

#include <ROOT/RSpan.hxx>

#include <memory>
#include <string>
#include <vector>

class RooHistFactorySample : public RooAbsReal {
public:
   /// The types of modifiers that scale each bin with its own parameter.
   enum class BinFactorType {
      StatError = 0,
      ShapeSys = 1,
      ShapeFactor = 2
   };

   /// Scales the sample with a parameter, or with a function of the parameters (a "custom" modifier in HS3).
   struct NormFactor {
      std::string name;
      RooAbsReal *param = nullptr;
   };

   /// Scales the sample with a factor that is interpolated between `lo` and `hi`. The interpolation codes are the ones
   /// of FlexibleInterpVar.
   struct NormSys {
      std::string name;
      RooAbsReal *param = nullptr;
      double lo = 1.0;
      double hi = 1.0;
      int interpCode = 4;
   };

   /// Interpolates the bin contents between the `lo` and `hi` variations. The interpolation codes are the ones of
   /// PiecewiseInterpolation.
   struct HistoSys {
      std::string name;
      RooAbsReal *param = nullptr;
      std::vector<double> lo;
      std::vector<double> hi;
      int interpCode = 4;
   };

   /// Scales each bin with its own parameter, given in the order of the bins.
   struct BinFactor {
      std::string name;
      BinFactorType type = BinFactorType::ShapeFactor;
      RooArgList params;
   };

   /// Everything that defines a sample. The bins are ordered like in RooDataHist and HS3: the last observable varies
   /// fastest.
   struct Spec {
      std::string sampleName;
      std::vector<double> contents;
      std::vector<double> errors; ///< Absolute statistical uncertainties, optional.
      std::vector<NormFactor> normFactors;
      std::vector<NormSys> normSys;
      std::vector<HistoSys> histoSys;
      std::vector<BinFactor> binFactors;
   };

   /// The immutable data of a sample, shared between the clones of a sample.
   struct Data {
      std::string sampleName;
      std::vector<std::vector<double>> edges; ///< The bin boundaries of each observable.
      std::vector<char> uniform;              ///< If the binning of an observable is uniform.
      std::vector<double> contents;
      std::vector<double> errors;

      std::vector<std::string> normFactorNames;
      std::vector<int> normFactorParams;

      std::vector<std::string> normSysNames;
      std::vector<int> normSysParams;
      std::vector<double> normSysLo;
      std::vector<double> normSysHi;
      std::vector<int> normSysCodes;

      std::vector<std::string> histoSysNames;
      std::vector<int> histoSysParams;
      std::vector<int> histoSysCodes;
      std::vector<double> histoSysLo; ///< One variation after the other.
      std::vector<double> histoSysHi; ///< One variation after the other.

      std::vector<std::string> binFactorNames;
      std::vector<int> binFactorTypes;
      std::vector<int> binFactorParams; ///< One modifier after the other, one parameter for each bin.

      std::size_t numBins() const { return contents.size(); }

      ClassDefNV(Data, 1);
   };

   RooHistFactorySample() = default;
   RooHistFactorySample(const char *name, const char *title, RooArgList const &observables, Spec const &spec);
   RooHistFactorySample(const RooHistFactorySample &other, const char *name = nullptr);
   TObject *clone(const char *newname = nullptr) const override { return new RooHistFactorySample(*this, newname); }

   std::string const &sampleName() const { return _data->sampleName; }
   RooArgList const &observables() const { return _obs; }
   RooArgList const &parameters() const { return _params; }
   Data const &data() const { return *_data; }
   std::size_t numBins() const { return _data->numBins(); }

   double binVolume(std::size_t iBin) const;
   void binValues(std::span<double> out, bool density) const;

   void setHistoSysInterpCode(RooAbsReal const &param, int code);
   void setNormSysInterpCode(RooAbsReal const &param, int code);

   bool isBinnedDistribution(const RooArgSet &obs) const override;
   std::list<double> *binBoundaries(RooAbsRealLValue &obs, double xlo, double xhi) const override;
   std::list<double> *plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const override;

   Int_t getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(Int_t code, const char *rangeName = nullptr) const override;
   std::vector<double> integralWeights(Int_t code, const char *rangeName = nullptr) const;

   std::unique_ptr<RooAbsArg>
   compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext &ctx) const override;

   /// If the values are the yields in the bins, which is the case in the binned likelihood. Otherwise, they are the
   /// densities, i.e. the yields divided by the bin volumes.
   bool yieldsMode() const { return _yieldsMode; }

   void printMetaArgs(std::ostream &os) const override;

protected:
   double evaluate() const override;
   void doEval(RooFit::EvalContext &ctx) const override;

private:
   int observableIndex(RooAbsArg const &obs) const;
   std::size_t binIndex(std::span<const double> values) const;
   double binValue(double const *params, std::size_t iBin) const;
   double scaleFactor(double const *params) const;
   template <class ParamValue>
   void fillParams(ParamValue &&paramValue) const;
   void setInterpCode(std::vector<int> Data::*codes, std::vector<int> Data::*params, RooAbsReal const &param, int code,
                      int maxCode, const char *what);

   RooListProxy _obs;
   RooListProxy _params;
   std::shared_ptr<const Data> _data; ///<! Custom streamer
   bool _yieldsMode = false;          ///<!

   mutable std::vector<double> _paramVals; ///<!
   mutable std::vector<double> _binVals;   ///<!
   mutable std::vector<double> _obsVals;   ///<!

   ClassDefOverride(RooHistFactorySample, 1);
};

#endif
