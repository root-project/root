// Author: Stephan Hageboeck, CERN  26 Jul 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "VectorisedPDFTests.h"

#include "RooRealVar.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooAbsRealLValue.h"
#include "RooGaussian.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooRandom.h"
#include "RooConstVar.h"
#include "Math/Util.h"
#include "RooHelpers.h"

#include <memory>
#include <numeric>
#include <ctime>
#include <chrono>

#ifdef __INTEL_COMPILER
#include "ittnotify.h"
#else
void __itt_resume() {}
void __itt_pause() {}
#endif

class MyTimer {
  public:
    MyTimer(std::string&& name)
  : m_name(name), m_startTime(clock()), m_endTime(0),
    m_steadyStart(std::chrono::steady_clock::now()), m_steadyEnd() { }

    clock_t diffTime() const {
      return clock() - m_startTime;
    }

    void interval() {
      m_endTime = clock();
      m_steadyEnd = std::chrono::steady_clock::now();
    }

    void print(std::ostream& str) {
      clock_t diff = m_endTime - m_startTime;
      std::chrono::duration<double> diffSteady = m_steadyEnd - m_steadyStart;
      str << "\n" << "Timer '" << m_name << "':\t" << double(diff)/CLOCKS_PER_SEC << "s (CPU) "
          << diffSteady.count() << "s (wall)" << std::endl;
    }

  private:
    std::string m_name;
    clock_t m_startTime;
    clock_t m_endTime;
    std::chrono::time_point<std::chrono::steady_clock> m_steadyStart;
    std::chrono::time_point<std::chrono::steady_clock> m_steadyEnd;
};

std::ostream& operator<<(std::ostream& str, MyTimer& timer) {
  timer.interval();
  timer.print(str);
  return str;
}

PDFTest::PDFTest(std::string&& name, std::size_t nEvt) :
_name(name),
_nEvents(nEvt)
{
  //Shut up integration messages
  auto& msg = RooMsgService::instance();
  msg.getStream(0).minLevel = RooFit::WARNING;
  msg.getStream(1).minLevel = RooFit::WARNING;

  RooRandom::randomGenerator()->SetSeed(1337);
}

void PDFTest::SetUp() {
  _origParameters.addClone(_parameters);
  _origYields.addClone(_yields);
}

void PDFTest::makeFitData() {
  _dataFit.reset(_pdf->generate(_variables, _nEvents));
}

void PDFTest::makeUniformData() {
  RooDataSet* data = new RooDataSet("testData", "testData", _variables);
  for (auto var : _variables) {
    auto lv = static_cast<RooRealVar*>(var);
    const double max = lv->getMax();
    const double min = lv->getMin();
    unsigned int nBatch = _nEvents/_variables.size();
    for (unsigned int i=0; i < nBatch; ++i) {
      lv->setVal(min + (max - min)/nBatch * i);
      data->add(_variables);
    }
  }

  _dataUniform.reset(data);
}

void PDFTest::randomiseParameters(ULong_t seed) {
  auto random = RooRandom::randomGenerator();
  random->SetSeed(seed);

  for (auto param : _parameters) {
    auto par = static_cast<RooAbsRealLValue*>(param);
    const double uni = random->Uniform();
    const double min = par->getMin();
    const double max = par->getMax();
    par->setVal(min + uni*(max-min));
  }
}

void PDFTest::makePlots(std::string&& fitStage) const{
  for (auto elm : _variablesToPlot) {
    auto var = static_cast<RooRealVar*>(elm);
    auto canv = std::make_unique<TCanvas>();
    auto frame = std::unique_ptr<RooPlot>(var->frame());
    _dataFit->plotOn(frame.get());
    _pdf->plotOn(frame.get(), RooFit::Precision(-1.));
    _pdf->paramOn(frame.get());
    frame->Draw();
    canv->Draw();
    std::string filename = _plotDirectory + _name + "_";
    filename += var->GetName();
    filename += "_" + fitStage + ".png";
    std::replace(filename.begin(), filename.end(), ' ', '_');
    canv->SaveAs(filename.c_str());
  }
}


void PDFTest::setValuesConstant(const RooAbsCollection& coll, bool constant) const {
  for (auto obj : coll) {
    auto lvalue = dynamic_cast<RooAbsRealLValue*>(obj);
    if (lvalue)
      lvalue->setConstant(constant);
  }
}

void PDFTest::resetParameters() {
  _parameters = _origParameters;
}

void PDFTest::kickParameters() {

  //Kick parameters away from best-fit value
  for (auto param : _parameters) {
    auto lval = static_cast<RooAbsRealLValue*>(param);
    auto orig = static_cast<RooAbsRealLValue*>(_origParameters.find(param->GetName()));
    if (orig->isConstant())
      continue;

    *lval = orig->getVal() * 1.3 + (orig->getVal() == 0. ? 0.1 : 0.);
  }

  for (auto yield : _yields) {
    auto lval = static_cast<RooAbsRealLValue*>(yield);
    auto orig = static_cast<RooAbsRealLValue*>(_origYields.find(yield->GetName()));
    if (orig->isConstant())
      continue;

    *lval = orig->getVal() * 1.3 + (orig->getVal() == 0. ? 0.1 : 0.);
  }

  setValuesConstant(_otherObjects, true);
}



void PDFTest::compareFixedValues(double& maximalError, bool normalise, bool compareLogs, bool runTimer, unsigned int nChunks) {
  if (!_dataUniform)
    makeUniformData();

  if (nChunks == 0) {
    nChunks = 1;
  }

  const RooArgSet* normSet = nullptr;
  std::string timerSuffix = normalise ? " norm " :" unnorm ";
  if (compareLogs) timerSuffix = " (logs)" + timerSuffix;

  const double toleranceCompare = compareLogs ? _toleranceCompareLogs : _toleranceCompareBatches;

  RooArgSet* observables = _pdf->getObservables(*_dataUniform);
  RooArgSet* parameters  = _pdf->getParameters(*_dataUniform);

  std::vector<RooBatchCompute::RunContext> evalData;
  auto callBatchFunc = [compareLogs,&evalData,this](const RooAbsPdf& pdf, std::size_t begin, std::size_t len, const RooArgSet* theNormSet)
      -> RooSpan<const double> {
    evalData.emplace_back();
    _dataUniform->getBatches(evalData.back(), begin, len);

    if (compareLogs) {
      return pdf.getLogProbabilities(evalData.back(), theNormSet);
    } else {
      return pdf.getValues(evalData.back(), theNormSet);
    }
  };

  auto callScalarFunc = [compareLogs](const RooAbsPdf& pdf, const RooArgSet* theNormSet) {
    if (compareLogs)
      return pdf.getLogVal(theNormSet);
    else
      return pdf.getVal(theNormSet);
  };

  if (normalise) {
    normSet = &_variables;
  }

  std::vector<RooSpan<const double>> batchResults;
  MyTimer batchTimer("Evaluate batch" + timerSuffix + _name);
  __itt_resume();
  const std::size_t chunkSize = _dataUniform->numEntries() / nChunks + (_dataUniform->numEntries() % nChunks != 0);
  for (unsigned int chunk = 0; chunk < nChunks; ++chunk) {
    auto outputsBatch = callBatchFunc(*_pdf,
        chunkSize * chunk,
        chunk+1 < nChunks ? chunkSize : _dataUniform->numEntries() / nChunks,
        normSet);
    batchResults.push_back(outputsBatch);
  }
  __itt_pause();
  if (runTimer)
    std::cout << batchTimer;

  const auto totalSize = std::accumulate(batchResults.begin(),  batchResults.end(), 0,
      [](std::size_t acc, const RooSpan<const double>& span){ return acc + span.size(); });
  ASSERT_EQ(totalSize, _dataUniform->numEntries());

  for (auto& outputsBatch : batchResults) {
    const double front = outputsBatch[0];
    const bool allEqual = std::all_of(outputsBatch.begin(), outputsBatch.end(),
        [front](double val){
      return val == front;
    });
    ASSERT_FALSE(allEqual) << "All return values of batch run equal. "
        << outputsBatch[0] << " " << outputsBatch[1] << " " << outputsBatch[2];
  }

  _dataUniform->resetBuffers();




  // Scalar run
  std::vector<double> outputsScalar(_dataUniform->numEntries(), -1.);
  if (normalise) {
    //      normSet = new RooArgSet(*observables);
    normSet = &_variables;
  }
  *parameters = _parameters;

  {
    MyTimer singleTimer("Evaluate scalar" + timerSuffix + _name);
    for (int i=0; i < _dataUniform->numEntries(); ++i) {
      *observables = *_dataUniform->get(i);
      outputsScalar[i] = callScalarFunc(*_pdf, normSet);
    }
    if (runTimer)
      std::cout << singleTimer;
  }

  const bool outputsChanged = std::any_of(outputsScalar.begin(), outputsScalar.end(),
      [](double val){
    return val != -1.;
  });
  ASSERT_TRUE(outputsChanged) << "All return values of scalar run are -1.";

  const double frontSc = outputsScalar.front();
  const bool allEqualSc = std::all_of(outputsScalar.begin(), outputsScalar.end(),
      [frontSc](double val){
    return val == frontSc;
  });
  ASSERT_FALSE(allEqualSc) << "All return values of scalar run equal.\n\t"
      << outputsScalar[0] << " " << outputsScalar[1] << " " << outputsScalar[2] << " "
      << outputsScalar[3] << " " << outputsScalar[4] << " " << outputsScalar[5] << " ...";



  // Compare runs
  unsigned int nOff = 0;
  unsigned int nFarOff = 0;
  constexpr double thresholdFarOff = 1.E-9;

  maximalError = 0.0;
  ROOT::Math::KahanSum<> sumDiffs;
  ROOT::Math::KahanSum<> sumVars;
  auto currentBatch = batchResults.begin();
  unsigned int currentBatchIndex = 0;
  for (std::size_t i=0; i < outputsScalar.size(); ++i, ++currentBatchIndex) {
    if (currentBatchIndex >= currentBatch->size()) {
      ++currentBatch;
      ASSERT_TRUE(currentBatch != batchResults.end());
      currentBatchIndex = 0;
    }
    const double batchVal = (*currentBatch)[currentBatchIndex];

    const double relDiff = batchVal != 0. ?
        (outputsScalar[i]-batchVal)/batchVal
        : outputsScalar[i];

    maximalError = std::max(maximalError,fabs(relDiff));
    sumDiffs += relDiff;
    sumVars  += relDiff * relDiff;


    // Check accuracy of computations, but give it some leniency for very small likelihoods
    if ( (fabs(relDiff) > toleranceCompare && fabs(outputsScalar[i]) > 1.E-50 ) ||
         (fabs(relDiff) > 1.E-10 && fabs(outputsScalar[i]) > 1.E-300) ||
         (fabs(relDiff) > 1.E-9)) {
      if (nOff < 5) {
        *observables = *_dataUniform->get(i);
        std::cout << "Compare event " << i << "\t" << std::setprecision(15);
        observables->printStream(std::cout, RooPrintable::kValue | RooPrintable::kName, RooPrintable::kStandard, "  ");
        _parameters.Print("V");
        std::cout << "\n\tscalar   = " << outputsScalar[i] << "\tpdf->getVal() = " << _pdf->getVal()
                        << "\n\tbatch    = " << batchVal
                                                             << "\n\trel diff = " << relDiff << std::endl;
      }
      ++nOff;

      if (fabs(relDiff) > thresholdFarOff)
        ++nFarOff;

#ifdef ROOFIT_CHECK_CACHED_VALUES
      try {
        *observables = *_dataUniform->get(i);
        _pdf->getVal(normSet);

        BatchInterfaceAccessor::checkBatchComputation(*_pdf,
            evalData[currentBatch-batchResults.begin()],
            currentBatchIndex, normSet, toleranceCompare);

      } catch (std::exception& e) {
        ADD_FAILURE() << " ERROR when checking batch computation for event " << i << ":\n"
            << e.what() << "\n"
            << "PDF is:"<< std::endl;
        _pdf->Print("T");
      }
#endif
    }
  }

  EXPECT_LT(nOff, 5u);
  EXPECT_EQ(nFarOff, 0u);
  EXPECT_GT(sumDiffs/outputsScalar.size(), -toleranceCompare) << "Batch outputs biased towards negative.";
  EXPECT_LT(sumDiffs/outputsScalar.size(), toleranceCompare)  << "Batch outputs biased towards positive.";
  EXPECT_LT(sqrt(sumVars/outputsScalar.size()), toleranceCompare) << "High standard deviation for batch results vs scalar.";
}


void PDFTest::checkParameters() {
  ASSERT_FALSE(_parameters.overlaps(_otherObjects)) << "Collections of parameters and other objects "
      << "cannot overlap. This will lead to wrong results, as parameters get kicked before the fit, "
      << "other objects are set constant. Hence, the fit cannot change them.";
  ASSERT_FALSE(_yields.overlaps(_otherObjects)) << "Collections of yields and other objects "
      << "cannot overlap. This will lead to wrong results, as parameters get kicked before the fit, "
      << "other objects are set constant. Hence, the fit cannot change them.";

  for (auto param : _parameters) {
    auto postFit = static_cast<RooRealVar*>(param);
    auto preFit  = static_cast<RooRealVar*>(_origParameters.find(param->GetName()));
    ASSERT_NE(preFit, nullptr) << "for parameter '" << param->GetName() << '\'';
    EXPECT_LE(fabs(postFit->getVal() - preFit->getVal()), 2.*postFit->getError())
    << "[Within 2 std-dev: " << param->GetName()
    << " (" << postFit->getVal() << " +- " << 2.*postFit->getError() << ")"
    << " == " << preFit->getVal() << "]";

    EXPECT_LE(fabs(postFit->getVal() - preFit->getVal()), 1.5*postFit->getError())
    << "[Within 1.5 std-dev: " << param->GetName()
    << " (" << postFit->getVal() << " +- " << 1.5*postFit->getError() << ")"
    << " == " << preFit->getVal() << "]";

    EXPECT_NEAR(postFit->getVal(), preFit->getVal(), fabs(postFit->getVal())*5.E-2)
    << "[Within 5% for parameter '" << param->GetName() << "']";

  }

  if (!_yields.empty()) {
    const double totalPre = std::accumulate(_origYields.begin(), _origYields.end(), 0.,
        [](double acc, const RooAbsArg* arg){
      return acc + static_cast<const RooAbsReal*>(arg)->getVal();
    });
    const double totalPost = std::accumulate(_yields.begin(), _yields.end(), 0.,
        [](double acc, const RooAbsArg* arg){
      return acc + static_cast<const RooAbsReal*>(arg)->getVal();
    });
    ASSERT_NE(totalPre, 0.);
    ASSERT_NE(totalPost, 0.);
    ASSERT_LE(fabs(totalPost - _nEvents) / _nEvents, 0.1) << "Total event yield not matching"
        << " number of generated events.";

    for (auto yield : _yields) {
      auto postFit = static_cast<RooRealVar*>(yield);
      auto preFit  = static_cast<RooRealVar*>(_origYields.find(yield->GetName()));
      ASSERT_NE(preFit, nullptr) << "for parameter '" << yield->GetName() << '\'';

      EXPECT_NEAR(postFit->getVal()/totalPost,
          preFit->getVal()/totalPre, 0.01) << "Yield " << yield->GetName()
                   << " = " << postFit->getVal()
                   << " does not match pre-fit ratios.";
    }
  }
}

void PDFTest::runBatchVsScalar(bool clonePDF) {
  RooAbsPdf* pdfScalar = _pdf.get();
  RooAbsPdf* pdfBatch  = _pdf.get();
  std::unique_ptr<RooAbsPdf> cleanupScalar;
  std::unique_ptr<RooAbsPdf> cleanupBatch;

  if (clonePDF) {
    pdfScalar = static_cast<RooAbsPdf*>(_pdf->cloneTree("PDFForScalar"));
    pdfBatch  = static_cast<RooAbsPdf*>(_pdf->cloneTree("PDFForScalar"));

    cleanupScalar.reset(pdfScalar);
    cleanupBatch.reset(pdfBatch);
  }

  auto resultScalar = runScalarFit(pdfScalar);
  auto resultBatch = runBatchFit(pdfBatch);

  resetParameters();

  ASSERT_NE(resultScalar, nullptr);
  ASSERT_NE(resultBatch,  nullptr);

  EXPECT_TRUE(resultScalar->isIdentical(*resultBatch, _toleranceParameter, _toleranceCorrelation));
}

std::unique_ptr<RooFitResult> PDFTest::runBatchFit(RooAbsPdf* pdf) {
  if (!_dataFit)
    makeFitData();

  kickParameters();
  makePlots(::testing::UnitTest::GetInstance()->current_test_info()->name()+std::string("_batch_prefit"));

  auto pars = pdf->getParameters(*_dataFit);
  *pars = _parameters;

  for (unsigned int index = 0; index < pars->size(); ++index) {
    auto pdfParameter = static_cast<RooAbsReal*>((*pars)[index]);
    auto origParameter = static_cast<RooAbsReal*>(_origParameters.find(*pdfParameter));
    if (!origParameter || origParameter->isConstant())
      continue;

    EXPECT_NE(pdfParameter->getVal(), origParameter->getVal())
        << "Parameter #" << index << "=" << pdfParameter->GetName() << " is identical after kicking.";
  }

  if (HasFailure()) {
    std::cout << "Pre-fit parameters:\n";
    _parameters.Print("V");
    std::cout << "Orig parameters:\n";
    _origParameters.Print("V");
  }

  MyTimer batchTimer("Fitting batch mode " + _name);
  auto result = pdf->fitTo(*_dataFit,
      RooFit::BatchMode(true),
      RooFit::SumW2Error(false),
      RooFit::Optimize(1),
      RooFit::PrintLevel(_printLevel), RooFit::Save(),
      _multiProcess > 0 ? RooFit::NumCPU(_multiProcess) : RooCmdArg()
  );
  std::cout << batchTimer;
  EXPECT_NE(result, nullptr);
  if (!result)
    return nullptr;

  EXPECT_EQ(result->status(), 0) << "[Batch fit did not converge.]";

  makePlots(::testing::UnitTest::GetInstance()->current_test_info()->name()+std::string("_batch_postfit"));

  return std::unique_ptr<RooFitResult>(result);
}

std::unique_ptr<RooFitResult> PDFTest::runScalarFit(RooAbsPdf* pdf) {
  if (!_dataFit)
    makeFitData();

  kickParameters();
  makePlots(::testing::UnitTest::GetInstance()->current_test_info()->name()+std::string("_scalar_prefit"));

  auto pars = pdf->getParameters(*_dataFit);
  *pars = _parameters;

  for (unsigned int index = 0; index < pars->size(); ++index) {
    auto pdfParameter = static_cast<RooAbsReal*>((*pars)[index]);
    auto origParameter = static_cast<RooAbsReal*>(_origParameters.find(*pdfParameter));
    if (!origParameter || origParameter->isConstant())
      continue;

    EXPECT_NE(pdfParameter->getVal(), origParameter->getVal())
        << "Parameter #" << index << "=" << pdfParameter->GetName() << " is identical after kicking.";
  }

  if (HasFailure()) {
    std::cout << "Pre-fit parameters:\n";
    _parameters.Print("V");
    std::cout << "Orig parameters:\n";
    _origParameters.Print("V");
  }

  MyTimer singleTimer("Fitting scalar mode " + _name);
  auto result = pdf->fitTo(*_dataFit,
      RooFit::BatchMode(false),
      RooFit::SumW2Error(false),
      RooFit::PrintLevel(_printLevel), RooFit::Save(),
      _multiProcess > 0 ? RooFit::NumCPU(_multiProcess) : RooCmdArg()
  );
  std::cout << singleTimer;
  EXPECT_NE(result, nullptr);
  if (!result)
    return nullptr;


  EXPECT_EQ(result->status(), 0) << "[Scalar fit did not converge.]";

  makePlots(::testing::UnitTest::GetInstance()->current_test_info()->name()+std::string("_scalar_postfit"));

  return std::unique_ptr<RooFitResult>(result);
}


void PDFTestWeightedData::makeFitData() {
  PDFTest::makeFitData();
  RooRealVar var("gausWeight", "gausWeight", 0, 10);
  RooConstVar mean("meanWeight", "", 1.);
  RooConstVar sigma("sigmaWeight", "", 0.2);
  RooGaussian gausDistr("gausDistr", "gausDistr", var, mean, sigma);
  std::unique_ptr<RooDataSet> gaussData(gausDistr.generate(RooArgSet(var), _dataFit->numEntries()));
  _dataFit->merge(gaussData.get());

  auto wdata = new RooDataSet(_dataFit->GetName(), _dataFit->GetTitle(), *_dataFit->get(),
      RooFit::Import(*_dataFit), RooFit::WeightVar("gausWeight"));

  _dataFit.reset(wdata);
}

