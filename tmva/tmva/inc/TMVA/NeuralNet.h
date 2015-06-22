#ifndef TMVA_NEURAL_NET
#define TMVA_NEURAL_NET
#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <tuple>
#include <math.h>
#include <cassert>
#include <random>
#include <thread>
#include <future>

#include "Pattern.h"
#include "Monitoring.h"

#include "TApplication.h"
#include "Timer.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TStyle.h"

#include <fenv.h> // turn on or off exceptions for NaN and other numeric exceptions



namespace TMVA
{

namespace NN
{

//    double gaussDouble (double mean, double sigma);








enum class EnumFunction
{
    ZERO = '0',
    LINEAR = 'L',
    TANH = 'T',
    RELU = 'R',
    SYMMRELU = 'r',
    TANHSHIFT = 't',
    SIGMOID = 's',
    SOFTSIGN = 'S',
    GAUSS = 'G',
    GAUSSCOMPLEMENT = 'C',
    DOUBLEINVERTEDGAUSS = 'D'
};



enum class EnumRegularization
{
    NONE, L1, L2, L1MAX
};


enum class ModeOutputValues
{
    DIRECT = 'd',
    SIGMOID = 's',
    SOFTMAX = 'S'
};


class Net;







typedef std::vector<char> DropContainer;


class Batch 
{
public:
    typedef typename std::vector<Pattern>::const_iterator const_iterator;

    Batch (typename std::vector<Pattern>::const_iterator itBegin, typename std::vector<Pattern>::const_iterator itEnd)
	: m_itBegin (itBegin)
	, m_itEnd (itEnd)
    {}

    const_iterator begin () const { return m_itBegin; }
    const_iterator end   () const { return m_itEnd; }

private:
    const_iterator m_itBegin;
    const_iterator m_itEnd;
};






template <typename ItSource, typename ItWeight, typename ItTarget>
    void applyWeights (ItSource itSourceBegin, ItSource itSourceEnd, ItWeight itWeight, ItTarget itTargetBegin, ItTarget itTargetEnd);



template <typename ItSource, typename ItWeight, typename ItPrev>
    void applyWeightsBackwards (ItSource itCurrBegin, ItSource itCurrEnd, ItWeight itWeight, ItPrev itPrevBegin, ItPrev itPrevEnd);





template <typename ItValue, typename ItFunction>
    void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction);


template <typename ItValue, typename ItFunction, typename ItInverseFunction, typename ItGradient>
    void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction, ItInverseFunction itInverseFunction, ItGradient itGradient);



template <typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient);



template <EnumRegularization Regularization, typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient, typename ItWeight>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient, 
	     ItWeight itWeight, double weightDecay);



// ----- signature of a minimizer -------------
// class Minimizer
// {
// public:

//     template <typename Function, typename Variables, typename PassThrough>
//     double operator() (Function& fnc, Variables& vars, PassThrough& passThrough) 
//     {
//         // auto itVars = begin (vars);
//         // auto itVarsEnd = end (vars);

//         std::vector<double> myweights;
//         std::vector<double> gradients;

//         double value = fnc (passThrough, myweights);
//         value = fnc (passThrough, myweights, gradients);
//         return value;
//     } 
// };



enum MinimizerType
{
    fSteepest
};





class Steepest
{
public:

    size_t m_repetitions;

    Steepest (double learningRate = 1e-4, 
              double momentum = 0.5, 
              size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_alpha (learningRate)
        , m_beta (momentum)
    {}

    template <typename Function, typename Weights, typename PassThrough>
        double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough);


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;
};









// test multithreaded training
class SteepestThreaded
{
public:

    size_t m_repetitions;

    SteepestThreaded (double learningRate = 1e-4, double learningRatePrev = 1e-4, size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_alpha (learningRate)
        , m_beta (learningRatePrev)
    {}


    template <typename Function, typename Weights, typename Gradients, typename PassThrough>
        double fitWrapper (Function& function, PassThrough& passThrough, Weights weights);


    template <typename Function, typename Weights, typename PassThrough>
        double operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough);


    double m_alpha;
    double m_beta;
    std::vector<double> m_prevGradients;
};









// walk along the maximum gradient
class MaxGradWeight
{
public:

    size_t m_repetitions;

    MaxGradWeight (double learningRate = 1e-4, size_t repetitions = 10) 
	: m_repetitions (repetitions)
        , m_learningRate (learningRate)
    {}



    template <typename Function, typename Weights, typename PassThrough>
        double operator() (Function& fitnessFunction, const Weights& weights, PassThrough& passThrough);

private:
    double m_learningRate;
};








template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
    double sumOfSquares (ItOutput itOutputBegin, ItOutput itOutputEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight);



template <typename ItProbability, typename ItTruth, typename ItDelta, typename ItInvActFnc>
    double crossEntropy (ItProbability itProbabilityBegin, ItProbability itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight);




template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
    double softMaxCrossEntropy (ItOutput itProbabilityBegin, ItOutput itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight);





template <typename ItWeight>
    double weightDecay (double error, ItWeight itWeight, ItWeight itWeightEnd, double factorWeightDecay, EnumRegularization eRegularization);














// the actual data for the layer (not the layout)
class LayerData
{
public:
    typedef std::vector<double> container_type;

    typedef typename container_type::iterator iterator_type;
    typedef typename container_type::const_iterator const_iterator_type;

    typedef std::vector<std::function<double(double)> > function_container_type;
    typedef typename function_container_type::iterator function_iterator_type;
    typedef typename function_container_type::const_iterator const_function_iterator_type;

    LayerData (const_iterator_type itInputBegin, const_iterator_type itInputEnd, ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);


    ~LayerData ()    {}


    LayerData (size_t size, 
	       const_iterator_type itWeightBegin, 
	       iterator_type itGradientBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       const_function_iterator_type itInverseFunctionBegin,
	       ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);

    LayerData (size_t size, const_iterator_type itWeightBegin, 
	       const_function_iterator_type itFunctionBegin, 
	       ModeOutputValues eModeOutput = ModeOutputValues::DIRECT);

    LayerData (const LayerData& other)
    : m_size (other.m_size)
    , m_itInputBegin (other.m_itInputBegin)
    , m_itInputEnd (other.m_itInputEnd)
    , m_deltas (other.m_deltas)
    , m_valueGradients (other.m_valueGradients)
    , m_values (other.m_values)
    , m_itConstWeightBegin   (other.m_itConstWeightBegin)
    , m_itGradientBegin (other.m_itGradientBegin)
    , m_itFunctionBegin (other.m_itFunctionBegin)
    , m_itInverseFunctionBegin (other.m_itInverseFunctionBegin)
    , m_isInputLayer (other.m_isInputLayer)
    , m_hasWeights (other.m_hasWeights)
    , m_hasGradients (other.m_hasGradients)
    , m_eModeOutput (other.m_eModeOutput) 
    {}

    LayerData (LayerData&& other)
    : m_size (other.m_size)
    , m_itInputBegin (other.m_itInputBegin)
    , m_itInputEnd (other.m_itInputEnd)
    , m_deltas (other.m_deltas)
    , m_valueGradients (other.m_valueGradients)
    , m_values (other.m_values)
    , m_itConstWeightBegin   (other.m_itConstWeightBegin)
    , m_itGradientBegin (other.m_itGradientBegin)
    , m_itFunctionBegin (other.m_itFunctionBegin)
    , m_itInverseFunctionBegin (other.m_itInverseFunctionBegin)
    , m_isInputLayer (other.m_isInputLayer)
    , m_hasWeights (other.m_hasWeights)
    , m_hasGradients (other.m_hasGradients)
    , m_eModeOutput (other.m_eModeOutput) 
    {}


    void setInput (const_iterator_type itInputBegin, const_iterator_type itInputEnd)
    {
        m_isInputLayer = true;
        m_itInputBegin = itInputBegin;
        m_itInputEnd = itInputEnd;
    }

    void clear ()
    {
        m_values.assign (m_values.size (), 0.0);
        m_deltas.assign (m_deltas.size (), 0.0);
    }

    const_iterator_type valuesBegin () const { return m_isInputLayer ? m_itInputBegin : begin (m_values); }
    const_iterator_type valuesEnd   () const { return m_isInputLayer ? m_itInputEnd   : end (m_values); }
    
    iterator_type valuesBegin () { assert (!m_isInputLayer); return begin (m_values); }
    iterator_type valuesEnd   () { assert (!m_isInputLayer); return end (m_values); }

    ModeOutputValues outputMode () const { return m_eModeOutput; }
    container_type probabilities () { return computeProbabilities (); }

    iterator_type deltasBegin () { return begin (m_deltas); }
    iterator_type deltasEnd   () { return end   (m_deltas); }

    const_iterator_type deltasBegin () const { return begin (m_deltas); }
    const_iterator_type deltasEnd   () const { return end   (m_deltas); }

    iterator_type valueGradientsBegin () { return begin (m_valueGradients); }
    iterator_type valueGradientsEnd   () { return end   (m_valueGradients); }

    const_iterator_type valueGradientsBegin () const { return begin (m_valueGradients); }
    const_iterator_type valueGradientsEnd   () const { return end   (m_valueGradients); }

    iterator_type gradientsBegin () { assert (m_hasGradients); return m_itGradientBegin; }
    const_iterator_type gradientsBegin () const { assert (m_hasGradients); return m_itGradientBegin; }
    const_iterator_type weightsBegin   () const { assert (m_hasWeights); return m_itConstWeightBegin; }

    const_function_iterator_type functionBegin () const { return m_itFunctionBegin; }
    const_function_iterator_type inverseFunctionBegin () const { return m_itInverseFunctionBegin; }

    size_t size () const { return m_size; }

private:

    container_type computeProbabilities ();

private:
    
    size_t m_size;

    const_iterator_type m_itInputBegin;
    const_iterator_type m_itInputEnd;

    std::vector<double> m_deltas;
    std::vector<double> m_valueGradients;
    std::vector<double> m_values;

    const_iterator_type m_itConstWeightBegin;
    iterator_type       m_itGradientBegin;

    const_function_iterator_type m_itFunctionBegin;

    const_function_iterator_type m_itInverseFunctionBegin;

    bool m_isInputLayer;
    bool m_hasWeights;
    bool m_hasGradients;

    ModeOutputValues m_eModeOutput;

};





// defines the layout of a layer
class Layer
{
public:

    Layer (size_t numNodes, EnumFunction activationFunction, ModeOutputValues eModeOutputValues = ModeOutputValues::DIRECT);

    ModeOutputValues modeOutputValues () const { return m_eModeOutputValues; }
    EnumFunction activationFunction () const { return m_activationFunction; }

    void modeOutputValues (ModeOutputValues eModeOutputValues) { m_eModeOutputValues = eModeOutputValues; }

    size_t numNodes () const { return m_numNodes; }
    size_t numWeights (size_t numInputNodes) const { return numInputNodes * numNodes (); } // fully connected

    const std::vector<std::function<double(double)> >& activationFunctions  () const { return m_vecActivationFunctions; }
    const std::vector<std::function<double(double)> >& inverseActivationFunctions  () const { return m_vecInverseActivationFunctions; }


private:


    std::vector<std::function<double(double)> > m_vecActivationFunctions;
    std::vector<std::function<double(double)> > m_vecInverseActivationFunctions;

    EnumFunction m_activationFunction;

    size_t m_numNodes;

    ModeOutputValues m_eModeOutputValues;

    friend class Net;
};





template <typename LAYERDATA>
    void forward (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData);

template <typename LAYERDATA>
    void forward_training (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData);


template <typename LAYERDATA>
    void backward (LAYERDATA& prevLayerData, LAYERDATA& currLayerData);


template <typename LAYERDATA>
    void update (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData, double weightDecay, EnumRegularization regularization);



class Settings
{
public:

    Settings (TString name,
              size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
	      double _factorWeightDecay = 1e-5, TMVA::NN::EnumRegularization _regularization = TMVA::NN::EnumRegularization::NONE,
              MinimizerType _eMinimizerType = MinimizerType::fSteepest, 
              double _learningRate = 1e-5, double _momentum = 0.3, int _repetitions = 3);
    
    virtual ~Settings ();


    template <typename Iterator>
        void setDropOut (Iterator begin, Iterator end, size_t _dropRepetitions) { m_dropOut.assign (begin, end); m_dropRepetitions = _dropRepetitions; }

    size_t dropRepetitions () const { return m_dropRepetitions; }
    const std::vector<double>& dropFractions () const { return m_dropOut; }

    
    void setMonitoring (std::shared_ptr<Monitoring> ptrMonitoring) { fMonitoring = ptrMonitoring; }

    size_t convergenceSteps () const { return m_convergenceSteps; }
    size_t batchSize () const { return m_batchSize; }
    size_t testRepetitions () const { return m_testRepetitions; }
    double factorWeightDecay () const { return m_factorWeightDecay; }

    double learningRate () const { return fLearningRate; }
    double momentum () const { return fMomentum; }
    int repetitions () const { return fRepetitions; }
    MinimizerType minimizerType () const { return fMinimizerType; }





    virtual void testSample (double /*error*/, double /*output*/, double /*target*/, double /*weight*/) {}
    virtual void startTrainCycle () {}
    virtual void endTrainCycle (double /*error*/) {}

    virtual void setProgressLimits (double minProgress = 0, double maxProgress = 100) 
    { 
        m_minProgress = minProgress;
        m_maxProgress = maxProgress; 
    }
    virtual void startTraining () 
    {
        m_timer.DrawProgressBar (Int_t(m_minProgress));
    }
    virtual void cycle (double progress, TString text) 
    {
        m_timer.DrawProgressBar (Int_t(m_minProgress+(m_maxProgress-m_minProgress)*(progress/100.0)), text);
    }

    virtual void startTestCycle () {}
    virtual void endTestCycle () {}
    virtual void testIteration () {}
    virtual void drawSample (const std::vector<double>& /*input*/, const std::vector<double>& /* output */, const std::vector<double>& /* target */, double /* patternWeight */) {}

    virtual void computeResult (const Net& /* net */, std::vector<double>& /* weights */) {}


    EnumRegularization regularization () const { return m_regularization; }


    void pads (int numPads) { if (fMonitoring) fMonitoring->pads (numPads); }
    void create (std::string histoName, int bins, double min, double max) { if (fMonitoring) fMonitoring->create (histoName, bins, min, max); }
    void create (std::string histoName, int bins, double min, double max, int bins2, double min2, double max2) { if (fMonitoring) fMonitoring->create (histoName, bins, min, max, bins2, min2, max2); }
    void addPoint (std::string histoName, double x) { if (fMonitoring) fMonitoring->addPoint (histoName, x); }
    void addPoint (std::string histoName, double x, double y) {if (fMonitoring) fMonitoring->addPoint (histoName, x, y); }
    void plot (std::string histoName, std::string options, int pad, EColor color) { if (fMonitoring) fMonitoring->plot (histoName, options, pad, color); }
    void clear (std::string histoName) { if (fMonitoring) fMonitoring->clear (histoName); }
    bool exists (std::string histoName) { if (fMonitoring) return fMonitoring->exists (histoName); return false; }

public:
    Timer  m_timer;
    double m_minProgress;
    double m_maxProgress;


    size_t m_convergenceSteps;
    size_t m_batchSize;
    size_t m_testRepetitions;
    double m_factorWeightDecay;

    size_t count_E;
    size_t count_dE;
    size_t count_mb_E;
    size_t count_mb_dE;

    EnumRegularization m_regularization;

    double m_dropRepetitions;
    std::vector<double> m_dropOut;

    double fLearningRate;
    double fMomentum;
    int fRepetitions;
    MinimizerType fMinimizerType;

protected:
    std::shared_ptr<Monitoring> fMonitoring;

};























// enthaelt additional zu den settings die plot-kommandos fuer die graphischen
// ausgaben. 
class ClassificationSettings : public Settings
{
public:
    ClassificationSettings (TString name,
                            size_t _convergenceSteps = 15, size_t _batchSize = 10, size_t _testRepetitions = 7, 
			    double _factorWeightDecay = 1e-5, EnumRegularization _regularization = EnumRegularization::NONE, 
			    size_t _scaleToNumEvents = 0, MinimizerType _eMinimizerType = MinimizerType::fSteepest, 
                            double _learningRate = 1e-5, double _momentum = 0.3, int _repetitions = 3)
        : Settings (name, _convergenceSteps, _batchSize, _testRepetitions, _factorWeightDecay, 
                    _regularization, _eMinimizerType, _learningRate, _momentum, _repetitions)
        , m_ams ()
        , m_sumOfSigWeights (0)
        , m_sumOfBkgWeights (0)
	, m_scaleToNumEvents (_scaleToNumEvents)
	, m_cutValue (10.0)
	, m_pResultPatternContainer (NULL)
	, m_fileNameResult ()
	, m_fileNameNetConfig ()
    {
    }

    virtual ~ClassificationSettings () 
    {
    }

    void startTrainCycle ();
    void endTrainCycle (double /*error*/);
    void testIteration () { if (fMonitoring) fMonitoring->ProcessEvents (); }


    /* void createHistograms () */
    /* { */
    /*     std::cout << "is hist ROC existing?" << std::endl; */
    /*     if (m_histROC) */
    /*     { */
    /*         std::cout << "--> yes" << std::endl; */
    /*         fMonitoring->ProcessEvents (); */
    /*         return; */
    /*     } */

    /*     std::cout << "create histograms" << std::endl; */
    /*     TCanvas* canvas = fMonitoring->GetCanvas (); */
    /*     if (canvas) */
    /*     { */
    /*         std::cout << "canvas divide" << std::endl; */
    /*         canvas->cd (); */
    /*         canvas->Divide (2,2); */
    /*     } */
    /*     if (!m_histROC)  */
    /*     {  */
    /*         m_histROC = new TH2F ("ROC","ROC", 1000, 0, 1.0, 1000, 0, 1.0); m_histROC->SetDirectory (0);  */
    /*         m_histROC->SetLineColor (kBlue); */
    /*     } */
    /*     if (!m_histSignificance)  */
    /*     {  */
    /*         m_histSignificance = new TH2F ("Significance", "Significance", 1000, 0,1.0, 5, 0.0, 2.0);  */
    /*         m_histSignificance->SetDirectory (0);  */
    /*         m_histSignificance->SetBit (TH1::kCanRebin);  */
    /*         m_histROC->SetLineColor (kRed); */
    /*     } */
    /*     if (!m_histError)  */
    /*     {  */
    /*         m_histError = new TH1F ("Error", "Error", 100, 0, 100);  */
    /*         m_histError->SetDirectory (0);  */
    /*         m_histError->SetBit (TH1::kCanRebin); */
    /*         m_histROC->SetLineColor (kGreen); */
    /*     } */
    /*     if (!m_histOutputSignal)  */
    /*     {  */
    /*         m_histOutputSignal = new TH1F ("Signal", "Signal", 100, 0, 1.0);   */
    /*         m_histOutputSignal->SetDirectory (0);  */
    /*         m_histOutputSignal->SetBit (TH1::kCanRebin); */
    /*     } */
    /*     if (!m_histOutputBackground)  */
    /*     {  */
    /*         m_histOutputBackground = new TH1F ("Background", "Background", 100, 0, 1.0);  */
    /*         m_histOutputBackground->SetDirectory (0);  */
    /*         m_histOutputBackground->SetBit (TH1::kCanRebin); */
    /*     } */

    /*     fMonitoring->ProcessEvents (); */
    /* } */

    void testSample (double error, double output, double target, double weight);

    virtual void startTestCycle ();
    virtual void endTestCycle ();


    void setWeightSums (double sumOfSigWeights, double sumOfBkgWeights);
    void setResultComputation (std::string _fileNameNetConfig, std::string _fileNameResult, std::vector<Pattern>* _resultPatternContainer);

    std::vector<double> m_input;
    std::vector<double> m_output;
    std::vector<double> m_targets;
    std::vector<double> m_weights;

    std::vector<double> m_ams;
    std::vector<double> m_significances;


    double m_sumOfSigWeights;
    double m_sumOfBkgWeights;
    size_t m_scaleToNumEvents;

    double m_cutValue;
    std::vector<Pattern>* m_pResultPatternContainer;
    std::string m_fileNameResult;
    std::string m_fileNameNetConfig;


    /* TH2F* m_histROC; */
    /* TH2F* m_histSignificance; */

    /* TH1F* m_histError; */
    /* TH1F* m_histOutputSignal; */
    /* TH1F* m_histOutputBackground; */
};








enum class ModeOutput
{
    FETCH
};


enum class ModeErrorFunction
{
    SUMOFSQUARES = 'S',
    CROSSENTROPY = 'C',
    CROSSENTROPY_MUTUALEXCLUSIVE = 'M'
};

enum class WeightInitializationStrategy
{
    XAVIER, TEST, LAYERSIZE, XAVIERUNIFORM
};



class Net
{
public:

    typedef std::vector<double> container_type;
    typedef container_type::iterator iterator_type;
    typedef std::pair<iterator_type,iterator_type> begin_end_type;


    Net () 
	: m_eErrorFunction (ModeErrorFunction::SUMOFSQUARES)
	, m_sizeInput (0)
    {
    }

    Net (const Net& other)
        : m_eErrorFunction (other.m_eErrorFunction)
        , m_sizeInput (other.m_sizeInput)
        , m_layers (other.m_layers)
    {
    }

    void setInputSize (size_t sizeInput) { m_sizeInput = sizeInput; }
    void addLayer (Layer& layer) { m_layers.push_back (layer); }
    void addLayer (Layer&& layer) { m_layers.push_back (layer); }
    void setErrorFunction (ModeErrorFunction eErrorFunction) { m_eErrorFunction = eErrorFunction; }
    
    size_t inputSize () const { return m_sizeInput; }

    template <typename WeightsType, typename DropProbabilities>
        void dropOutWeightFactor (WeightsType& weights,
                                  const DropProbabilities& drops, 
                                  bool inverse = false);

    template <typename Minimizer>
    double train (std::vector<double>& weights, 
		  std::vector<Pattern>& trainPattern, 
		  const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings);


    template <typename Iterator, typename Minimizer>
    inline double trainCycle (Minimizer& minimizer, std::vector<double>& weights, 
			      Iterator itPatternBegin, Iterator itPatternEnd, Settings& settings, DropContainer& dropContainer);

    size_t numWeights (size_t numInputNodes, size_t trainingStartLayer = 0) const;

    template <typename Weights>
        std::vector<double> compute (const std::vector<double>& input, const Weights& weights) const;

    template <typename Weights, typename PassThrough>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights) const;

    template <typename Weights, typename PassThrough, typename OutContainer>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights, ModeOutput eFetch, OutContainer& outputContainer) const;
    
    template <typename Weights, typename Gradients, typename PassThrough>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients) const;

    template <typename Weights, typename Gradients, typename PassThrough, typename OutContainer>
        double operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients, ModeOutput eFetch, OutContainer& outputContainer) const;




    template <typename LayerContainer, typename PassThrough, typename ItWeight, typename ItGradient, typename OutContainer>
    double forward_backward (LayerContainer& layers, PassThrough& settingsAndBatch, 
			     ItWeight itWeightBegin, 
			     ItGradient itGradientBegin, ItGradient itGradientEnd, 
			     size_t trainFromLayer, 
			     OutContainer& outputContainer, bool fetchOutput) const;


    
    double E ();
    void dE ();


    template <typename Container, typename ItWeight>
        double errorFunction (LayerData& layerData,
                              Container truth,
                              ItWeight itWeight,
                              ItWeight itWeightEnd,
                              double patternWeight,
                              double factorWeightDecay,
                              EnumRegularization eRegularization) const;


    const std::vector<Layer>& layers () const { return m_layers; }
    std::vector<Layer>& layers ()  { return m_layers; }



    void clear () 
    {
        m_layers.clear ();
	m_eErrorFunction = ModeErrorFunction::SUMOFSQUARES;
    }


    template <typename ItPat, typename OutIterator>
    void initializeWeights (WeightInitializationStrategy eInitStrategy, 
			    ItPat itPatternBegin, 
			    ItPat itPatternEnd, 
			    OutIterator itWeight);


private:

    std::vector<Layer> m_layers;
    ModeErrorFunction m_eErrorFunction;
    size_t m_sizeInput;

};











}; // namespace NN
}; // namespace TMVA


// include the implementations (in header file, because they are templated)
#include "NeuralNet_i.h"

#endif

