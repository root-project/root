#ifndef TMVA_NEURAL_NET_I
#define TMVA_NEURAL_NET_I
#pragma once

namespace TMVA
{
namespace NN
{




static std::function<double(double)> ZeroFnc = [](double /*value*/){ return 0; };


static std::function<double(double)> Sigmoid = [](double value){ value = std::max (-100.0, std::min (100.0,value)); return 1.0/(1.0 + std::exp (-value)); };
static std::function<double(double)> InvSigmoid = [](double value){ double s = Sigmoid (value); return s*(1.0-s); };

static std::function<double(double)> Tanh = [](double value){ return tanh (value); };
static std::function<double(double)> InvTanh = [](double value){ return 1.0 - std::pow (value, 2.0); };

static std::function<double(double)> Linear = [](double value){ return value; };
static std::function<double(double)>  InvLinear = [](double /*value*/){ return 1.0; };

static std::function<double(double)> SymmReLU = [](double value){ const double margin = 0.3; return value > margin ? value-margin : value < -margin ? value+margin : 0; };
static std::function<double(double)> InvSymmReLU = [](double value){ const double margin = 0.3; return value > margin ? 1.0 : value < -margin ? 1.0 : 0; };

static std::function<double(double)> ReLU = [](double value){ return value > 0 ? value : 0; };
static std::function<double(double)> InvReLU = [](double value){ return value > 0 ? 1.0 : 0; };

static std::function<double(double)> SoftPlus = [](double value){ return std::log (1.0+ std::exp (value)); };
static std::function<double(double)> InvSoftPlus = [](double value){ return 1.0 / (1.0 + std::exp (-value)); };

static std::function<double(double)> TanhShift = [](double value){ return tanh (value-0.3); };
static std::function<double(double)> InvTanhShift = [](double value){ return 0.3 + (1.0 - std::pow (value, 2.0)); };

static std::function<double(double)> SoftSign = [](double value){ return value / (1.0 + fabs (value)); };
static std::function<double(double)> InvSoftSign = [](double value){ return std::pow ((1.0 - fabs (value)),2.0); };

static std::function<double(double)> Gauss = [](double value){ const double s = 6.0; return exp (-std::pow(value*s,2.0)); };
static std::function<double(double)> InvGauss = [](double value){ const double s = 6.0; return -2.0 * value * s*s * Gauss (value); };

static std::function<double(double)> GaussComplement = [](double value){ const double s = 6.0; return 1.0 - exp (-std::pow(value*s,2.0));; };
static std::function<double(double)> InvGaussComplement = [](double value){ const double s = 6.0; return +2.0 * value * s*s * GaussComplement (value); };

static std::function<double(double)> DoubleInvertedGauss = [](double value)
{ const double s = 8.0; const double shift = 0.1; return exp (-std::pow((value-shift)*s,2.0)) - exp (-std::pow((value+shift)*s,2.0)); };
static std::function<double(double)> InvDoubleInvertedGauss = [](double value)
{ const double s = 8.0; const double shift = 0.1; return -2.0 * (value-shift) * s*s * DoubleInvertedGauss (value-shift) + 2.0 * (value+shift) * s*s * DoubleInvertedGauss (value+shift);  };




template <typename T>
T uniformFromTo (T from, T to)
{
    return from + (rand ()* (to - from)/RAND_MAX);
}



template <typename Container, typename T>
void uniform (Container& container, T maxValue)
{
    for (auto it = begin (container), itEnd = end (container); it != itEnd; ++it)
    {
//        (*it) = uniformFromTo (-1.0*maxValue, 1.0*maxValue);
        (*it) = TMVA::NN::uniformFromTo (-1.0*maxValue, 1.0*maxValue);
    }
}



// apply weights using drop-out
// itDrop correlates with itSource
template <typename ItSource, typename ItWeight, typename ItTarget, typename ItDrop>
    void applyWeights (ItSource itSourceBegin, ItSource itSourceEnd,
                       ItWeight itWeight,
                       ItTarget itTargetBegin, ItTarget itTargetEnd,
                       ItDrop itDrop)
{
    for (auto itSource = itSourceBegin; itSource != itSourceEnd; ++itSource)
    {
        for (auto itTarget = itTargetBegin; itTarget != itTargetEnd; ++itTarget)
        {
            if (*itDrop)
                (*itTarget) += (*itSource) * (*itWeight);
            ++itWeight;
        }
        ++itDrop;        
    }
}



// apply weights without drop-out
template <typename ItSource, typename ItWeight, typename ItTarget>
    void applyWeights (ItSource itSourceBegin, ItSource itSourceEnd,
                       ItWeight itWeight,
                       ItTarget itTargetBegin, ItTarget itTargetEnd)
{
    for (auto itSource = itSourceBegin; itSource != itSourceEnd; ++itSource)
    {
        for (auto itTarget = itTargetBegin; itTarget != itTargetEnd; ++itTarget)
        {
            (*itTarget) += (*itSource) * (*itWeight);
            ++itWeight;
        }
    }
}



// apply weights backwards (for backprop)
template <typename ItSource, typename ItWeight, typename ItPrev>
void applyWeightsBackwards (ItSource itCurrBegin, ItSource itCurrEnd,
                            ItWeight itWeight,
                            ItPrev itPrevBegin, ItPrev itPrevEnd)
{
    for (auto itPrev = itPrevBegin; itPrev != itPrevEnd; ++itPrev)
    {
	for (auto itCurr = itCurrBegin; itCurr != itCurrEnd; ++itCurr)
	{
            (*itPrev) += (*itCurr) * (*itWeight);
            ++itWeight;
        }
    }
}



// apply weights backwards (for backprop)
// itDrop correlates with itPrev (to be in agreement with "applyWeights" where it correlates with itSource (same node as itTarget here in applybackwards)
template <typename ItSource, typename ItWeight, typename ItPrev, typename ItDrop>
void applyWeightsBackwards (ItSource itCurrBegin, ItSource itCurrEnd,
                            ItWeight itWeight,
                            ItPrev itPrevBegin, ItPrev itPrevEnd,
                            ItDrop itDrop)
{
    for (auto itPrev = itPrevBegin; itPrev != itPrevEnd; ++itPrev)
    {
	for (auto itCurr = itCurrBegin; itCurr != itCurrEnd; ++itCurr)
	{
            if (*itDrop)
                (*itPrev) += (*itCurr) * (*itWeight);
            ++itWeight; 
        }
        ++itDrop;
    }
}





template <typename ItValue, typename ItFunction>
void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction)
{
    while (itValue != itValueEnd)
    {
        auto& value = (*itValue);
        value = (*itFunction) (value);

        ++itValue; ++itFunction;
    }
}


template <typename ItValue, typename ItFunction, typename ItInverseFunction, typename ItGradient>
void applyFunctions (ItValue itValue, ItValue itValueEnd, ItFunction itFunction, ItInverseFunction itInverseFunction, ItGradient itGradient)
{
    while (itValue != itValueEnd)
    {
        auto& value = (*itValue);
        value = (*itFunction) (value);
        (*itGradient) = (*itInverseFunction) (value);
        
        ++itValue; ++itFunction; ++itInverseFunction; ++itGradient;
    }
}



template <typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient)
{
    while (itSource != itSourceEnd)
    {
        auto itTargetDelta = itTargetDeltaBegin;
        auto itTargetGradient = itTargetGradientBegin;
        while (itTargetDelta != itTargetDeltaEnd)
        {
            (*itGradient) += - (*itTargetDelta) * (*itSource) * (*itTargetGradient);
            ++itTargetDelta; ++itTargetGradient; ++itGradient;
        }
        ++itSource; 
    }
}




template <EnumRegularization Regularization>
    inline double computeRegularization (double weight, const double& factorWeightDecay)
{
    return 0;
}

// L1 regularization
template <>
    inline double computeRegularization<EnumRegularization::L1> (double weight, const double& factorWeightDecay)
{
    return weight == 0.0 ? 0.0 : std::copysign (factorWeightDecay, weight);
}

// L2 regularization
template <>
    inline double computeRegularization<EnumRegularization::L2> (double weight, const double& factorWeightDecay)
{
    return factorWeightDecay * weight;
}


template <EnumRegularization Regularization, typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient, typename ItWeight>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient, 
	     ItWeight itWeight, double weightDecay)
{
    // ! the factor weightDecay has to be already scaled by 1/n where n is the number of weights
    while (itSource != itSourceEnd)
    {
        auto itTargetDelta = itTargetDeltaBegin;
        auto itTargetGradient = itTargetGradientBegin;
        while (itTargetDelta != itTargetDeltaEnd)
        {
	    (*itGradient) -= + (*itTargetDelta) * (*itSource) * (*itTargetGradient) + computeRegularization<Regularization>(*itWeight,weightDecay);
            ++itTargetDelta; ++itTargetGradient; ++itGradient; ++itWeight;
        }
        ++itSource; 
    }
}






#define USELOCALWEIGHTS 1



    template <typename Function, typename Weights, typename PassThrough>
        double Steepest::operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<double> gradients (numWeights, 0.0);

#ifdef USELOCALWEIGHTS
	std::vector<double> localWeights (begin (weights), end (weights));
#endif
        double E = 1e10;
        if (m_prevGradients.size () != numWeights)
        {
            m_prevGradients.clear ();
            m_prevGradients.assign (weights.size (), 0);
        }

        bool success = true;
        size_t currentRepetition = 0;
        while (success)
        {
            if (currentRepetition >= m_repetitions)
                break;

            gradients.assign (numWeights, 0.0);
#ifdef USELOCALWEIGHTS
            E = fitnessFunction (passThrough, localWeights, gradients);
#else            
            E = fitnessFunction (passThrough, weights, gradients);
#endif         
//            plotGradients (gradients);

            double alpha = gaussDouble (m_alpha, m_alpha/2.0);
//            double alpha = m_alpha;

            auto itG = begin (gradients);
            auto itGEnd = end (gradients);
            auto itPrevG = begin (m_prevGradients);
            double maxGrad = 0.0;
            for (; itG != itGEnd; ++itG, ++itPrevG)
            {
                double currGrad = (*itG);
                double prevGrad = (*itPrevG);
                currGrad *= alpha;
                
                (*itPrevG) = m_beta * (prevGrad + currGrad);
                (*itG) = currGrad + prevGrad;

                if (std::fabs (currGrad) > maxGrad)
                    maxGrad = currGrad;
            }

            if (maxGrad > 1)
            {
                m_alpha /= 2;
                std::cout << "learning rate reduced to " << m_alpha << std::endl;
                std::for_each (weights.begin (), weights.end (), [maxGrad](double& w)
                               {
                                   w /= maxGrad;
                               });
                m_prevGradients.clear ();
            }
            else
            {
                auto itW = std::begin (weights);
                std::for_each (std::begin (gradients), std::end (gradients), [&itW](double& g)
                               {
                                   *itW += g;
                                   ++itW;
                               });
//                std::copy (std::begin (localWeights), std::end (localWeights), std::begin (weights));
                /* for (auto itL = std::begin (localWeights), itLEnd = std::end (localWeights), itW = std::begin (weights), itG = std::begin (gradients); */
                /*      itL != itLEnd; ++itL, ++itW, ++itG) */
                /* { */
                /*     if (*itG > maxGrad/2.0) */
                /*         *itW = *itL; */
                /* } */
            }

            ++currentRepetition;
        }
        return E;
    }













    /* template <typename Function, typename Weights, typename PassThrough> */
    /*     double MaxGradWeight::operator() (Function& fitnessFunction, const Weights& weights, PassThrough& passThrough)  */
    /* { */
    /*     double alpha = m_learningRate; */

    /*     size_t numWeights = weights.size (); */
    /*     std::vector<double> gradients (numWeights, 0.0); */
    /*     std::vector<double> localWeights (begin (weights), end (weights)); */


    /*     double Ebase = fitnessFunction (passThrough, weights, gradients); */
    /*     double Emin = Ebase; */

    /*     bool success = true; */
    /*     size_t currentRepetition = 0; */
    /*     while (success) */
    /*     { */
    /*         if (currentRepetition >= m_repetitions) */
    /*             break; */

    /*         auto itMaxGradElement = std::max_element (begin (gradients), end (gradients)); */
    /*         auto idx = std::distance (begin (gradients), itMaxGradElement); */
    /*         localWeights.at (idx) += alpha*(*itMaxGradElement); */
    /*         gradients.assign (numWeights, 0.0); */
    /*         double E = fitnessFunction (passThrough, localWeights, gradients); */

    /*         if (E < Emin) */
    /*         { */
    /*             Emin = E; */

    /*             auto itLocW = begin (localWeights); */
    /*             auto itLocWEnd = end (localWeights); */
    /*             auto itW = begin (weights); */
    /*             for (; itLocW != itLocWEnd; ++itLocW, ++itW) */
    /*             { */
    /*                 (*itW) = (*itLocW); */
    /*             } */
    /*         } */
    /*         ++currentRepetition; */
    /*     } */
    /*     return Emin; */
    /* } */









template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double sumOfSquares (ItOutput itOutputBegin, ItOutput itOutputEnd, ItTruth itTruthBegin, ItTruth /*itTruthEnd*/, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight) 
{
    double errorSum = 0.0;

    // output - truth
    ItTruth itTruth = itTruthBegin;
    bool hasDeltas = (itDelta != itDeltaEnd);
    for (ItOutput itOutput = itOutputBegin; itOutput != itOutputEnd; ++itOutput, ++itTruth)
    {
//	assert (itTruth != itTruthEnd);
	double output = (*itOutput);
	double error = output - (*itTruth);
	if (hasDeltas)
	{
	    (*itDelta) = (*itInvActFnc)(output) * error * patternWeight;
	    ++itDelta; ++itInvActFnc;
	}
	errorSum += error*error  * patternWeight;
    }

    return 0.5*errorSum;
}



template <typename ItProbability, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double crossEntropy (ItProbability itProbabilityBegin, ItProbability itProbabilityEnd, ItTruth itTruthBegin, ItTruth /*itTruthEnd*/, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc /*itInvActFnc*/, double patternWeight) 
{
    bool hasDeltas = (itDelta != itDeltaEnd);
    
    double errorSum = 0.0;
    for (ItProbability itProbability = itProbabilityBegin; itProbability != itProbabilityEnd; ++itProbability)
    {
        double probability = *itProbability;
        double truth = *itTruthBegin;
        truth = truth < 0.1 ? 0.1 : truth;
        truth = truth > 0.9 ? 0.9 : truth;
        if (hasDeltas)
        {
            double delta = probability - truth;
	    (*itDelta) = delta*patternWeight;
//	    (*itDelta) = (*itInvActFnc)(probability) * delta * patternWeight;
            ++itDelta;
        }
        double error (0);
        if (probability == 0) // protection against log (0)
        {
            if (truth >= 0.5)
                error += 1.0;
        }
        else if (probability == 1)
        {
            if (truth < 0.5)
                error += 1.0;
        }
        else
            error += - (truth * log (probability) + (1.0-truth) * log (1.0-probability)); // cross entropy function
        errorSum += error * patternWeight;
        
    }
    return errorSum;
}




template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double softMaxCrossEntropy (ItOutput itProbabilityBegin, ItOutput itProbabilityEnd, ItTruth itTruthBegin, ItTruth /*itTruthEnd*/, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc /*itInvActFnc*/, double patternWeight) 
{
    double errorSum = 0.0;

    bool hasDeltas = (itDelta != itDeltaEnd);
    // output - truth
    ItTruth itTruth = itTruthBegin;
    for (auto itProbability = itProbabilityBegin; itProbability != itProbabilityEnd; ++itProbability, ++itTruth)
    {
//	assert (itTruth != itTruthEnd);
	double probability = (*itProbability);
	double truth = (*itTruth);
	if (hasDeltas)
	{
            (*itDelta) = probability - truth;
//	    (*itDelta) = (*itInvActFnc)(sm) * delta * patternWeight;
	    ++itDelta; //++itInvActFnc;
	}
        double error (0);

	error += truth * log (probability);
	errorSum += error;
    }

    return -errorSum * patternWeight;
}









template <typename ItWeight>
    double weightDecay (double error, ItWeight itWeight, ItWeight itWeightEnd, double factorWeightDecay, EnumRegularization eRegularization)
{
    if (eRegularization == EnumRegularization::L1)
    {
        // weight decay (regularization)
        double w = 0;
        size_t n = 0;
        for (; itWeight != itWeightEnd; ++itWeight, ++n)
        {
            double weight = (*itWeight);
            w += std::fabs (weight);
        }
        return error + 0.5 * w * factorWeightDecay / n;
    }
    else if (eRegularization == EnumRegularization::L2)
    {
        // weight decay (regularization)
        double w = 0;
        size_t n = 0;
        for (; itWeight != itWeightEnd; ++itWeight, ++n)
        {
            double weight = (*itWeight);
            w += weight*weight;
        }
        return error + 0.5 * w * factorWeightDecay / n;
    }
    else
        return error;
}














template <typename LAYERDATA>
void forward (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    if (prevLayerData.hasDropOut ())
    {        
        applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
                      currLayerData.weightsBegin (), 
                      currLayerData.valuesBegin (), currLayerData.valuesEnd (),
                      prevLayerData.dropOut ());
    }
    else
    {
        applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
                      currLayerData.weightsBegin (), 
                      currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    }
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin ());
}

template <typename LAYERDATA>
void forward_training (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    if (prevLayerData.hasDropOut ())
    {        
        applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
                      currLayerData.weightsBegin (), 
                      currLayerData.valuesBegin (), currLayerData.valuesEnd (),
                      prevLayerData.dropOut ());
    }
    else
    {
        applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
                      currLayerData.weightsBegin (), 
                      currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    }
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin (), 
		    currLayerData.inverseFunctionBegin (), currLayerData.valueGradientsBegin ());
}


template <typename LAYERDATA>
void backward (LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    if (prevLayerData.hasDropOut ())
    {
        applyWeightsBackwards (currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
                               currLayerData.weightsBegin (), 
                               prevLayerData.deltasBegin (), prevLayerData.deltasEnd (),
                               prevLayerData.dropOut ());
    }
    else
    {
        applyWeightsBackwards (currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
                               currLayerData.weightsBegin (), 
                               prevLayerData.deltasBegin (), prevLayerData.deltasEnd ());
    }
}



template <typename LAYERDATA>
void update (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData, double factorWeightDecay, EnumRegularization regularization)
{
    // ! the "factorWeightDecay" has already to be scaled by 1/n where n is the number of weights
    if (factorWeightDecay != 0.0) // has weight regularization
	if (regularization == EnumRegularization::L1)  // L1 regularization ( sum(|w|) )
	{
	    update<EnumRegularization::L1> (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
			  currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			  currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin (), 
			  currLayerData.weightsBegin (), factorWeightDecay);
	}
	else if (regularization == EnumRegularization::L2) // L2 regularization ( sum(w^2) )
	{
	    update<EnumRegularization::L2> (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
			   currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			   currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin (), 
			   currLayerData.weightsBegin (), factorWeightDecay);
	}
	else 
	{
            update (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
                    currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
                    currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin ());
	}
    
    else
    { // no weight regularization
	update (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
		currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin ());
    }
}












    template <typename WeightsType, typename DropProbabilities>
        void Net::dropOutWeightFactor (WeightsType& weights,
                                       const DropProbabilities& drops, 
                                       bool inverse)
    {
	if (drops.empty () || weights.empty ())
	    return;

        auto itWeight = std::begin (weights);
        auto itWeightEnd = std::end (weights);
        auto itDrop = std::begin (drops);
        auto itDropEnd = std::end (drops);
	size_t numNodesPrev = inputSize ();
        double dropFractionPrev = *itDrop;
	++itDrop;

        for (auto& layer : layers ())
        {
            if (itDrop == itDropEnd)
                break;

	    size_t numNodes = layer.numNodes ();

            double dropFraction = *itDrop;
            double pPrev = 1.0 - dropFractionPrev;
            double p = 1.0 - dropFraction;
	    p *= pPrev;

            if (inverse)
            {
                p = 1.0/p;
            }
	    size_t _numWeights = layer.numWeights (numNodesPrev);
            for (size_t iWeight = 0; iWeight < _numWeights; ++iWeight)
            {
                if (itWeight == itWeightEnd)
                    break;
                
                *itWeight *= p;
                ++itWeight;
            }
	    numNodesPrev = numNodes;
	    dropFractionPrev = dropFraction;
	    ++itDrop;
        }
    }



        
    

    template <typename Minimizer>
        double Net::train (std::vector<double>& weights, 
		  std::vector<Pattern>& trainPattern, 
		  const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings)
    {
//        std::cout << "START TRAINING" << std::endl;
        settings.startTrainCycle ();

        settings.pads (4);
        settings.create ("trainErrors", 100, 0, 100, 100, 0,1);
        settings.create ("testErrors", 100, 0, 100, 100, 0,1);

        size_t cycleCount = 0;
        size_t testCycleCount = 0;
        double testError = 1e20;
        double trainError = 1e20;
        size_t dropOutChangeCount = 0;

	DropContainer dropContainer;
	DropContainer dropContainerTest;
        const std::vector<double>& dropFractions = settings.dropFractions ();
        bool isWeightsForDrop = false;

        
        // until convergence
        do
        {
            ++cycleCount;

	    // if dropOut enabled
            size_t dropIndex = 0;
            if (!dropFractions.empty () && dropOutChangeCount % settings.dropRepetitions () == 0)
	    {
		// fill the dropOut-container
		dropContainer.clear ();
                size_t numNodes = inputSize ();
                double dropFraction = 0.0;
                dropFraction = dropFractions.at (dropIndex);
                ++dropIndex;
                fillDropContainer (dropContainer, dropFraction, numNodes);
		for (auto itLayer = begin (m_layers), itLayerEnd = end (m_layers); itLayer != itLayerEnd; ++itLayer, ++dropIndex)
		{
		    auto& layer = *itLayer;
                    numNodes = layer.numNodes ();
		    // how many nodes have to be dropped
                    dropFraction = 0.0;
                    if (dropFractions.size () > dropIndex)
                        dropFraction = dropFractions.at (dropIndex);
                    
                    fillDropContainer (dropContainer, dropFraction, numNodes);
		}
                isWeightsForDrop = true;
	    }

	    // execute training cycle
            trainError = trainCycle (minimizer, weights, begin (trainPattern), end (trainPattern), settings, dropContainer);
	    

	    // check if we execute a test
            bool hasConverged = false;
            if (testCycleCount % settings.testRepetitions () == 0)
            {
                if (isWeightsForDrop)
                {
                    dropOutWeightFactor (weights, dropFractions);
                    isWeightsForDrop = false;
                }

                testError = 0;
                double weightSum = 0;
                settings.startTestCycle ();
                std::vector<double> output;
                for (auto it = begin (testPattern), itEnd = end (testPattern); it != itEnd; ++it)
                {
                    const Pattern& p = (*it);
                    double weight = p.weight ();
                    Batch batch (it, it+1);
                    output.clear ();
		    std::tuple<Settings&, Batch&, DropContainer&> passThrough (settings, batch, dropContainerTest);
                    double testPatternError = (*this) (passThrough, weights, ModeOutput::FETCH, output);
                    if (output.size () == 1)
		    {
                        settings.testSample (testPatternError, output.at (0), p.output ().at (0), weight);
		    }
                    weightSum += fabs (weight);
                    testError += testPatternError*weight;
                }
                settings.endTestCycle ();
                testError /= weightSum;

		settings.computeResult (*this, weights);

                hasConverged = settings.hasConverged (testError);
                if (!hasConverged && !isWeightsForDrop)
                {
                    dropOutWeightFactor (weights, dropFractions, true); // inverse
                    isWeightsForDrop = true;
                }
            }
            ++testCycleCount;
	    ++dropOutChangeCount;


            static double x = -1.0;
            x += 1.0;
//            settings.resetPlot ("errors");
            settings.addPoint ("trainErrors", cycleCount, trainError);
            settings.addPoint ("testErrors", cycleCount, testError);
            settings.plot ("trainErrors", "C", 1, kBlue);
            settings.plot ("testErrors", "C", 1, kMagenta);



            if (hasConverged)
                break;


            TString convText = Form( "<D^2> (train/test/epoch/conv/maxConv): %.4g/%.4g/%d/%d/%d",
                                     trainError,
                                     testError,
                                     (int)cycleCount,
                                     (int)settings.convergenceCount (),
                                     (int)settings.maxConvergenceCount ());
            double progress = 100*(double)settings.maxConvergenceCount () /(double)settings.convergenceSteps ();
            settings.cycle (progress, convText);
        }
	while (true);
        settings.endTrainCycle (trainError);
        
        TString convText = Form( "<D^2> (train/test/epoch): %.4g/%.4g/%d", trainError, testError, (int)cycleCount);
        double progress = 100*(double)settings.maxConvergenceCount() /(double)settings.convergenceSteps ();
        settings.cycle (progress, convText);

        return testError;
    }



    template <typename Iterator, typename Minimizer>
        inline double Net::trainCycle (Minimizer& minimizer, std::vector<double>& weights, 
			      Iterator itPatternBegin, Iterator itPatternEnd, Settings& settings, DropContainer& dropContainer)
    {
	double error = 0.0;
	size_t numPattern = std::distance (itPatternBegin, itPatternEnd);
	size_t numBatches = numPattern/settings.batchSize ();
	size_t numBatches_stored = numBatches;

	Iterator itPatternBatchBegin = itPatternBegin;
	Iterator itPatternBatchEnd = itPatternBatchBegin;
	std::random_shuffle (itPatternBegin, itPatternEnd);

        // create batches
        std::vector<Batch> batches;
        while (numBatches > 0)
        {
	    std::advance (itPatternBatchEnd, settings.batchSize ());
            batches.push_back (Batch (itPatternBatchBegin, itPatternBatchEnd));
	    itPatternBatchBegin = itPatternBatchEnd;
	    --numBatches;
        }

        // add the last pattern to the last batch
	if (itPatternBatchEnd != itPatternEnd)
            batches.push_back (Batch (itPatternBatchEnd, itPatternEnd));


        if (settings.useMultithreading ())
        {
            // -------------------- divide the batches into bunches for each thread --------------
            size_t numThreads = std::thread::hardware_concurrency ();
            size_t batchesPerThread = batches.size () / numThreads;
            typedef std::vector<Batch>::iterator batch_iterator;
            std::vector<std::pair<batch_iterator,batch_iterator>> batchVec;
            batch_iterator itBatchBegin = std::begin (batches);
            batch_iterator itBatchCurrEnd = std::begin (batches);
            batch_iterator itBatchEnd = std::end (batches);
            for (size_t iT = 0; iT < numThreads; ++iT)
            {
                if (iT == numThreads-1)
                    itBatchCurrEnd = itBatchEnd;
                else
                    std::advance (itBatchCurrEnd, batchesPerThread);
                batchVec.push_back (std::make_pair (itBatchBegin, itBatchCurrEnd));
                itBatchBegin = itBatchCurrEnd;
            }
        
            // -------------------- loop  over batches -------------------------------------------
            std::vector<std::future<double>> futures;
            for (auto& batchRange : batchVec)
            {
                futures.push_back (
                    std::async (std::launch::async, [&]() 
                                {
                                    double localError = 0.0;
                                    for (auto it = batchRange.first, itEnd = batchRange.second; it != itEnd; ++it)
                                    {
                                        Batch& batch = *it;
                                        std::tuple<Settings&, Batch&, DropContainer&> settingsAndBatch (settings, batch, dropContainer);
                                        localError += minimizer ((*this), weights, settingsAndBatch);
                                    }
                                    return localError;
                                })
                    );
            }

            for (auto& f : futures)
                error += f.get ();
        }
        else
        {
            for (auto& batch : batches)
            {
                std::tuple<Settings&, Batch&, DropContainer&> settingsAndBatch (settings, batch, dropContainer);
                error += minimizer ((*this), weights, settingsAndBatch);
            }
        }
        
	error /= numBatches_stored;
        settings.testIteration ();
    
	return error;
    }





    template <typename Weights>
        std::vector<double> Net::compute (const std::vector<double>& input, const Weights& weights) const
    {
	std::vector<LayerData> layerData;
	layerData.reserve (m_layers.size ()+1);
	auto itWeight = begin (weights);
	auto itInputBegin = begin (input);
	auto itInputEnd = end (input);
	layerData.push_back (LayerData (itInputBegin, itInputEnd));
	size_t numNodesPrev = input.size ();
	for (auto& layer: m_layers)
	{
	    layerData.push_back (LayerData (layer.numNodes (), itWeight, 
						   begin (layer.activationFunctions ()),
						   layer.modeOutputValues ()));
	    size_t _numWeights = layer.numWeights (numNodesPrev);
	    itWeight += _numWeights;
	    numNodesPrev = layer.numNodes ();
	}
	    

	// --------- forward -------------
	size_t idxLayer = 0, idxLayerEnd = m_layers.size ();
	for (; idxLayer < idxLayerEnd; ++idxLayer)
	{
	    LayerData& prevLayerData = layerData.at (idxLayer);
	    LayerData& currLayerData = layerData.at (idxLayer+1);
		
	    forward (prevLayerData, currLayerData);
	}

	// ------------- fetch output ------------------
	if (layerData.back ().outputMode () == ModeOutputValues::DIRECT)
	{
	    std::vector<double> output;
	    output.assign (layerData.back ().valuesBegin (), layerData.back ().valuesEnd ());
	    return output;
	}
	return layerData.back ().probabilities ();
    }


    template <typename Weights, typename PassThrough>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights) const
    {
	std::vector<double> nothing; // empty gradients; no backpropagation is done, just forward
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (nothing), std::end (nothing), 100, nothing, false);
        return error;
    }

    template <typename Weights, typename PassThrough, typename OutContainer>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights, ModeOutput /*eFetch*/, OutContainer& outputContainer) const
    {
	std::vector<double> nothing; // empty gradients; no backpropagation is done, just forward
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (nothing), std::end (nothing), 1000, outputContainer, true);
        return error;
    }

    
    template <typename Weights, typename Gradients, typename PassThrough>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients) const
    {
        std::vector<double> nothing;
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (gradients), std::end (gradients), 0, nothing, false);
        return error;
    }

    template <typename Weights, typename Gradients, typename PassThrough, typename OutContainer>
        double Net::operator() (PassThrough& settingsAndBatch, const Weights& weights, Gradients& gradients, ModeOutput eFetch, OutContainer& outputContainer) const
    {
	double error = forward_backward(m_layers, settingsAndBatch, std::begin (weights), std::begin (gradients), std::end (gradients), 0, outputContainer, true);
        return error;
    }





    template <typename LayerContainer, typename PassThrough, typename ItWeight, typename ItGradient, typename OutContainer>
        double Net::forward_backward (LayerContainer& _layers, PassThrough& settingsAndBatch, 
			     ItWeight itWeightBegin, 
			     ItGradient itGradientBegin, ItGradient itGradientEnd, 
			     size_t trainFromLayer, 
			     OutContainer& outputContainer, bool fetchOutput) const
    {
        Settings& settings = std::get<0>(settingsAndBatch);
        Batch& batch = std::get<1>(settingsAndBatch);
	DropContainer& dropContainer = std::get<2>(settingsAndBatch);

	bool usesDropOut = !dropContainer.empty ();

        LayerData::const_dropout_iterator itDropOut;
        if (usesDropOut)
            itDropOut = std::begin (dropContainer);
        
	if (_layers.empty ())
        {
            std::cout << "no layers in this net" << std::endl;
	    throw std::string ("no layers in this net");
        }


	double sumError = 0.0;
	double sumWeights = 0.0;	// -------------

        // ----------- create layer data -----------------
        assert (_layers.back ().numNodes () == outputSize ());
        size_t totalNumWeights = 0;
        std::vector<LayerData> layerData;
        layerData.reserve (_layers.size ()+1);
        ItWeight itWeight = itWeightBegin;
        ItGradient itGradient = itGradientBegin;
        size_t numNodesPrev = inputSize ();
        layerData.push_back (LayerData (numNodesPrev));
        if (usesDropOut)
        {
            layerData.back ().setDropOut (itDropOut);
            itDropOut += _layers.back ().numNodes ();
        }
        for (auto& layer: _layers)
        {
            if (itGradientBegin == itGradientEnd)
                layerData.push_back (LayerData (layer.numNodes (), itWeight, 
                                                std::begin (layer.activationFunctions ()),
                                                layer.modeOutputValues ()));
            else
                layerData.push_back (LayerData (layer.numNodes (), itWeight, itGradient, 
                                                std::begin (layer.activationFunctions ()),
                                                std::begin (layer.inverseActivationFunctions ()),
                                                layer.modeOutputValues ()));

            if (usesDropOut)
            {
                layerData.back ().setDropOut (itDropOut);
                itDropOut += layer.numNodes ();
            }
            size_t _numWeights = layer.numWeights (numNodesPrev);
            totalNumWeights += _numWeights;
            itWeight += _numWeights;
            itGradient += _numWeights;
            numNodesPrev = layer.numNodes ();
//                std::cout << layerData.back () << std::endl;
        }
	assert (totalNumWeights > 0);



        // ---------------------------------- loop over pattern -------------------------------------------------------
        typename Pattern::const_iterator itInputBegin;
        typename Pattern::const_iterator itInputEnd;
	for (const Pattern& _pattern : batch)
	{
            bool isFirst = true;
            for (auto& _layerData: layerData)
            {
                _layerData.clear ();
                if (isFirst)
                {
                    itInputBegin = _pattern.beginInput ();
                    itInputEnd = _pattern.endInput ();
                    _layerData.setInput (itInputBegin, itInputEnd);
                    isFirst = false;
                }
            }
            
	    // --------- forward -------------
//            std::cout << "forward" << std::endl;
	    bool doTraining (true);
	    size_t idxLayer = 0, idxLayerEnd = _layers.size ();
	    for (; idxLayer < idxLayerEnd; ++idxLayer)
	    {
		LayerData& prevLayerData = layerData.at (idxLayer);
		LayerData& currLayerData = layerData.at (idxLayer+1);
		
		doTraining = idxLayer >= trainFromLayer;
		if (doTraining)
		    forward_training (prevLayerData, currLayerData);
		else
		    forward (prevLayerData, currLayerData);
	    }

            
            // ------------- fetch output ------------------
            if (fetchOutput)
            {
		if (layerData.back ().outputMode () == ModeOutputValues::DIRECT)
		    outputContainer.insert (outputContainer.end (), layerData.back ().valuesBegin (), layerData.back ().valuesEnd ());
		else
		    outputContainer = layerData.back ().probabilities ();
            }


	    // ------------- error computation -------------
	    // compute E and the deltas of the computed output and the true output 
	    itWeight = itWeightBegin;
	    double error = errorFunction (layerData.back (), _pattern.output (), 
					  itWeight, itWeight + totalNumWeights, 
					  _pattern.weight (), settings.factorWeightDecay (),
                                          settings.regularization ());
	    sumWeights += fabs (_pattern.weight ());
	    sumError += error;

	    if (!doTraining) // no training
		continue;

	    // ------------- backpropagation -------------
	    idxLayer = layerData.size ();
	    for (auto itLayer = end (_layers), itLayerBegin = begin (_layers); itLayer != itLayerBegin; --itLayer)
	    {
		--idxLayer;
		doTraining = idxLayer >= trainFromLayer;
		if (!doTraining) // no training
		    break;

		LayerData& currLayerData = layerData.at (idxLayer);
		LayerData& prevLayerData = layerData.at (idxLayer-1);

		backward (prevLayerData, currLayerData);

                // the factorWeightDecay has to be scaled by 1/n where n is the number of weights (synapses)
                // because L1 and L2 regularization
                //
                //  http://neuralnetworksanddeeplearning.com/chap3.html#overfitting_and_regularization
                //
                // L1 : -factorWeightDecay*sgn(w)/numWeights
                // L2 : -factorWeightDecay/numWeights
		update (prevLayerData, currLayerData, settings.factorWeightDecay ()/totalNumWeights, settings.regularization ());
	    }
	}
        
        double batchSize = std::distance (std::begin (batch), std::end (batch));
        for (auto it = itGradientBegin; it != itGradientEnd; ++it)
            (*it) /= batchSize;


	sumError /= sumWeights;
	return sumError;
    }



    template <typename OutIterator>
    void Net::initializeWeights (WeightInitializationStrategy eInitStrategy, OutIterator itWeight)
    {
        if (eInitStrategy == WeightInitializationStrategy::XAVIER)
        {
            // input and output properties
            int numInput = inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
                double nIn = numInput;
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    (*itWeight) = NN::gaussDouble (0.0, sqrt (2.0/nIn)); // factor 2.0 for ReLU
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }

        if (eInitStrategy == WeightInitializationStrategy::XAVIERUNIFORM)
        {
            // input and output properties
            int numInput = inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
                double nIn = numInput;
                double minVal = -sqrt(2.0/nIn);
                double maxVal = sqrt (2.0/nIn);
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    
                    (*itWeight) = NN::uniformDouble (minVal, maxVal); // factor 2.0 for ReLU
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }
        
        if (eInitStrategy == WeightInitializationStrategy::TEST)
        {
            // input and output properties
            int numInput = inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
//                double nIn = numInput;
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    (*itWeight) = NN::gaussDouble (0.0, 0.1);
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }

        if (eInitStrategy == WeightInitializationStrategy::LAYERSIZE)
        {
            // input and output properties
            int numInput = inputSize ();

            // compute variance and mean of input and output
            //...
	

            // compute the weights
            for (auto& layer: layers ())
            {
                double nIn = numInput;
                for (size_t iWeight = 0, iWeightEnd = layer.numWeights (numInput); iWeight < iWeightEnd; ++iWeight)
                {
                    (*itWeight) = NN::gaussDouble (0.0, sqrt (layer.numWeights (nIn))); // factor 2.0 for ReLU
                    ++itWeight;
                }
                numInput = layer.numNodes ();
            }
            return;
        }

    }


    


    template <typename Container, typename ItWeight>
        double Net::errorFunction (LayerData& layerData,
                                   Container truth,
                                   ItWeight itWeight,
                                   ItWeight itWeightEnd,
                                   double patternWeight,
                                   double factorWeightDecay,
                                   EnumRegularization eRegularization) const
    {
	double error (0);
	switch (m_eErrorFunction)
	{
	case ModeErrorFunction::SUMOFSQUARES:
	{
	    error = sumOfSquares (layerData.valuesBegin (), layerData.valuesEnd (), begin (truth), end (truth), 
				  layerData.deltasBegin (), layerData.deltasEnd (), 
				  layerData.inverseFunctionBegin (), 
				  patternWeight);
	    break;
	}
	case ModeErrorFunction::CROSSENTROPY:
	{
	    assert (layerData.outputMode () != ModeOutputValues::DIRECT);
	    std::vector<double> probabilities = layerData.probabilities ();
	    error = crossEntropy (begin (probabilities), end (probabilities), 
				  begin (truth), end (truth), 
				  layerData.deltasBegin (), layerData.deltasEnd (), 
				  layerData.inverseFunctionBegin (), 
				  patternWeight);
	    break;
	}
	case ModeErrorFunction::CROSSENTROPY_MUTUALEXCLUSIVE:
	{
	    assert (layerData.outputMode () != ModeOutputValues::DIRECT);
	    std::vector<double> probabilities = layerData.probabilities ();
	    error = softMaxCrossEntropy (begin (probabilities), end (probabilities), 
					 begin (truth), end (truth), 
					 layerData.deltasBegin (), layerData.deltasEnd (), 
					 layerData.inverseFunctionBegin (), 
					 patternWeight);
	    break;
	}
	}
	if (factorWeightDecay != 0 && eRegularization != EnumRegularization::NONE)
        {
            error = weightDecay (error, itWeight, itWeightEnd, factorWeightDecay, eRegularization);
        }
	return error;
    } 








    template <typename Minimizer>
        void Net::preTrain (std::vector<double>& weights,
        	  std::vector<Pattern>& trainPattern,
        	  const std::vector<Pattern>& testPattern,
                  Minimizer& minimizer, Settings& settings)
    {
        auto itWeightGeneral = std::begin (weights);
        std::vector<Pattern> prePatternTrain (trainPattern.size ());
        std::vector<Pattern> prePatternTest (testPattern.size ());

        size_t _inputSize = inputSize ();

        // transform pattern using the created preNet
        auto initializePrePattern = [&](const std::vector<Pattern>& pttrnInput, std::vector<Pattern>& pttrnOutput)
        {
            pttrnOutput.clear ();
            std::transform (std::begin (pttrnInput), std::end (pttrnInput),
                        std::back_inserter (pttrnOutput), 
                        [](const Pattern& p)
                        {
                            Pattern pat (p.input (), p.input (), p.weight ());
                            return pat;
                        });
        };

        initializePrePattern (trainPattern, prePatternTrain);
        initializePrePattern (testPattern, prePatternTest);

        std::vector<double> originalDropFractions = settings.dropFractions ();

        for (auto& _layer : layers ())
        {
            // compute number of weights (as a function of the number of incoming nodes)
            // fetch number of nodes
            size_t numNodes = _layer.numNodes ();
            size_t numWeights = _layer.numWeights (_inputSize);

            // ------------------
            NN::Net preNet;
            if (!originalDropFractions.empty ())
            {
                originalDropFractions.erase (originalDropFractions.begin ());
                settings.setDropOut (originalDropFractions.begin (), originalDropFractions.end (), settings.dropRepetitions ());
            }
            std::vector<double> preWeights;

            // define the preNet (pretraining-net) for this layer
            // outputSize == inputSize, because this is an autoencoder;
            preNet.setInputSize (_inputSize);
            preNet.addLayer (NN::Layer (numNodes, _layer.activationFunction ()));
            preNet.addLayer (NN::Layer (_inputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::DIRECT)); 
            preNet.setErrorFunction (NN::ModeErrorFunction::SUMOFSQUARES);
            preNet.setOutputSize (_inputSize); // outputSize is the inputSize (autoencoder)

            // initialize weights
            preNet.initializeWeights (NN::WeightInitializationStrategy::XAVIERUNIFORM, 
                                      std::back_inserter (preWeights));

            // overwrite already existing weights from the "general" weights
            std::copy (itWeightGeneral, itWeightGeneral+numWeights, preWeights.begin ());
            

            // train the "preNet"
            preNet.train (preWeights, prePatternTrain, prePatternTest, minimizer, settings);

            // fetch the pre-trained weights (without the output part of the autoencoder)
            std::copy (std::begin (preWeights), std::begin (preWeights) + numWeights, itWeightGeneral);

            // advance the iterator on the incoming weights
            itWeightGeneral += numWeights;

            // remove the weights of the output layer of the preNet
            preWeights.erase (preWeights.begin () + numWeights, preWeights.end ());

            // remove the outputLayer of the preNet
            preNet.removeLayer ();

            // set the output size to the number of nodes in the new output layer (== last hidden layer)
            preNet.setOutputSize (numNodes);
            
            // transform pattern using the created preNet
            auto proceedPattern = [&](std::vector<Pattern>& pttrn)
            {
                std::for_each (std::begin (pttrn), std::end (pttrn),
                                [&preNet,&preWeights](Pattern& p)
                {
                    std::vector<double> output = preNet.compute (p.input (), preWeights);
                    Pattern pat (output, output, p.weight ());
                    p = pat;
                });
            };


            proceedPattern (prePatternTrain);
            proceedPattern (prePatternTest);


            // the new input size is the output size of the already reduced preNet
            _inputSize = preNet.layers ().back ().numNodes ();
        }
    }
















}; // namespace NN
}; // namespace TMVA

#endif
