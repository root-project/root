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




double gaussDouble (double mean, double sigma);


int randomInt (int maxValue);


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



template <typename ItSource, typename ItWeight, typename ItTarget>
void applyWeights (ItSource itSourceBegin, ItSource itSourceEnd, ItWeight itWeight, ItTarget itTargetBegin, ItTarget itTargetEnd)
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



template <typename ItSource, typename ItWeight, typename ItPrev>
void applyWeightsBackwards (ItSource itCurrBegin, ItSource itCurrEnd, ItWeight itWeight, ItPrev itPrevBegin, ItPrev itPrevEnd)
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



template <bool isL1, typename ItSource, typename ItDelta, typename ItTargetGradient, typename ItGradient, typename ItWeight>
void update (ItSource itSource, ItSource itSourceEnd, 
	     ItDelta itTargetDeltaBegin, ItDelta itTargetDeltaEnd, 
	     ItTargetGradient itTargetGradientBegin, 
	     ItGradient itGradient, 
	     ItWeight itWeight, double weightDecay)
{
    while (itSource != itSourceEnd)
    {
        auto itTargetDelta = itTargetDeltaBegin;
        auto itTargetGradient = itTargetGradientBegin;
        while (itTargetDelta != itTargetDeltaEnd)
        {
            //                                                                                       L1 regularization                   L2 regularization
	    (*itGradient) -= + (*itTargetDelta) * (*itSource) * (*itTargetGradient) + (isL1 ? std::copysign (weightDecay,(*itWeight)) : (*itWeight) * weightDecay);
            ++itTargetDelta; ++itTargetGradient; ++itGradient; ++itWeight;
        }
        ++itSource; 
    }
}








    template <typename Function, typename Weights, typename PassThrough>
        double Steepest::operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<double> gradients (numWeights, 0.0);
	std::vector<double> localWeights (begin (weights), end (weights));
        if (m_prevGradients.empty ())
            m_prevGradients.assign (weights.size (), 0);


        double E = fitnessFunction (passThrough, weights, gradients);
//        double Emin = E;

        bool success = true;
        size_t currentRepetition = 0;
        while (success)
        {
            if (currentRepetition >= m_repetitions)
                break;

//            double alpha = gaussDouble (m_alpha, m_alpha/10.0);
            double alpha = m_alpha;

            auto itLocW = begin (localWeights);
            auto itLocWEnd = end (localWeights);
            auto itG = begin (gradients);
            auto itPrevG = begin (m_prevGradients);
            for (; itLocW != itLocWEnd; ++itLocW, ++itG, ++itPrevG)
            {
                (*itG) *= alpha;
                (*itG) += m_beta * (*itPrevG);
                (*itLocW) += (*itG);
                (*itPrevG) = (*itG);
            }
            gradients.assign (numWeights, 0.0);
            E = fitnessFunction (passThrough, localWeights, gradients);

            itLocW = begin (localWeights);
            itLocWEnd = end (localWeights);
            auto itW = begin (weights);
            for (; itLocW != itLocWEnd; ++itLocW, ++itW)
            {
                (*itW) = (*itLocW);
            }

            /* if (E < Emin) */
            /* { */
            /*     Emin = E; */
            /*     std::cout << "."; */
            /* } */
            /* else */
            /*     std::cout << "X"; */

            ++currentRepetition;
        }
        return E;
    }





    template <typename Function, typename Weights, typename Gradients, typename PassThrough>
        double SteepestThreaded::fitWrapper (Function& function, PassThrough& passThrough, Weights weights)
    {
	return fitnessFunction (passThrough, weights);
    }



    template <typename Function, typename Weights, typename PassThrough>
        double SteepestThreaded::operator() (Function& fitnessFunction, Weights& weights, PassThrough& passThrough) 
    {
	size_t numWeights = weights.size ();
	std::vector<double> gradients (numWeights, 0.0);
	std::vector<double> localWeights (begin (weights), end (weights));
        if (m_prevGradients.empty ())
            m_prevGradients.assign (weights.size (), 0);


        fitnessFunction (passThrough, weights, gradients);

        std::vector<std::future<double> > futures;
        std::vector<std::pair<double,double> > factors;
        for (size_t i = 0; i < m_repetitions; ++i)
        {
            std::vector<double> tmpWeights (weights);
            double alpha = gaussDouble (m_alpha, m_beta);
            double beta  = gaussDouble (m_alpha, m_beta);
            auto itGradient = begin (gradients);
            auto itPrevGradient = begin (m_prevGradients);
            std::for_each (begin (tmpWeights), end (tmpWeights), [alpha,beta,&itGradient,&itPrevGradient](double& w) 
                           { 
                               w += alpha * (*itGradient) + beta * (*itPrevGradient);
                               ++itGradient; ++itPrevGradient;
                           }
                );

	    // fitnessFunction is a function template which turns into a function at invocation
	    // if we call fitnessFunction directly in async, the templat parameters
	    // cannot be deduced correctly. Through the lambda function, the types are 
            // already deduced correctly for the lambda function and the async. The deduction for 
	    // the template function is then done from within the lambda function. 
	    futures.push_back (std::async (std::launch::async, [&fitnessFunction, &passThrough, tmpWeights]() mutable 
					   {  
					       return fitnessFunction (passThrough, tmpWeights); 
					   }) );

            factors.push_back (std::make_pair (alpha,beta));
        }

        // select best
        double bestAlpha = m_alpha, bestBeta = 0.0;
        auto itE = begin (futures);
        double bestE = 1e100;
        for (auto& alphaBeta : factors)
        {
            double E = (*itE).get ();
            if (E < bestE)
            {
                bestAlpha = alphaBeta.first;
                bestBeta = alphaBeta.second;
                bestE = E;
            }
            ++itE;
        }

        // walk this way
        auto itGradient = begin (gradients);
        auto itPrevGradient = begin (m_prevGradients);
        std::for_each (begin (weights), end (weights), [bestAlpha,bestBeta,&itGradient,&itPrevGradient](double& w) 
                       { 
                           double grad = bestAlpha * (*itGradient) + bestBeta * (*itPrevGradient);
                           w += grad;
                           (*itPrevGradient) = grad;
                           ++itGradient; ++itPrevGradient;
                       }
            );
        return bestE;
    }








    template <typename Function, typename Weights, typename PassThrough>
        double MaxGradWeight::operator() (Function& fitnessFunction, const Weights& weights, PassThrough& passThrough) 
    {
	double alpha = m_learningRate;

	size_t numWeights = weights.size ();
	std::vector<double> gradients (numWeights, 0.0);
	std::vector<double> localWeights (begin (weights), end (weights));


        double Ebase = fitnessFunction (passThrough, weights, gradients);
        double Emin = Ebase;

        bool success = true;
        size_t currentRepetition = 0;
        while (success)
        {
            if (currentRepetition >= m_repetitions)
                break;

	    auto itMaxGradElement = std::max_element (begin (gradients), end (gradients));
	    auto idx = std::distance (begin (gradients), itMaxGradElement);
	    localWeights.at (idx) += alpha*(*itMaxGradElement);
            gradients.assign (numWeights, 0.0);
            double E = fitnessFunction (passThrough, localWeights, gradients);

            if (E < Emin)
            {
                Emin = E;

                auto itLocW = begin (localWeights);
                auto itLocWEnd = end (localWeights);
                auto itW = begin (weights);
                for (; itLocW != itLocWEnd; ++itLocW, ++itW)
                {
                    (*itW) = (*itLocW);
                }
            }
            ++currentRepetition;
        }
        return Emin;
    }









template <typename ItOutput, typename ItTruth, typename ItDelta, typename ItInvActFnc>
double sumOfSquares (ItOutput itOutputBegin, ItOutput itOutputEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc itInvActFnc, double patternWeight) 
{
    double errorSum = 0.0;

    // output - truth
    ItTruth itTruth = itTruthBegin;
    bool hasDeltas = (itDelta != itDeltaEnd);
    for (ItOutput itOutput = itOutputBegin; itOutput != itOutputEnd; ++itOutput, ++itTruth)
    {
	assert (itTruth != itTruthEnd);
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
    double softMaxCrossEntropy (ItOutput itProbabilityBegin, ItOutput itProbabilityEnd, ItTruth itTruthBegin, ItTruth itTruthEnd, ItDelta itDelta, ItDelta itDeltaEnd, ItInvActFnc /*itInvActFnc*/, double patternWeight) 
{
    double errorSum = 0.0;

    bool hasDeltas = (itDelta != itDeltaEnd);
    // output - truth
    ItTruth itTruth = itTruthBegin;
    for (auto itProbability = itProbabilityBegin; itProbability != itProbabilityEnd; ++itProbability, ++itTruth)
    {
	assert (itTruth != itTruthEnd);
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
double weightDecay (double error, ItWeight itWeight, ItWeight itWeightEnd, double factorWeightDecay)
{

    // weight decay (regularization)
    double w = 0;
    double sumW = 0;
    for (; itWeight != itWeightEnd; ++itWeight)
    {
	double weight = (*itWeight);
	w += weight*weight;
        sumW += fabs (weight);
    }
    return error + 0.5 * w * factorWeightDecay / sumW;
}














template <typename LAYERDATA>
void forward (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		  currLayerData.weightsBegin (), 
		  currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin ());
}

template <typename LAYERDATA>
void forward_training (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    applyWeights (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		  currLayerData.weightsBegin (), 
		  currLayerData.valuesBegin (), currLayerData.valuesEnd ());
    applyFunctions (currLayerData.valuesBegin (), currLayerData.valuesEnd (), currLayerData.functionBegin (), 
		    currLayerData.inverseFunctionBegin (), currLayerData.valueGradientsBegin ());
}


template <typename LAYERDATA>
void backward (LAYERDATA& prevLayerData, LAYERDATA& currLayerData)
{
    applyWeightsBackwards (currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			   currLayerData.weightsBegin (), 
			   prevLayerData.deltasBegin (), prevLayerData.deltasEnd ());
}



template <typename LAYERDATA>
void update (const LAYERDATA& prevLayerData, LAYERDATA& currLayerData, double weightDecay, bool isL1)
{
    if (weightDecay != 0.0) // has weight regularization
	if (isL1)  // L1 regularization ( sum(|w|) )
	{
	    update<true> (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
			  currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			  currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin (), 
			  currLayerData.weightsBegin (), weightDecay);
	}
	else // L2 regularization ( sum(w^2) )
	{
	    update<false> (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
			   currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
			   currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin (), 
			   currLayerData.weightsBegin (), weightDecay);
	}
    else
    { // no weight regularization
	update (prevLayerData.valuesBegin (), prevLayerData.valuesEnd (), 
		currLayerData.deltasBegin (), currLayerData.deltasEnd (), 
		currLayerData.valueGradientsBegin (), currLayerData.gradientsBegin ());
    }
}












    template <typename WeightsType>
        void Net::dropOutWeightFactor (const DropContainer& dropContainer, WeightsType& weights, double factor)
    {
//        return;
	// reduce weights because of dropped nodes
	// if dropOut enabled
	if (dropContainer.empty ())
	    return;

	// fill the dropOut-container
	auto itWeight = begin (weights);
	auto itDrop = begin (dropContainer);
	for (auto itLayer = begin (m_layers), itLayerEnd = end (m_layers)-1; itLayer != itLayerEnd; ++itLayer)
	{
	    auto& layer = *itLayer;
	    auto& nextLayer = *(itLayer+1);
	    /* // in the first and last layer, all the nodes are always on */
	    /* if (itLayer == begin (m_layers)) // is first layer */
	    /* { */
	    /*     itDrop += layer.numNodes (); */
	    /*     itWeight += layer.numNodes () * nextLayer.numNodes (); */
	    /*     continue; */
	    /* } */

	    auto itLayerDrop = itDrop;
	    for (size_t i = 0, iEnd = layer.numNodes (); i < iEnd; ++i)
	    {
		auto itNextDrop = itDrop + layer.numNodes ();
	    
		bool drop = (*itLayerDrop);
		for (size_t j = 0, jEnd = nextLayer.numNodes (); j < jEnd; ++j)
		{
		    if (drop && (*itNextDrop))
		    {
			(*itWeight) *= factor;
		    }
		    ++itWeight;
		    ++itNextDrop;
		}
		++itLayerDrop;
	    }
	}
    }



    template <typename Minimizer>
        double Net::train (std::vector<double>& weights, 
		  std::vector<Pattern>& trainPattern, 
		  const std::vector<Pattern>& testPattern, 
                  Minimizer& minimizer, Settings& settings)
    {
//        std::cout << "START TRAINING" << std::endl;
        settings.pads (4);
        settings.create ("trainErrors", 100, 0, 100, 100, 0,1);
        settings.create ("testErrors", 100, 0, 100, 100, 0,1);

        size_t convergenceCount = 0;
        size_t maxConvergenceCount = 0;
        double minError = 1e10;

        size_t cycleCount = 0;
        size_t testCycleCount = 0;
        double testError = 1e20;
        double trainError = 1e20;
        size_t dropOutChangeCount = 0;

	DropContainer dropContainer;

        settings.startTraining ();
        // until convergence
        do
        {
//            std::cout << "train cycle " << cycleCount << std::endl;
            ++cycleCount;

	    // shuffle training pattern
//            std::random_shuffle (begin (trainPattern), end (trainPattern));
	    double dropFraction = settings.dropFraction ();

	    // if dropOut enabled
            if (dropFraction > 0 && dropOutChangeCount % settings.dropRepetitions () == 0)
	    {
		if (dropOutChangeCount > 0)
		    dropOutWeightFactor (dropContainer, weights, dropFraction);

		// fill the dropOut-container
		dropContainer.clear ();
		for (auto itLayer = begin (m_layers), itLayerEnd = end (m_layers); itLayer != itLayerEnd; ++itLayer)
		{
		    auto& layer = *itLayer;
		    // in the first and last layer, all the nodes are always on
		    if (itLayer == begin (m_layers) || itLayer == end (m_layers)-1) // is first layer or is last layer
		    {
			dropContainer.insert (end (dropContainer), layer.numNodes (), true);
			continue;
		    }
		    // how many nodes have to be dropped
		    size_t numDrops = settings.dropFraction () * layer.numNodes ();
		    dropContainer.insert (end (dropContainer), layer.numNodes ()-numDrops, true); // add the markers for the nodes which are enabled
		    dropContainer.insert (end (dropContainer), numDrops, false); // add the markers for the disabled nodes
		    // shuffle 
		    std::random_shuffle (end (dropContainer)-layer.numNodes (), end (dropContainer)); // shuffle enabled and disabled markers
		}
		if (dropOutChangeCount > 0)
                    dropOutWeightFactor (dropContainer, weights, 1.0/dropFraction);
	    }

	    // execute training cycle
            settings.startTrainCycle ();
            trainError = trainCycle (minimizer, weights, begin (trainPattern), end (trainPattern), settings, dropContainer);
            settings.endTrainCycle (trainError);
	    

	    // check if we execute a test
            if (testCycleCount % settings.testRepetitions () == 0)
            {
		if (dropOutChangeCount > 0)
		    dropOutWeightFactor (dropContainer, weights, dropFraction);

		dropContainer.clear (); // execute test on all the nodes
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
		    std::tuple<Settings&, Batch&, DropContainer&> passThrough (settings, batch, dropContainer);
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



//	(*this).print (std::cout);
//            std::cout << "check convergence; minError " << minError << "  current " << testError << "  current convergence count " << convergenceCount << std::endl;
            if (testError < minError)
            {
                convergenceCount = 0;
                minError = testError;
            }
            else
            {
                ++convergenceCount;
                maxConvergenceCount = std::max (convergenceCount, maxConvergenceCount);
            }


	    if (convergenceCount >= settings.convergenceSteps () || testError <= 0)
	    {
		if (dropOutChangeCount > 0)
		    dropOutWeightFactor (dropContainer, weights, dropFraction);
		break;
	    }


            TString convText = Form( "<D^2> (train/test/epoch/conv/maxConv): %.4g/%.4g/%d/%d/%d", trainError, testError, (int)cycleCount, (int)convergenceCount, (int)maxConvergenceCount);
            double progress = 100*(double)maxConvergenceCount /(double)settings.convergenceSteps ();
            settings.cycle (progress, convText);
        }
	while (true);
        TString convText = Form( "<D^2> (train/test/epoch): %.4g/%.4g/%d", trainError, testError, (int)cycleCount);
        double progress = 100*(double)maxConvergenceCount /(double)settings.convergenceSteps ();
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
	while (numBatches > 0)
	{
            settings.testIteration ();
	    std::advance (itPatternBatchEnd, settings.batchSize ());
            Batch batch (itPatternBatchBegin, itPatternBatchEnd);
            std::tuple<Settings&, Batch&, DropContainer&> settingsAndBatch (settings, batch, dropContainer);
	    error += minimizer ((*this), weights, settingsAndBatch);
	    itPatternBatchBegin = itPatternBatchEnd;
	    --numBatches;
	}
	if (itPatternBatchEnd != itPatternEnd)
        {
            settings.testIteration ();
            Batch batch (itPatternBatchEnd, itPatternEnd);
            std::tuple<Settings&, Batch&, DropContainer&> settingsAndBatch (settings, batch, dropContainer);
	    error += minimizer ((*this), weights, settingsAndBatch);
        }
	error /= numBatches_stored;
        settings.testIteration ();
    
	return error;
    }






    /* size_t Net::numWeights (size_t numInputNodes, size_t trainingStartLayer) const  */
    /* { */
    /*     size_t num (0); */
    /*     size_t index (0); */
    /*     size_t prevNodes (numInputNodes); */
    /*     for (auto& layer : m_layers) */
    /*     { */
    /*         if (index >= trainingStartLayer) */
    /*     	num += layer.numWeights (prevNodes); */
    /*         prevNodes = layer.numNodes (); */
    /*         ++index; */
    /*     } */
    /*     return num; */
    /* } */


    template <typename Weights>
        std::vector<double> Net::compute (const std::vector<double>& input, const Weights& weights) const
    {
	std::vector<LayerData> layerData;
	layerData.reserve (m_layers.size ()+1);
	auto itWeight = begin (weights);
	auto itInputBegin = begin (input);
	auto itInputEnd = end (input);
	DropContainer drop;
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
	DropContainer& drop = std::get<2>(settingsAndBatch);
	
	bool usesDropOut = !drop.empty ();

	std::vector<std::vector<std::function<double(double)> > > activationFunctionsDropOut;
	std::vector<std::vector<std::function<double(double)> > > inverseActivationFunctionsDropOut;

	if (_layers.empty ())
        {
            std::cout << "no layers in this net" << std::endl;
	    throw std::string ("no layers in this net");
        }

	if (usesDropOut)
	{
	    auto itDrop = begin (drop);
	    for (auto& layer: _layers)
	    {
		activationFunctionsDropOut.push_back (std::vector<std::function<double(double)> >());
		inverseActivationFunctionsDropOut.push_back (std::vector<std::function<double(double)> >());
		auto& actLine = activationFunctionsDropOut.back ();
		auto& invActLine = inverseActivationFunctionsDropOut.back ();
		auto& actFnc = layer.activationFunctions ();
		auto& invActFnc = layer.inverseActivationFunctions ();
		for (auto itAct = begin (actFnc), itActEnd = end (actFnc), itInv = begin (invActFnc); itAct != itActEnd; ++itAct, ++itInv)
		{
		    if (!*itDrop)
		    {
			actLine.push_back (ZeroFnc);
			invActLine.push_back (ZeroFnc);
		    }
		    else
		    {
			actLine.push_back (*itAct);
			invActLine.push_back (*itInv);
		    }
		    ++itDrop;
		}
	    }
	}

	double sumError = 0.0;
	double sumWeights = 0.0;	// -------------
	for (const Pattern& pattern : batch)
	{
	    assert (_layers.back ().numNodes () == pattern.output ().size ());
	    size_t totalNumWeights = 0;
	    std::vector<LayerData> layerData;
            layerData.reserve (_layers.size ()+1);
	    ItWeight itWeight = itWeightBegin;
	    ItGradient itGradient = itGradientBegin;
	    typename Pattern::const_iterator itInputBegin = pattern.beginInput ();
	    typename Pattern::const_iterator itInputEnd = pattern.endInput ();
	    layerData.push_back (LayerData (itInputBegin, itInputEnd));
	    size_t numNodesPrev = pattern.input ().size ();
	    auto itActFncLayer = begin (activationFunctionsDropOut);
	    auto itInvActFncLayer = begin (inverseActivationFunctionsDropOut);
	    for (auto& layer: _layers)
	    {
		const std::vector<std::function<double(double)> >& actFnc = usesDropOut ? (*itActFncLayer) : layer.activationFunctions ();
		const std::vector<std::function<double(double)> >& invActFnc = usesDropOut ? (*itInvActFncLayer) : layer.inverseActivationFunctions ();
		if (usesDropOut)
		{
		    ++itActFncLayer;
		    ++itInvActFncLayer;
		}
		if (itGradientBegin == itGradientEnd)
		    layerData.push_back (LayerData (layer.numNodes (), itWeight, 
						    begin (actFnc),
						    layer.modeOutputValues ()));
		else
		    layerData.push_back (LayerData (layer.numNodes (), itWeight, itGradient, 
						    begin (actFnc), begin (invActFnc),
						    layer.modeOutputValues ()));
		size_t _numWeights = layer.numWeights (numNodesPrev);
		totalNumWeights += _numWeights;
		itWeight += _numWeights;
		itGradient += _numWeights;
		numNodesPrev = layer.numNodes ();
//                std::cout << layerData.back () << std::endl;
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
	    double error = errorFunction (layerData.back (), pattern.output (), 
					  itWeight, itWeight + totalNumWeights, 
					  pattern.weight (), settings.factorWeightDecay ());
	    sumWeights += fabs (pattern.weight ());
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
		update (prevLayerData, currLayerData, settings.factorWeightDecay ()/sumWeights, settings.isL1 ());
	    }
	}
        
        double batchSize = std::distance (std::begin (batch), std::end (batch));
        for (auto it = itGradientBegin; it != itGradientEnd; ++it)
            (*it) /= batchSize;


	sumError /= sumWeights;
	return sumError;
    }



    template <typename ItPat, typename OutIterator>
    void Net::initializeWeights (WeightInitializationStrategy eInitStrategy, 
				     ItPat itPatternBegin, 
                                 ItPat /*itPatternEnd*/, 
				     OutIterator itWeight)
    {
        if (eInitStrategy == WeightInitializationStrategy::XAVIER)
        {
            // input and output properties
            int numInput = (*itPatternBegin).inputSize ();

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

        if (eInitStrategy == WeightInitializationStrategy::TEST)
        {
            // input and output properties
            int numInput = (*itPatternBegin).inputSize ();

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
            int numInput = (*itPatternBegin).inputSize ();

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
        double Net::errorFunction (LayerData& layerData, Container truth, ItWeight itWeight, ItWeight itWeightEnd, double patternWeight, double factorWeightDecay) const
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
	if (factorWeightDecay != 0)
	    error = weightDecay (error, itWeight, itWeightEnd, factorWeightDecay);
	return error;
    } 



























}; // namespace NN
}; // namespace TMVA

#endif
