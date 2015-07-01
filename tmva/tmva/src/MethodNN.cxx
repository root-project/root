// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodNN                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A neural network implementation                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer      <peter.speckmayer@gmx.ch> - CERN, Switzerland       *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// neural network implementation
//_______________________________________________________________________

#include "TString.h"
#include "TTree.h"
#include "TFile.h"
#include "TFormula.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodNN.h"
#include "TMVA/Timer.h"
#include "TMVA/Types.h"
#include "TMVA/Tools.h"
#include "TMVA/Config.h"
#include "TMVA/Ranking.h"

#include "TMVA/NeuralNet.h"
#include "TMVA/Monitoring.h"

#include <algorithm>
#include <iostream>

REGISTER_METHOD(NN)

ClassImp(TMVA::MethodNN)




namespace TMVA
{
namespace NN
{
template <typename Container, typename T>
void gaussDistribution (Container& container, T mean, T sigma)
{
    for (auto it = begin (container), itEnd = end (container); it != itEnd; ++it)
    {
        (*it) = NN::gaussDouble (mean, sigma);
    }
}
};
};






//______________________________________________________________________________
TMVA::MethodNN::MethodNN( const TString& jobName,
                          const TString& methodTitle,
                          DataSetInfo& theData,
                          const TString& theOption,
                          TDirectory* theTargetDir )
    : MethodBase( jobName, Types::kNN, methodTitle, theData, theOption, theTargetDir )
    , fResume (false)
{
   // standard constructor
}

//______________________________________________________________________________
TMVA::MethodNN::MethodNN( DataSetInfo& theData,
                          const TString& theWeightFile,
                          TDirectory* theTargetDir )
   : MethodBase( Types::kNN, theData, theWeightFile, theTargetDir )
    , fResume (false)
{
   // constructor from a weight file
}

//______________________________________________________________________________
TMVA::MethodNN::~MethodNN()
{
   // destructor
   // nothing to be done
}

//_______________________________________________________________________
Bool_t TMVA::MethodNN::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // MLP can handle classification with 2 classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   if (type == Types::kMulticlass ) return kTRUE;
   if (type == Types::kRegression ) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
void TMVA::MethodNN::Init()
{
   // default initializations
}

//_______________________________________________________________________
void TMVA::MethodNN::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // know options:
   // TrainingMethod  <string>     Training method
   //    available values are:         BP   Back-Propagation <default>
   //                                  GA   Genetic Algorithm (takes a LONG time)
   //
   // LearningRate    <float>      NN learning rate parameter
   // DecayRate       <float>      Decay rate for learning parameter
   // TestRate        <int>        Test for overtraining performed at each #th epochs
   //
   // BPMode          <string>     Back-propagation learning mode
   //    available values are:         sequential <default>
   //                                  batch
   //
   // BatchSize       <int>        Batch size: number of events/batch, only set if in Batch Mode,
   //                                          -1 for BatchSize=number_of_events

   // DeclareOptionRef(fTrainMethodS="SD", "TrainingMethod",
   //                  "Train with back propagation steepest descend");
   // AddPreDefVal(TString("SD"));

//   DeclareOptionRef(fLayoutString="TANH|(N+30)*2,TANH|(N+30),LINEAR",    "Layout",    "neural network layout");
// DeclareOptionRef(fLayoutString="RELU|(N+20)*2,RELU|(N+10)*2,LINEAR",    "Layout",    "neural network layout");
   DeclareOptionRef(fLayoutString="SOFTSIGN|(N+100)*2,LINEAR",    "Layout",    "neural network layout");


   DeclareOptionRef(fErrorStrategy="CROSSENTROPY",    "ErrorStrategy",    "error strategy (regression: sum of squares; classification: crossentropy; multiclass: crossentropy/mutual exclusive cross entropy");
   AddPreDefVal(TString("CROSSENTROPY"));
   AddPreDefVal(TString("SUMOFSQUARES"));
   AddPreDefVal(TString("MUTUALEXCLUSIVE"));
   AddPreDefVal(TString("CHECKGRADIENTS"));


   DeclareOptionRef(fWeightInitializationStrategyString="XAVIER",    "WeightInitialization",    "Weight initialization strategy");
   AddPreDefVal(TString("XAVIER"));
   AddPreDefVal(TString("XAVIERUNIFORM"));
   AddPreDefVal(TString("LAYERSIZE"));


   DeclareOptionRef(fTrainingStrategy="LearningRate=1e-1,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=30,TestRepetitions=7,WeightDecay=0.0,Renormalize=L2,DropConfig=0.0,DropRepetitions=5|LearningRate=1e-4,Momentum=0.3,Repetitions=3,ConvergenceSteps=50,BatchSize=20,TestRepetitions=7,WeightDecay=0.001,Renormalize=L2,DropFraction=0.0,DropRepetitions=5",    "TrainingStrategy",    "defines the training strategies");

   DeclareOptionRef(fSumOfSigWeights_test=1000.0,    "SignalWeightsSum",    "Sum of weights of signal; Is used to compute the significance on the fly");
   DeclareOptionRef(fSumOfBkgWeights_test=1000.0,    "BackgroundWeightsSum",    "Sum of weights of background; Is used to compute the significance on the fly");
}


std::vector<std::pair<int,TMVA::NN::EnumFunction>> TMVA::MethodNN::ParseLayoutString(TString layerSpec)
{
    // parse layout specification string and return a vector, each entry
    // containing the number of neurons to go in each successive layer
    std::vector<std::pair<int,TMVA::NN::EnumFunction>> layout;
    const TString delim_Layer (",");
    const TString delim_Sub ("|");

    const size_t inputSize = GetNvar ();

    TObjArray* layerStrings = layerSpec.Tokenize (delim_Layer);
    TIter nextLayer (layerStrings);
    TObjString* layerString = (TObjString*)nextLayer ();
    for (; layerString != NULL; layerString = (TObjString*)nextLayer ())
    {
        int numNodes = 0;
        TMVA::NN::EnumFunction eActivationFunction = NN::EnumFunction::TANH;

        TObjArray* subStrings = layerString->GetString ().Tokenize (delim_Sub);
        TIter nextToken (subStrings);
        TObjString* token = (TObjString*)nextToken ();
        int idxToken = 0;
        for (; token != NULL; token = (TObjString*)nextToken ())
        {
            switch (idxToken)
            {
            case 0:
            {
                TString strActFnc (token->GetString ());
                if (strActFnc == "RELU")
                    eActivationFunction = NN::EnumFunction::RELU;
                else if (strActFnc == "TANH")
                    eActivationFunction = NN::EnumFunction::TANH;
                else if (strActFnc == "SYMMRELU")
                    eActivationFunction = NN::EnumFunction::SYMMRELU;
                else if (strActFnc == "SOFTSIGN")
                    eActivationFunction = NN::EnumFunction::SOFTSIGN;
                else if (strActFnc == "SIGMOID")
                    eActivationFunction = NN::EnumFunction::SIGMOID;
                else if (strActFnc == "LINEAR")
                    eActivationFunction = NN::EnumFunction::LINEAR;
                else if (strActFnc == "GAUSS")
                    eActivationFunction = NN::EnumFunction::GAUSS;
            }
            break;
            case 1: // number of nodes
            {
                TString strNumNodes (token->GetString ());
                TString strN ("x");
                strNumNodes.ReplaceAll ("N", strN);
                strNumNodes.ReplaceAll ("n", strN);
                TFormula fml ("tmp",strNumNodes);
                numNodes = fml.Eval (inputSize);
            }
            break;
            }
            ++idxToken;
        }
        layout.push_back (std::make_pair (numNodes,eActivationFunction));
    }
    return layout;
}



// parse key value pairs in blocks -> return vector of blocks with map of key value pairs
std::vector<std::map<TString,TString>> TMVA::MethodNN::ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim)
{
    std::vector<std::map<TString,TString>> blockKeyValues;
    const TString keyValueDelim ("=");

//    const size_t inputSize = GetNvar ();

    TObjArray* blockStrings = parseString.Tokenize (blockDelim);
    TIter nextBlock (blockStrings);
    TObjString* blockString = (TObjString*)nextBlock ();
    for (; blockString != NULL; blockString = (TObjString*)nextBlock ())
    {
        blockKeyValues.push_back (std::map<TString,TString> ()); // new block
        std::map<TString,TString>& currentBlock = blockKeyValues.back ();

        TObjArray* subStrings = blockString->GetString ().Tokenize (tokenDelim);
        TIter nextToken (subStrings);
        TObjString* token = (TObjString*)nextToken ();
       
        for (; token != NULL; token = (TObjString*)nextToken ())
        {
            TString strKeyValue (token->GetString ());
            int delimPos = strKeyValue.First (keyValueDelim.Data ());
            if (delimPos <= 0)
                continue;

            TString strKey = TString (strKeyValue (0, delimPos));
            strKey.ToUpper ();
            TString strValue = TString (strKeyValue (delimPos+1, strKeyValue.Length ()));

            strKey.Strip (TString::kBoth, ' ');
            strValue.Strip (TString::kBoth, ' ');

            currentBlock.insert (std::make_pair (strKey, strValue));
        }
    }
    return blockKeyValues;
}


TString fetchValue (const std::map<TString, TString>& keyValueMap, TString _key)
{
    TString key (_key);
    key.ToUpper ();
    std::map<TString, TString>::const_iterator it = keyValueMap.find (key);
    if (it == keyValueMap.end ())
        return TString ("");
    return it->second;
}

template <typename T>
T fetchValue (const std::map<TString,TString>& keyValueMap, TString key, T defaultValue);

template <>
int fetchValue (const std::map<TString,TString>& keyValueMap, TString key, int defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    return value.Atoi ();
}

template <>
double fetchValue (const std::map<TString,TString>& keyValueMap, TString key, double defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    return value.Atof ();
}

template <>
TString fetchValue (const std::map<TString,TString>& keyValueMap, TString key, TString defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    return value;
}

template <>
bool fetchValue (const std::map<TString,TString>& keyValueMap, TString key, bool defaultValue)
{
    TString value (fetchValue (keyValueMap, key));
    if (value == "")
        return defaultValue;
    value.ToUpper ();
    if (value == "TRUE" ||
        value == "T" ||
        value == "1")
        return true;
    return false;
}

template <>
std::vector<double> fetchValue (const std::map<TString,TString>& keyValueMap, TString key, std::vector<double> defaultValue)
{
    TString parseString (fetchValue (keyValueMap, key));
    if (parseString == "")
        return defaultValue;
    parseString.ToUpper ();
    std::vector<double> values;

    const TString tokenDelim ("+");
    TObjArray* tokenStrings = parseString.Tokenize (tokenDelim);
    TIter nextToken (tokenStrings);
    TObjString* tokenString = (TObjString*)nextToken ();
    for (; tokenString != NULL; tokenString = (TObjString*)nextToken ())
    {
        std::stringstream sstr;
        double currentValue;
        sstr << tokenString->GetString ().Data ();
        sstr >> currentValue;
        values.push_back (currentValue);
    }
    return values;
}



//_______________________________________________________________________
void TMVA::MethodNN::ProcessOptions()
{
   // process user options
//   MethodBase::ProcessOptions();

   if (fErrorStrategy == "CHECKGRADIENTS") 
       return checkGradients ();


   
   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kINFO 
            << "Will ignore negative events in training!"
            << Endl;
   }

   fLayout = TMVA::MethodNN::ParseLayoutString (fLayoutString);

   //                                                                                         block-delimiter  token-delimiter
   std::vector<std::map<TString,TString>> strategyKeyValues = ParseKeyValueString (fTrainingStrategy, TString ("|"), TString (","));


   if (fWeightInitializationStrategyString == "XAVIER")
       fWeightInitializationStrategy = TMVA::NN::WeightInitializationStrategy::XAVIER;
   if (fWeightInitializationStrategyString == "XAVIERUNIFORM")
       fWeightInitializationStrategy = TMVA::NN::WeightInitializationStrategy::XAVIERUNIFORM;
   else if (fWeightInitializationStrategyString == "LAYERSIZE")
       fWeightInitializationStrategy = TMVA::NN::WeightInitializationStrategy::LAYERSIZE;
   else if (fWeightInitializationStrategyString == "TEST")
       fWeightInitializationStrategy = TMVA::NN::WeightInitializationStrategy::TEST;
   else
       fWeightInitializationStrategy = TMVA::NN::WeightInitializationStrategy::XAVIER;

   // create settings
   if (fAnalysisType == Types::kClassification)
   {

       if (fErrorStrategy == "SUMOFSQUARES") fModeErrorFunction = TMVA::NN::ModeErrorFunction::SUMOFSQUARES;
       if (fErrorStrategy == "CROSSENTROPY") fModeErrorFunction = TMVA::NN::ModeErrorFunction::CROSSENTROPY;
       if (fErrorStrategy == "MUTUALEXCLUSIVE") fModeErrorFunction = TMVA::NN::ModeErrorFunction::CROSSENTROPY_MUTUALEXCLUSIVE;

       for (auto& block : strategyKeyValues)
       {
           size_t convergenceSteps = fetchValue (block, "ConvergenceSteps", 100);
           int batchSize = fetchValue (block, "BatchSize", 30);
           int testRepetitions = fetchValue (block, "TestRepetitions", 7);
           double factorWeightDecay = fetchValue (block, "WeightDecay", 0.0);
           TString regularization = fetchValue (block, "Regularization", TString ("NONE"));
           double learningRate = fetchValue (block, "LearningRate", 1e-5);
           double momentum = fetchValue (block, "Momentum", 0.3);
           int repetitions = fetchValue (block, "Repetitions", 3);
           TString strMultithreading = fetchValue (block, "Multithreading", TString ("True"));
           std::vector<double> dropConfig;
           dropConfig = fetchValue (block, "DropConfig", dropConfig);
           int dropRepetitions = fetchValue (block, "DropRepetitions", 3);

           TMVA::NN::EnumRegularization eRegularization = TMVA::NN::EnumRegularization::NONE;
           if (regularization == "L1")
               eRegularization = TMVA::NN::EnumRegularization::L1;
           else if (regularization == "L2")
               eRegularization = TMVA::NN::EnumRegularization::L2;
           else if (regularization == "L1MAX")
               eRegularization = TMVA::NN::EnumRegularization::L1MAX;


           strMultithreading.ToUpper ();
           bool multithreading = true;
           if (strMultithreading.BeginsWith ("T"))
               multithreading = true;
           else
               multithreading = false;
           
           
           std::shared_ptr<TMVA::NN::ClassificationSettings> ptrSettings = make_shared <TMVA::NN::ClassificationSettings> (
               GetName  (),
               convergenceSteps, batchSize, 
               testRepetitions, factorWeightDecay,
               eRegularization, fScaleToNumEvents, TMVA::NN::MinimizerType::fSteepest,
               learningRate, 
               momentum, repetitions, multithreading);

           if (dropRepetitions > 0 && !dropConfig.empty ())
           {
               ptrSettings->setDropOut (std::begin (dropConfig), std::end (dropConfig), dropRepetitions);
           }
           
           ptrSettings->setWeightSums (fSumOfSigWeights_test, fSumOfBkgWeights_test);
           fSettings.push_back (ptrSettings);
       }
   }
   // else if (fAnalysisType == Types::kMulticlass)
   // {
   //     ptrSettings = std::make_unique <MulticlassSettings> ((*itSetting).convergenceSteps, (*itSetting).batchSize, 
   //                                                          (*itSetting).testRepetitions, (*itSetting).factorWeightDecay,
   //                                                          (*itSetting).isL1, (*itSetting).dropFraction, (*itSetting).dropRepetitions,
   //                                                          fScaleToNumEvents); 
   // }
   // else if (fAnalysisType == Types::kRegression)
   // {
   //     ptrSettings = std::make_unique <RegressionSettings> ((*itSetting).convergenceSteps, (*itSetting).batchSize, 
   //                                                          (*itSetting).testRepetitions, (*itSetting).factorWeightDecay,
   //                                                          (*itSetting).isL1, (*itSetting).dropFraction, (*itSetting).dropRepetitions,
   //                                                          fScaleToNumEvents);
   // }



}

//______________________________________________________________________________
void TMVA::MethodNN::Train()
{
    
    fMonitoring = NULL;
    // if (!fMonitoring)
    // {
    //     fMonitoring = make_shared<Monitoring>();
    //     fMonitoring->Start ();
    // }

    // INITIALIZATION
    // create pattern
    std::vector<Pattern> trainPattern;
    std::vector<Pattern> testPattern;

    const std::vector<TMVA::Event*>& eventCollectionTraining = GetEventCollection (Types::kTraining);
    const std::vector<TMVA::Event*>& eventCollectionTesting  = GetEventCollection (Types::kTesting);

    for (size_t iEvt = 0, iEvtEnd = eventCollectionTraining.size (); iEvt < iEvtEnd; ++iEvt)
    {
        const TMVA::Event* event = eventCollectionTraining.at (iEvt);
        const std::vector<Float_t>& values  = event->GetValues  ();
        if (fAnalysisType == Types::kClassification)
        {
            double outputValue = event->GetClass () == 0 ? 0.1 : 0.9;
            trainPattern.push_back (Pattern (values.begin  (), values.end (), outputValue, event->GetWeight ()));
            trainPattern.back ().addInput (1.0); // bias node
        }
        else
        {
            const std::vector<Float_t>& targets = event->GetTargets ();
            trainPattern.push_back (Pattern (values.begin  (), values.end (), targets.begin (), targets.end (), event->GetWeight ()));
            trainPattern.back ().addInput (1.0); // bias node
        }
    }

    for (size_t iEvt = 0, iEvtEnd = eventCollectionTesting.size (); iEvt < iEvtEnd; ++iEvt)
    {
        const TMVA::Event* event = eventCollectionTesting.at (iEvt);
        const std::vector<Float_t>& values  = event->GetValues  ();
        if (fAnalysisType == Types::kClassification)
        {
            double outputValue = event->GetClass () == 0 ? 0.1 : 0.9;
            testPattern.push_back (Pattern (values.begin  (), values.end (), outputValue, event->GetWeight ()));
            testPattern.back ().addInput (1.0); // bias node
        }
        else
        {
            const std::vector<Float_t>& targets = event->GetTargets ();
            testPattern.push_back (Pattern (values.begin  (), values.end (), targets.begin (), targets.end (), event->GetWeight ()));
            testPattern.back ().addInput (1.0); // bias node
        }
    }

    if (trainPattern.empty () || testPattern.empty ())
        return;

    // create net and weights
    fNet.clear ();
    fWeights.clear ();

    // if "resume" from saved weights
    if (fResume)
    {
        std::cout << ".. resume" << std::endl;
//        std::tie (fNet, fWeights) = ReadWeights (fFileName);
    }
    else // initialize weights and net
    {
        size_t inputSize = GetNVariables (); //trainPattern.front ().input ().size ();
        size_t outputSize = fAnalysisType == Types::kClassification ? 1 : GetNTargets (); //trainPattern.front ().output ().size ();
        fNet.setInputSize (inputSize + 1); // num vars + bias node
//        fNet.setOutputSize (outputSize); // num vars + bias node
        
        // configure neural net
        auto itLayout = std::begin (fLayout), itLayoutEnd = std::end (fLayout)-1; // all layers except the last one
        for ( ; itLayout != itLayoutEnd; ++itLayout)
        {
            fNet.addLayer (NN::Layer ((*itLayout).first, (*itLayout).second)); 
            Log() << kINFO 
                  << "Add Layer with " << (*itLayout).first << " nodes." 
                  << Endl;
        }

        fNet.addLayer (NN::Layer (outputSize, (*itLayout).second, NN::ModeOutputValues::SIGMOID)); 
        Log() << kINFO 
              << "Add Layer with " << outputSize << " nodes." 
              << Endl << Endl;
        fNet.setErrorFunction (fModeErrorFunction); 

        size_t numWeights = fNet.numWeights (inputSize);
        Log() << kINFO 
              << "Total number of Synapses = " 
              << numWeights
              << Endl;

        // initialize weights
        fNet.initializeWeights (fWeightInitializationStrategy, 
                                trainPattern.begin (),
                                trainPattern.end (), 
                                std::back_inserter (fWeights));
    }


    // loop through settings 
    // and create "settings" and minimizer 
    int idxSetting = 0;
    for (auto itSettings = std::begin (fSettings), itSettingsEnd = std::end (fSettings); itSettings != itSettingsEnd; ++itSettings, ++idxSetting)
    {
        std::shared_ptr<TMVA::NN::Settings> ptrSettings = *itSettings;
        ptrSettings->setMonitoring (fMonitoring);
        Log() << kINFO
              << "Training with learning rate = " << ptrSettings->learningRate ()
              << ", momentum = " << ptrSettings->momentum ()
              << ", repetitions = " << ptrSettings->repetitions ()
              << Endl;

        ptrSettings->setProgressLimits ((idxSetting)*100.0/(fSettings.size ()), (idxSetting+1)*100.0/(fSettings.size ()));

        const std::vector<double>& dropConfig = ptrSettings->dropFractions ();
        if (!dropConfig.empty ())
        {
            Log () << kINFO << "Drop configuration" << Endl
                   << "    drop repetitions = " << ptrSettings->dropRepetitions () << Endl;
        }
        int idx = 0;
        for (auto f : dropConfig)
        {
            Log () << kINFO << "    Layer " << idx << " = " << f << Endl;
            ++idx;
        }
        Log () << kINFO << Endl;
        
        if (ptrSettings->minimizerType () == TMVA::NN::MinimizerType::fSteepest)
        {
            NN::Steepest minimizer (ptrSettings->learningRate (), ptrSettings->momentum (), ptrSettings->repetitions ());
            /*E =*/fNet.train (fWeights, trainPattern, testPattern, minimizer, *ptrSettings.get ());
        }
        ptrSettings.reset ();
    }
    fMonitoring = 0;
}





//_______________________________________________________________________
Double_t TMVA::MethodNN::GetMvaValue( Double_t* /*errLower*/, Double_t* /*errUpper*/ )
{
    if (fWeights.empty ())
        return 0.0;

    const std::vector<Float_t>& inputValues = GetEvent ()->GetValues ();
    std::vector<double> input (inputValues.begin (), inputValues.end ());
    input.push_back (1.0); // bias node
    std::vector<double> output = fNet.compute (input, fWeights);
    if (output.empty ())
        return 0.0;

    return output.at (0);
}






//_______________________________________________________________________
void TMVA::MethodNN::AddWeightsXMLTo( void* parent ) const 
{
   // create XML description of NN classifier
   // for all layers

   void* nn = gTools().xmlengine().NewChild(parent, 0, "Weights");
   void* xmlLayout = gTools().xmlengine().NewChild(nn, 0, "Layout");
   Int_t numLayers = fNet.layers ().size ();
   gTools().xmlengine().NewAttr(xmlLayout, 0, "NumberLayers", gTools().StringFromInt (numLayers) );
   for (Int_t i = 0; i < numLayers; i++) 
   {
       const TMVA::NN::Layer& layer = fNet.layers ().at (i);
       int numNodes = layer.numNodes ();
       char activationFunction = (char)(layer.activationFunction ());
       char outputMode = (char)layer.modeOutputValues ();

       void* layerxml = gTools().xmlengine().NewChild(xmlLayout, 0, "Layer");
       gTools().xmlengine().NewAttr(layerxml, 0, "Connection",    TString("FULL") );
       gTools().xmlengine().NewAttr(layerxml, 0, "Nodes",    gTools().StringFromInt(numNodes) );
       gTools().xmlengine().NewAttr(layerxml, 0, "ActivationFunction",    TString (activationFunction) );
       gTools().xmlengine().NewAttr(layerxml, 0, "OutputMode",    TString (outputMode) );
   }


   void* weightsxml = gTools().xmlengine().NewChild(nn, 0, "Synapses");
   gTools().xmlengine().NewAttr (weightsxml, 0, "InputSize", gTools().StringFromInt((int)fNet.inputSize ()));
//   gTools().xmlengine().NewAttr (weightsxml, 0, "OutputSize", gTools().StringFromInt((int)fNet.outputSize ()));
   gTools().xmlengine().NewAttr (weightsxml, 0, "NumberSynapses", gTools().StringFromInt((int)fWeights.size ()));
   std::stringstream s("");
   s.precision( 16 );
   for (std::vector<double>::const_iterator it = fWeights.begin (), itEnd = fWeights.end (); it != itEnd; ++it)
   {
       s << std::scientific << (*it) << " ";
   }
   gTools().xmlengine().AddRawLine (weightsxml, s.str().c_str());
}


//_______________________________________________________________________
void TMVA::MethodNN::ReadWeightsFromXML( void* wghtnode )
{
   // read MLP from xml weight file
    fNet.clear ();

   void* nn = gTools().GetChild(wghtnode, "Weights");
   if (!nn)
   {
//       std::cout << "no node \"Weights\" in XML, use weightnode" << std::endl;
      nn = wghtnode;
   }
   
   void* xmlLayout = NULL;
   xmlLayout = gTools().GetChild(wghtnode, "Layout");
   if (!xmlLayout)
   {
       std::cout << "no node Layout in XML" << std::endl;
       return;
   }


   
//   std::cout << "read layout from XML" << std::endl;
   void* ch = gTools().xmlengine().GetChild (xmlLayout);
   TString connection;
   UInt_t numNodes;
   TString activationFunction;
   TString outputMode;
   fNet.clear ();
   while (ch) 
   {
      gTools().ReadAttr (ch, "Connection", connection);
      gTools().ReadAttr (ch, "Nodes", numNodes);
      gTools().ReadAttr (ch, "ActivationFunction", activationFunction);
      gTools().ReadAttr (ch, "OutputMode", outputMode);
      ch = gTools().GetNextChild(ch);

      fNet.addLayer (NN::Layer (numNodes, (TMVA::NN::EnumFunction)activationFunction (0), (NN::ModeOutputValues)outputMode (0))); 
   }

//   std::cout << "read weights XML" << std::endl;

   void* xmlWeights  = NULL;
   xmlWeights = gTools().GetChild(wghtnode, "Synapses");
   if (!xmlWeights)
       return;

   Int_t numWeights (0);
   Int_t inputSize (0);
//   Int_t outputSize (0);
   gTools().ReadAttr (xmlWeights, "NumberSynapses", numWeights);
   gTools().ReadAttr (xmlWeights, "InputSize", inputSize);
//   gTools().ReadAttr (xmlWeights, "OutputSize", outputSize);
   fNet.setInputSize (inputSize);
//   fNet.setOutputSize (inputSize);

   const char* content = gTools().GetContent (xmlWeights);
   std::stringstream sstr (content);
   for (Int_t iWeight = 0; iWeight<numWeights; ++iWeight) 
   { // synapses
       Double_t weight;
       sstr >> weight;
       fWeights.push_back (weight);
   }
}


//_______________________________________________________________________
void TMVA::MethodNN::ReadWeightsFromStream( std::istream & /*istr*/)
{
   // // destroy/clear the network then read it back in from the weights file

   // // delete network so we can reconstruct network from scratch

   // TString dummy;

   // // synapse weights
   // Double_t weight;
   // std::vector<Double_t>* weights = new std::vector<Double_t>();
   // istr>> dummy;
   // while (istr>> dummy >> weight) weights->push_back(weight); // use w/ slower write-out

   // ForceWeights(weights);
   

   // delete weights;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodNN::CreateRanking()
{
   // compute ranking of input variables by summing function of weights

   // create the ranking object
   fRanking = new Ranking( GetName(), "Importance" );

   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
       fRanking->AddRank( Rank( GetInputLabel(ivar), 1.0));
   }

   // TNeuron*  neuron;
   // TSynapse* synapse;
   // Double_t  importance, avgVal;
   // TString varName;

   // for (UInt_t ivar = 0; ivar < GetNvar(); ivar++) {

   //    neuron = GetInputNeuron(ivar);
   //    Int_t numSynapses = neuron->NumPostLinks();
   //    importance = 0;
   //    varName = GetInputVar(ivar); // fix this line

   //    // figure out average value of variable i
   //    Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
   //    Statistics( TMVA::Types::kTraining, varName, 
   //                meanS, meanB, rmsS, rmsB, xmin, xmax );

   //    avgVal = (TMath::Abs(meanS) + TMath::Abs(meanB))/2.0;
   //    double meanrms = (TMath::Abs(rmsS) + TMath::Abs(rmsB))/2.;
   //    if (avgVal<meanrms) avgVal = meanrms;      
   //    if (IsNormalised()) avgVal = 0.5*(1 + gTools().NormVariable( avgVal, GetXmin( ivar ), GetXmax( ivar ))); 

   //    for (Int_t j = 0; j < numSynapses; j++) {
   //       synapse = neuron->PostLinkAt(j);
   //       importance += synapse->GetWeight() * synapse->GetWeight();
   //    }
      
   //    importance *= avgVal * avgVal;

   //    fRanking->AddRank( Rank( varName, importance ) );
   // }

   return fRanking;
}






//_______________________________________________________________________
void TMVA::MethodNN::MakeClassSpecific( std::ostream& /*fout*/, const TString& /*className*/ ) const
{
   // write specific classifier response
//   MethodANNBase::MakeClassSpecific(fout, className);
}

//_______________________________________________________________________
void TMVA::MethodNN::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   TString col    = gConfig().WriteOptionsReference() ? TString() : gTools().Color("bold");
   TString colres = gConfig().WriteOptionsReference() ? TString() : gTools().Color("reset");

   Log() << Endl;
   Log() << col << "--- Short description:" << colres << Endl;
   Log() << Endl;
   Log() << "The MLP artificial neural network (ANN) is a traditional feed-" << Endl;
   Log() << "forward multilayer perceptron impementation. The MLP has a user-" << Endl;
   Log() << "defined hidden layer architecture, while the number of input (output)" << Endl;
   Log() << "nodes is determined by the input variables (output classes, i.e., " << Endl;
   Log() << "signal and one background). " << Endl;
   Log() << Endl;
   Log() << col << "--- Performance optimisation:" << colres << Endl;
   Log() << Endl;
   Log() << "Neural networks are stable and performing for a large variety of " << Endl;
   Log() << "linear and non-linear classification problems. However, in contrast" << Endl;
   Log() << "to (e.g.) boosted decision trees, the user is advised to reduce the " << Endl;
   Log() << "number of input variables that have only little discrimination power. " << Endl;
   Log() << "" << Endl;
   Log() << "In the tests we have carried out so far, the MLP and ROOT networks" << Endl;
   Log() << "(TMlpANN, interfaced via TMVA) performed equally well, with however" << Endl;
   Log() << "a clear speed advantage for the MLP. The Clermont-Ferrand neural " << Endl;
   Log() << "net (CFMlpANN) exhibited worse classification performance in these" << Endl;
   Log() << "tests, which is partly due to the slow convergence of its training" << Endl;
   Log() << "(at least 10k training cycles are required to achieve approximately" << Endl;
   Log() << "competitive results)." << Endl;
   Log() << Endl;
   Log() << col << "Overtraining: " << colres
         << "only the TMlpANN performs an explicit separation of the" << Endl;
   Log() << "full training sample into independent training and validation samples." << Endl;
   Log() << "We have found that in most high-energy physics applications the " << Endl;
   Log() << "avaliable degrees of freedom (training events) are sufficient to " << Endl;
   Log() << "constrain the weights of the relatively simple architectures required" << Endl;
   Log() << "to achieve good performance. Hence no overtraining should occur, and " << Endl;
   Log() << "the use of validation samples would only reduce the available training" << Endl;
   Log() << "information. However, if the perrormance on the training sample is " << Endl;
   Log() << "found to be significantly better than the one found with the inde-" << Endl;
   Log() << "pendent test sample, caution is needed. The results for these samples " << Endl;
   Log() << "are printed to standard output at the end of each training job." << Endl;
   Log() << Endl;
   Log() << col << "--- Performance tuning via configuration options:" << colres << Endl;
   Log() << Endl;
   Log() << "The hidden layer architecture for all ANNs is defined by the option" << Endl;
   Log() << "\"HiddenLayers=N+1,N,...\", where here the first hidden layer has N+1" << Endl;
   Log() << "neurons and the second N neurons (and so on), and where N is the number  " << Endl;
   Log() << "of input variables. Excessive numbers of hidden layers should be avoided," << Endl;
   Log() << "in favour of more neurons in the first hidden layer." << Endl;
   Log() << "" << Endl;
   Log() << "The number of cycles should be above 500. As said, if the number of" << Endl;
   Log() << "adjustable weights is small compared to the training sample size," << Endl;
   Log() << "using a large number of training samples should not lead to overtraining." << Endl;
}



//_______________________________________________________________________
void  TMVA::MethodNN::WriteMonitoringHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   Log() << kINFO << "Write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
   BaseDir()->cd();
}




void TMVA::MethodNN::checkGradients ()
{
    size_t inputSize = 1;
    size_t outputSize = 1;

    fNet.clear ();

    fNet.setInputSize (inputSize);
//    fNet.setOutputSize (outputSize);
    fNet.addLayer (NN::Layer (100, NN::EnumFunction::SOFTSIGN)); 
    fNet.addLayer (NN::Layer (30, NN::EnumFunction::SOFTSIGN)); 
    fNet.addLayer (NN::Layer (outputSize, NN::EnumFunction::LINEAR, NN::ModeOutputValues::SIGMOID)); 
    fNet.setErrorFunction (NN::ModeErrorFunction::CROSSENTROPY);
//    net.setErrorFunction (ModeErrorFunction::SUMOFSQUARES);

    size_t numWeights = fNet.numWeights (inputSize);
    std::vector<double> weights (numWeights);
    //weights.at (0) = 1000213.2;

    std::vector<Pattern> pattern;
    for (size_t iPat = 0, iPatEnd = 10; iPat < iPatEnd; ++iPat)
    {
        std::vector<double> input;
        std::vector<double> output;
        for (size_t i = 0; i < inputSize; ++i)
        {
            input.push_back (TMVA::NN::gaussDouble (0.1, 4));
        }
        for (size_t i = 0; i < outputSize; ++i)
        {
            output.push_back (TMVA::NN::gaussDouble (0, 3));
        }
        pattern.push_back (Pattern (input,output));
    }


    NN::Settings settings (TString ("checkGradients"), /*_convergenceSteps*/ 15, /*_batchSize*/ 1, /*_testRepetitions*/ 7, /*_factorWeightDecay*/ 0, /*regularization*/ TMVA::NN::EnumRegularization::NONE);

    size_t improvements = 0;
    size_t worsenings = 0;
    size_t smallDifferences = 0;
    size_t largeDifferences = 0;
    for (size_t iTest = 0; iTest < 1000; ++iTest)
    {
        TMVA::NN::uniform (weights, 0.7);
        std::vector<double> gradients (numWeights, 0);
        NN::Batch batch (begin (pattern), end (pattern));
        NN::DropContainer dropContainer;
        std::tuple<NN::Settings&, NN::Batch&, NN::DropContainer&> settingsAndBatch (settings, batch, dropContainer);
        double E = fNet (settingsAndBatch, weights, gradients);
        std::vector<double> changedWeights;
        changedWeights.assign (weights.begin (), weights.end ());

        int changeWeightPosition = TMVA::NN::randomInt (numWeights);
        double dEdw = gradients.at (changeWeightPosition);
        while (dEdw == 0.0)
        {
            changeWeightPosition = TMVA::NN::randomInt (numWeights);
            dEdw = gradients.at (changeWeightPosition);
        }

        const double gamma = 0.01;
        double delta = gamma*dEdw;
        changedWeights.at (changeWeightPosition) += delta;
        if (dEdw == 0.0)
        {
            std::cout << "dEdw == 0.0 ";
            continue;
        }
        
        assert (dEdw != 0.0);
        double Echanged = fNet (settingsAndBatch, changedWeights);

//	double difference = fabs((E-Echanged) - delta*dEdw);
        double difference = fabs ((E+delta - Echanged)/E);
	bool direction = (E-Echanged)>0 ? true : false;
//	bool directionGrad = delta>0 ? true : false;
        bool isOk = difference < 0.3 && difference != 0;

	if (direction)
	    ++improvements;
	else
	    ++worsenings;

	if (isOk)
	    ++smallDifferences;
	else
	    ++largeDifferences;

        if (true || !isOk)
        {
	    if (!direction)
		std::cout << "=================" << std::endl;
            std::cout << "E = " << E << " Echanged = " << Echanged << " delta = " << delta << "   pos=" << changeWeightPosition << "   dEdw=" << dEdw << "  difference= " << difference << "  dirE= " << direction << std::endl;
        }
        if (isOk)
        {
        }
        else
        {
//            for_each (begin (weights), end (weights), [](double w){ std::cout << w << ", "; });
//            std::cout << std::endl;
//            assert (isOk);
        }
    }
    std::cout << "improvements = " << improvements << std::endl;
    std::cout << "worsenings = " << worsenings << std::endl;
    std::cout << "smallDifferences = " << smallDifferences << std::endl;
    std::cout << "largeDifferences = " << largeDifferences << std::endl;

    std::cout << "check gradients done" << std::endl;
}

