// @(#)root/mlp:$Name:  $:$Id: TMultiLayerPerceptron.cxx,v 1.17 2004/05/26 12:30:31 brun Exp $
// Author: Christophe.Delaere@cern.ch   20/07/03

///////////////////////////////////////////////////////////////////////////
//
// TMultiLayerPerceptron
//
// This class describes a neural network.
// There are facilities to train the network and use the output.
//
// The input layer is made of inactive neurons (returning the 
// normalized input), hidden layers are made of sigmoids and output 
// neurons are linear.
//
// The basic input is a TTree and two (training and test) TEventLists.
// Input and output neurons are assigned a value computed for each event 
// with the same possibilities as for TTree::Draw().
// Events may be weighted individualy or via TTree::SetWeight().
// 6 learning methods are available: kStochastic, kBatch, 
// kSteepestDescent, kRibierePolak, kFletcherReeves and kBFGS.
//
// This implementation, written by C. Delaere,  is *inspired* from 
// the mlpfit package from J.Schwindling et al.
//
///////////////////////////////////////////////////////////////////////////
//BEGIN_HTML <!--
/* -->
<UL>
	<LI><P><A NAME="intro"></A><FONT COLOR="#5c8526">
	<FONT SIZE=4 STYLE="font-size: 15pt">Introduction</FONT></FONT></P>
</UL>
<P>Neural Networks are more and more used in various fields for data
analysis and classification, both for research and commercial
institutions. Some randomly choosen examples are:</P>
<UL>
	<LI><P>image analysis</P>
	<LI><P>financial movements predictions and analysis</P>
	<LI><P>sales forecast and product shipping optimisation</P>
	<LI><P>in particles physics: mainly for classification tasks (signal
	over background discrimination)</P>
</UL>
<P>More than 50% of neural networks are multilayer perceptrons. This
implementation of multilayer perceptrons is inspired from the 
<A HREF="http://schwind.home.cern.ch/schwind/MLPfit.html">MLPfit
package</A> originaly written by Jerome Schwindling. MLPfit remains
one of the fastest tool for neural networks studies, and this ROOT
add-on will not try to compete on that. A clear and flexible Object
Oriented implementation has been choosen over a faster but more
difficult to maintain code. Nevertheless, the time penalty does not 
exceed a factor 2.</P>
<UL>
	<LI><P><A NAME="mlp"></A><FONT COLOR="#5c8526">
	<FONT SIZE=4 STYLE="font-size: 15pt">The
	MLP</FONT></FONT></P>
</UL>
<P>The multilayer perceptron is a simple feed-forward network with
the following structure:</P>
<P ALIGN=CENTER><IMG SRC="gif/mlp.png" NAME="MLP" 
ALIGN=MIDDLE WIDTH=333 HEIGHT=358 BORDER=0>
</P>
<P>It is made of neurons characterized by a bias and weighted links
between them (let's call those links synapses). The input neurons
receive the inputs, normalize them and forward them to the first
hidden layer. 
</P>
<P>Each neuron in any subsequent layer first computes a linear
combination of the outputs of the previous layer. The output of the
neuron is then function of that combination with <I>f</I> being
linear for output neurons or a sigmoid for hidden layers. This is
useful because of two theorems:</P>
<OL>
	<LI><P>A linear combination of sigmoids can approximate any
	continuous function.</P>
	<LI><P>Trained with output = 1 for the signal and 0 for the
	background, the approximated function of inputs X is the probability
	of signal, knowing X.</P>
</OL>
<UL>
	<LI><P><A NAME="lmet"></A><FONT COLOR="#5c8526">
	<FONT SIZE=4 STYLE="font-size: 15pt">Learning
	methods</FONT></FONT></P>
</UL>
<P>The aim of all learning methods is to minimize the total error on
a set of weighted examples. The error is defined as the sum in
quadrature, devided by two, of the error on each individual output
neuron.</P>
<P>In all methods implemented, one needs to compute
the first derivative of that error with respect to the weights.
Exploiting the well-known properties of the derivative, especialy the
derivative of compound functions, one can write:</P>
<UL>
	<LI><P>for a neuton: product of the local derivative with the
	weighted sum on the outputs of the derivatives.</P>
	<LI><P>for a synapse: product of the input with the local derivative
	of the output neuron.</P>
</UL>
<P>This computation is called back-propagation of the errors. A
loop over all examples is called an epoch.</P>
<P>Six learning methods are implemented.</P>
<P><FONT COLOR="#006b6b"><I>Stochastic minimization</I>:</FONT> This
is the most trivial learning method. This is the Robbins-Monro
stochastic approximation applied to multilayer perceptrons. The
weights are updated after each example according to the formula:</P>
<P ALIGN=CENTER>$w_{ij}(t+1) = w_{ij}(t) + \Delta w_{ij}(t)$ 
</P>
<P ALIGN=CENTER>with 
</P>
<P ALIGN=CENTER>$\Delta w_{ij}(t) = - \eta(\d e_p / \d w_{ij} +
\delta) + \epsilon \Deltaw_{ij}(t-1)$</P>
<P>The parameters for this method are Eta, EtaDecay, Delta and
Epsilon.</P>
<P><FONT COLOR="#006b6b"><I>Steepest descent with fixed step size
(batch learning)</I>:</FONT> It is the same as the stochastic
minimization, but the weights are updated after considering all the
examples, with the total derivative dEdw. The parameters for this
method are Eta, EtaDecay, Delta and Epsilon.</P>
<P><FONT COLOR="#006b6b"><I>Steepest descent algorithm</I>: </FONT>Weights
are set to the minimum along the line defined by the gradient. The
only parameter for this method is Tau. Lower tau = higher precision =
slower search. A value Tau = 3 seems reasonable.</P>
<P><FONT COLOR="#006b6b"><I>Conjugate gradients with the
Polak-Ribiere updating formula</I>: </FONT>Weights are set to the
minimum along the line defined by the conjugate gradient. Parameters
are Tau and Reset, which defines the epochs where the direction is
reset to the steepes descent.</P>
<P><FONT COLOR="#006b6b"><I>Conjugate gradients with the
Fletcher-Reeves updating formula</I>: </FONT>Weights are set to the
minimum along the line defined by the conjugate gradient. Parameters
are Tau and Reset, which defines the epochs where the direction is
reset to the steepes descent.</P>
<P><FONT COLOR="#006b6b"><I>Broyden, Fletcher, Goldfarb, Shanno
(BFGS) method</I>:</FONT> Implies the computation of a NxN matrix
computation, but seems more powerful at least for less than 300
weights. Parameters are Tau and Reset, which defines the epochs where
the direction is reset to the steepes descent.</P>
<UL>
	<LI><P><A NAME="use"></A><FONT COLOR="#5c8526">
	<FONT SIZE=4 STYLE="font-size: 15pt">How
	to use it...</FONT></FONT></P></LI>
</UL>
<P><FONT SIZE=3>TMLP is build from 3 classes: TNeuron, TSynapse and
TMultiLayerPerceptron. Only TMultiLayerPerceptron should be used
explicitely by the user.</FONT></P>
<P><FONT SIZE=3>TMultiLayerPerceptron will take examples from a TTree
given in the constructor. The network is described by a simple
string: The input/output layers are defined by giving the expression for 
each neuron, separated by comas. Hidden layers are just described
by the number of neurons. The layers are separated by semicolons.
In addition, output layer formulas can be preceded by '@' (e.g "@out")
if one wants to also normalize the output.
Input and outputs are taken from the TTree given as second argument. 
Expressions are evaluated as for TTree::Draw().
One defines the training and test datasets by TEventLists.</FONT></P>
<P STYLE="margin-left: 2cm"><FONT SIZE=3><SPAN STYLE="background: #e6e6e6">
<U><FONT COLOR="#ff0000">Example</FONT></U><SPAN STYLE="text-decoration: none">:
</SPAN>TMultiLayerPerceptron(&quot;x/F,y/F:10:5:f/F&quot;,inputTree);</SPAN></FONT></P>
<P><FONT SIZE=3>Both the TTree and the TEventLists can be defined in
the constructor, or later with the suited setter method.</FONT></P>
<P><FONT SIZE=3>The learning method is defined using the
TMultiLayerPerceptron::SetLearningMethod() . Learning methods are :</FONT></P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P><FONT SIZE=3>TMultiLayerPerceptron::kStochastic, <BR>
TMultiLayerPerceptron::kBatch,<BR>
TMultiLayerPerceptron::kSteepestDescent,<BR>
TMultiLayerPerceptron::kRibierePolak,<BR>
TMultiLayerPerceptron::kFletcherReeves,<BR>
TMultiLayerPerceptron::kBFGS<BR></FONT></P>
<P>A weight can be assigned to events, either in the constructor, either 
with TMultiLayerPerceptron::SetEventWeight(). In addition, the TTree weight
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
<P>xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<P>
is taken into account.</P>
<P><FONT SIZE=3>Finally, one starts the training with
TMultiLayerPerceptron::Train(Int_t nepoch, Option_t* options). The
first argument is the number of epochs while option is a string that
can contain: &quot;text&quot; (simple text output) , &quot;graph&quot;
(evoluting graphical training curves), &quot;update=X&quot; (step for
the text/graph output update) or &quot;+&quot; (will skip the
randomisation and start from the previous values). All combinations
are available. </FONT></P>
<P STYLE="margin-left: 2cm"><FONT SIZE=3><SPAN STYLE="background: #e6e6e6">
<U><FONT COLOR="#ff0000">Example</FONT></U>:
net.Train(100,&quot;text, graph, update=10&quot;).</SPAN></FONT></P>
<P><FONT SIZE=3>When the neural net is trained, it can be used
directly ( TMultiLayerPerceptron::Evaluate() ) or exported to a
standalone C++ code ( TMultiLayerPerceptron::Export() ).</FONT></P>
<P><FONT SIZE=3>Finaly, note that even if this implementation is inspired from the mlpfit code,
the feature lists are not exactly matching: 
<UL>
        <LI><P>mlpfit hybrid learning method is not implemented</P></LI>
	<LI><P>output neurons can be normalized, this is not the case for mlpfit</P></LI>
	<LI><P>the neural net is exported in C++ (not in FORTRAN)</P></LI>
	<LI><P>the drawResult() method allows a fast check of the learning procedure</P></LI>
</UL>
In addition, the paw version of mlpfit had additional limitations on the number of neurons, hidden layers and inputs/outputs that does not apply to TMultiLayerPerceptron.
<!-- */
// -->END_HTML

#include "TMultiLayerPerceptron.h"
#include "TSynapse.h"
#include "TNeuron.h"
#include "TROOT.h"
#include "TTree.h"
#include "TEventList.h"
#include "TRandom3.h"
#include "TTimeStamp.h"
#include "TRegexp.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TDirectory.h"
#include "TSystem.h"
#include "Riostream.h"
#include "TMath.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TMarker.h"
#include "TLine.h"
#include "TText.h"

ClassImp(TMultiLayerPerceptron)

//______________________________________________________________________________
TMultiLayerPerceptron::TMultiLayerPerceptron()
{
   // Default constructor
   if(!gROOT->GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fData = NULL;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = NULL;
   fTrainingOwner = false;
   fTest = NULL;
   fTestOwner = false;
   fEventWeight = NULL;
   fManager = NULL;
   fLearningMethod = TMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
}

//______________________________________________________________________________
TMultiLayerPerceptron::TMultiLayerPerceptron(const char * layout, TTree * data,
                                             TEventList * training,
                                             TEventList * test)
{
   // The network is described by a simple string: 
   // The input/output layers are defined by giving 
   // the branch names separated by comas.
   // Hidden layers are just described by the number of neurons.
   // The layers are separated by semicolons.
   // Ex: "x,y:10:5:f"
   // The output can be prepended by '@' if the variable has to be
   // normalized.
   // Input and outputs are taken from the TTree given as second argument.
   // training and test are the two TEventLists defining events 
   // to be used during the neural net training. 
   // Both the TTree and the TEventLists  can be defined in the constructor, 
   // or later with the suited setter method.
   
   if(!gROOT->GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = training;
   fTrainingOwner = false;
   fTest = test;
   fTestOwner = false;
   fWeight = "1";
   BuildNetwork();
   if (data)
      AttachData();
   fLearningMethod = TMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
}

//______________________________________________________________________________
TMultiLayerPerceptron::TMultiLayerPerceptron(const char * layout, const char * weight, TTree * data,
                                             TEventList * training,
                                             TEventList * test)
{
   // The network is described by a simple string: 
   // The input/output layers are defined by giving 
   // the branch names separated by comas.
   // Hidden layers are just described by the number of neurons.
   // The layers are separated by semicolons.
   // Ex: "x,y:10:5:f"
   // The output can be prepended by '@' if the variable has to be
   // normalized.
   // Input and outputs are taken from the TTree given as second argument.
   // training and test are the two TEventLists defining events 
   // to be used during the neural net training. 
   // Both the TTree and the TEventLists  can be defined in the constructor, 
   // or later with the suited setter method.
   
   if(!gROOT->GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fTraining = training;
   fTrainingOwner = false;
   fTest = test;
   fTestOwner = false;
   fWeight = weight;
   BuildNetwork();
   if (data)
      AttachData();
   fLearningMethod = TMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
}

//______________________________________________________________________________
TMultiLayerPerceptron::TMultiLayerPerceptron(const char * layout, TTree * data,
                                             const char * training,
                                             const char * test)
{
   // The network is described by a simple string: 
   // The input/output layers are defined by giving 
   // the branch names separated by comas.
   // Hidden layers are just described by the number of neurons.
   // The layers are separated by semicolons.
   // Ex: "x,y:10:5:f"
   // The output can be prepended by '@' if the variable has to be
   // normalized.
   // Input and outputs are taken from the TTree given as second argument.
   // training and test are the two TEventLists defining events 
   // to be used during the neural net training. 
   // Both the TTree and the TEventLists  can be defined in the constructor, 
   // or later with the suited setter method.
   
   if(!gROOT->GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fCurrentTreeWeight = 1;
   fTraining = new TEventList(Form("fTrainingList_%i",this));
   fTrainingOwner = true;
   fTest = new TEventList(Form("fTestList_%i",this));
   fTestOwner = true;
   fWeight = "1";
   TString testcut = test;
   if(testcut=="") testcut = Form("!(%s)",training);
   BuildNetwork();
   if (data) {
      data->Draw(Form(">>fTrainingList_%i",this),training,"goff");
      data->Draw(Form(">>fTestList_%i",this),(const char *)testcut,"goff");
      AttachData();
   }
   else {
      Warning("TMultiLayerPerceptron::TMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
   fLearningMethod = TMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
}

//______________________________________________________________________________
TMultiLayerPerceptron::TMultiLayerPerceptron(const char * layout, const char * weight, TTree * data,
                                             const char * training,
                                             const char * test)
{
   // The network is described by a simple string: 
   // The input/output layers are defined by giving 
   // the branch names separated by comas.
   // Hidden layers are just described by the number of neurons.
   // The layers are separated by semicolons.
   // Ex: "x,y:10:5:f"
   // The output can be prepended by '@' if the variable has to be
   // normalized.
   // Input and outputs are taken from the TTree given as second argument.
   // training and test are the two TEventLists defining events 
   // to be used during the neural net training. 
   // Both the TTree and the TEventLists  can be defined in the constructor, 
   // or later with the suited setter method.
   
   if(!gROOT->GetClass("TTreePlayer")) gSystem->Load("libTreePlayer");
   fNetwork.SetOwner(true);
   fFirstLayer.SetOwner(false);
   fLastLayer.SetOwner(false);
   fSynapses.SetOwner(true);
   fStructure = layout;
   fData = data;
   fCurrentTree = -1;
   fTraining = new TEventList(Form("fTrainingList_%i",this));
   fTrainingOwner = true;
   fTest = new TEventList(Form("fTestList_%i",this));
   fTestOwner = true;
   fWeight = weight;
   TString testcut = test;
   if(testcut=="") testcut = Form("!(%s)",training);
      BuildNetwork();
   if (data) {
      data->Draw(Form(">>fTrainingList_%i",this),training,"goff");
      data->Draw(Form(">>fTestList_%i",this),(const char *)testcut,"goff");
      AttachData();
   }
   else {
      Warning("TMultiLayerPerceptron::TMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
   fLearningMethod = TMultiLayerPerceptron::kBFGS;
   fEta = .1;
   fEtaDecay = 1;
   fDelta = 0;
   fEpsilon = 0;
   fTau = 3;
   fLastAlpha = 0;
   fReset = 50;
}

//______________________________________________________________________________
TMultiLayerPerceptron::~TMultiLayerPerceptron() { 
  if(fTraining && fTrainingOwner) delete fTraining;
  if(fTest && fTestOwner) delete fTest;
}
 
//______________________________________________________________________________
void TMultiLayerPerceptron::SetData(TTree * data)
{
   // Set the data source
   if (fData) {
      cerr << "Error: data already defined." << endl;
      return;
   }
   fData = data;
   if (data)
      AttachData();
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetEventWeight(const char * branch)
{
   // Set the event weight
   fWeight=branch;
   if (fData) {
      if (fEventWeight) {
         fManager->Remove(fEventWeight);
         delete fEventWeight;
      }
      fManager->Add((fEventWeight = new TTreeFormula("NNweight",fWeight.Data(),fData)));
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetTrainingDataSet(TEventList* train) 
{ 
   // Sets the Training dataset. 
   // Those events will be used for the minimization
   if(fTraining && fTrainingOwner) delete fTraining;
   fTraining = train; 
   fTrainingOwner = false;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetTestDataSet(TEventList* test) 
{
   // Sets the Test dataset.
   // Those events will not be used for the minimization but for control	   
   if(fTest && fTestOwner) delete fTest;
   fTest = test; 
   fTestOwner = false;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetTrainingDataSet(const char * train) 
{ 
   // Sets the Training dataset. 
   // Those events will be used for the minimization.
   // Note that the tree must be already defined.
   if(fTraining && fTrainingOwner) delete fTraining;
   fTraining = new TEventList(Form("fTrainingList_%i",this));
   fTrainingOwner = true;
   if (fData) {
      fData->Draw(Form(">>fTrainingList_%i",this),train,"goff");
   }
   else {
      Warning("TMultiLayerPerceptron::TMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetTestDataSet(const char * test) 
{
   // Sets the Test dataset.
   // Those events will not be used for the minimization but for control. 
   // Note that the tree must be already defined.
   if(fTest && fTestOwner) delete fTest;
   if(fTest) if(strncmp(fTest->GetName(),Form("fTestList_%i",this),10)) delete fTest;
   fTest = new TEventList(Form("fTestList_%i",this));
   fTestOwner = true;
   if (fData) {
      fData->Draw(Form(">>fTestList_%i",this),test,"goff");
   }
   else {
      Warning("TMultiLayerPerceptron::TMultiLayerPerceptron","Data not set. Cannot define datasets");
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetLearningMethod(TMultiLayerPerceptron::LearningMethod method)
{
   // Sets the learning method.
   // Available methods are: kStochastic, kBatch,
   // kSteepestDescent, kRibierePolak, kFletcherReeves and kBFGS.
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fLearningMethod = method;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetEta(Double_t eta) 
{
   // Sets Eta - used in stochastic minimisation
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fEta = eta; 
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetEpsilon(Double_t eps) 
{ 
   // Sets Epsilon - used in stochastic minimisation
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fEpsilon = eps; 
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetDelta(Double_t delta) 
{ 
   // Sets Delta - used in stochastic minimisation
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fDelta = delta; 
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetEtaDecay(Double_t ed) 
{ 
   // Sets EtaDecay - Eta *= EtaDecay at each epoch
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fEtaDecay = ed; 
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetTau(Double_t tau) 
{ 
   // Sets Tau - used in line search
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fTau = tau; 
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetReset(Int_t reset) 
{ 
   // Sets number of epochs between two resets of the 
   // search direction to the steepest descent.
   // (look at the constructor for the complete description 
   // of learning methods and parameters)
   fReset = reset; 
}

//______________________________________________________________________________
void TMultiLayerPerceptron::GetEntry(Int_t entry) const
{
   // Load an entry into the network
   if (fData)
      fData->GetEntry(entry);
   if (fData->GetTreeNumber() != fCurrentTree) {
      ((TMultiLayerPerceptron*)this)->fCurrentTree = fData->GetTreeNumber();
      fManager->Notify();
      ((TMultiLayerPerceptron*)this)->fCurrentTreeWeight = fData->GetWeight();
   }
   Int_t nentries = fNetwork.GetEntriesFast();
   for (Int_t i=0;i<nentries;i++) {
      TNeuron *neuron = (TNeuron *)fNetwork.UncheckedAt(i);
      neuron->SetNewEvent();
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::Train(Int_t nEpoch, Option_t * option)
{
   // Train the network.
   // nEpoch is the number of iterations.
   // option can contain:
   // - "text" (simple text output)
   // - "graph" (evoluting graphical training curves)
   // - "update=X" (step for the text/graph output update)
   // - "+" will skip the randomisation and start from the previous values. 
   // All combinations are available. 
   
   Int_t i;
   TString opt = option;
   opt.ToLower();
   // Decode options and prepare training.
   Int_t verbosity = 0;
   if (opt.Contains("text"))
      verbosity += 1;
   if (opt.Contains("graph"))
      verbosity += 2;
   Int_t DisplayStepping = 1;
   if (opt.Contains("update=")) {
      TRegexp reg("update=[0-9]*");
      TString out = opt(reg);
      DisplayStepping = atoi(out.Data() + 7);
   }
   TCanvas *canvas = NULL;
   TMultiGraph *residual_plot = NULL;
   TGraph *train_residual_plot = NULL;
   TGraph *test_residual_plot = NULL;
   if ((!fData) || (!fTraining) || (!fTest)) {
      Error("Train","Training/Test samples still not defined. Cannot train the neural network");
      return;
   }
   // Text and Graph outputs
   if (verbosity % 2)
      cout << "Training the Neural Network" << endl;
   if (verbosity / 2) {
      residual_plot = new TMultiGraph;
      canvas = new TCanvas("NNtraining", "Neural Net training");
      Double_t *epoch_axis = new Double_t[nEpoch];
      Double_t *train_residual = new Double_t[nEpoch];
      Double_t *test_residual = new Double_t[nEpoch];
      for (i = 0; i < nEpoch; i++) {
         epoch_axis[i] = i;
         train_residual[i] = 0;
         test_residual[i] = 0;
      }
      train_residual_plot = new TGraph(nEpoch, epoch_axis, train_residual);
      test_residual_plot = new TGraph(nEpoch, epoch_axis, test_residual);
      delete[]train_residual;
      delete[]test_residual;
      delete[]epoch_axis;
      train_residual_plot->SetLineColor(4);
      test_residual_plot->SetLineColor(2);
      residual_plot->Add(train_residual_plot);
      residual_plot->Add(test_residual_plot);
      residual_plot->Draw("LA");
      residual_plot->GetXaxis()->SetTitle("Epoch");
      residual_plot->GetYaxis()->SetTitle("Error");
      residual_plot->Draw("LA");
      canvas->Update();
   }
   // If the option "+" is not set, one has to randomize the weights first
   if (!opt.Contains("+"))
      Randomize();
   // Initialisation
   fLastAlpha = 0;
   Int_t els = fNetwork.GetEntriesFast() + fSynapses.GetEntriesFast();
   Double_t *buffer = new Double_t[els];
   Double_t *dir = new Double_t[els];
   for (i = 0; i < els; i++)
      buffer[i] = 0;
   TMatrixD BFGSH(els, els);
   TMatrixD gamma(els, 1);
   TMatrixD delta(els, 1);
   // Epoch loop. Here is the training itself.
   for (Int_t iepoch = 0; iepoch < nEpoch; iepoch++) {
      switch (fLearningMethod) {
      case TMultiLayerPerceptron::kStochastic:
         {
            MLP_Stochastic(buffer);
            break;
         }
      case TMultiLayerPerceptron::kBatch:
         {
            ComputeDEDw();
            MLP_Batch(buffer);
            break;
         }
      case TMultiLayerPerceptron::kSteepestDescent:
         {
            ComputeDEDw();
            SteepestDir(dir);
            if (LineSearch(dir, buffer))
               MLP_Batch(buffer);
            break;
         }
      case TMultiLayerPerceptron::kRibierePolak:
         {
            ComputeDEDw();
            if (!(iepoch % fReset)) {
               SteepestDir(dir);
            } else {
               Double_t norm = 0;
               Double_t Onorm = 0;
               for (i = 0; i < els; i++)
                  Onorm += dir[i] * dir[i];
               Double_t prod = 0;
               Int_t idx = 0;
               TNeuron *neuron = NULL;
               TSynapse *synapse = NULL;
               Int_t nentries = fNetwork.GetEntriesFast();
               for (i=0;i<nentries;i++) {
                  neuron = (TNeuron *) fNetwork.UncheckedAt(i);
                  prod -= dir[idx++] * neuron->GetDEDw();
                  norm += neuron->GetDEDw() * neuron->GetDEDw();
               }
               nentries = fSynapses.GetEntriesFast();
               for (i=0;i<nentries;i++) {
                  synapse = (TSynapse *) fSynapses.UncheckedAt(i);
                  prod -= dir[idx++] * synapse->GetDEDw();
                  norm += synapse->GetDEDw() * synapse->GetDEDw();
               }
               ConjugateGradientsDir(dir, (norm - prod) / Onorm);
            }
            if (LineSearch(dir, buffer))
               MLP_Batch(buffer);
            break;
         }
      case TMultiLayerPerceptron::kFletcherReeves:
         {
            ComputeDEDw();
            if (!(iepoch % fReset)) {
               SteepestDir(dir);
            } else {
               Double_t norm = 0;
               Double_t Onorm = 0;
               for (i = 0; i < els; i++)
                  Onorm += dir[i] * dir[i];
               TNeuron *neuron = NULL;
               TSynapse *synapse = NULL;
               Int_t nentries = fNetwork.GetEntriesFast();
               for (i=0;i<nentries;i++) {
                  neuron = (TNeuron *) fNetwork.UncheckedAt(i);
                  norm += neuron->GetDEDw() * neuron->GetDEDw();
               }
               nentries = fSynapses.GetEntriesFast();
               for (i=0;i<nentries;i++) {
                  synapse = (TSynapse *) fSynapses.UncheckedAt(i);
                  norm += synapse->GetDEDw() * synapse->GetDEDw();
               }
               ConjugateGradientsDir(dir, norm / Onorm);
            }
            if (LineSearch(dir, buffer))
               MLP_Batch(buffer);
            break;
         }
      case TMultiLayerPerceptron::kBFGS:
         {
            SetGammaDelta(gamma, delta, buffer);
            if (!(iepoch % fReset)) {
               SteepestDir(dir);
               BFGSH.UnitMatrix();
            } else {
               if (GetBFGSH(BFGSH, gamma, delta)) {
                  SteepestDir(dir);
                  BFGSH.UnitMatrix();
               } else {
                  BFGSDir(BFGSH, dir);
               }
            }
            if (DerivDir(dir) > 0) {
               SteepestDir(dir);
               BFGSH.UnitMatrix();
            }
            if (LineSearch(dir, buffer)) {
               BFGSH.UnitMatrix();
               SteepestDir(dir);
               if (LineSearch(dir, buffer)) {
                  Error("TMultiLayerPerceptron::Train()","Line search fail");
                  iepoch = nEpoch;
               }
            }
            break;
         }
      }
      // Security: would the learning lead to non real numbers, 
      // the learning should stop now.
      if (isnan(GetError(TMultiLayerPerceptron::kTraining))) {
         Error("TMultiLayerPerceptron::Train()","Stop.");
         iepoch = nEpoch;
      }
      // Process other ROOT events.  Time penalty is less than 
      // 1/1000 sec/evt on a mobile AMD Athlon(tm) XP 1500+
      gSystem->ProcessEvents();
      // Intermediate graph and text output
      if ((verbosity % 2) && ((!(iepoch % DisplayStepping)) 
          || (iepoch == nEpoch - 1)))
         cout << "Epoch: " << iepoch 
		   << " learn=" 
                   << TMath::Sqrt(GetError(TMultiLayerPerceptron::kTraining) 
                      / fTraining->GetN())
                   << " test=" 
		   << TMath::Sqrt(GetError(TMultiLayerPerceptron::kTest) 
                      / fTest->GetN())
                   << endl;
      if (verbosity / 2) {
         train_residual_plot->SetPoint(iepoch, iepoch,
           TMath::Sqrt(GetError(TMultiLayerPerceptron::kTraining) 
                       / fTraining->GetN()));
         test_residual_plot->SetPoint(iepoch, iepoch,
           TMath::Sqrt(GetError(TMultiLayerPerceptron::kTest) 
                       / fTest->GetN()));
         if (!iepoch) {
            Double_t trp = train_residual_plot->GetY()[iepoch];
            Double_t tep = test_residual_plot->GetY()[iepoch];
            for (i = 1; i < nEpoch; i++) {
               train_residual_plot->SetPoint(i, i, trp);
               test_residual_plot->SetPoint(i, i, tep);
            }
         }
         if ((!(iepoch % DisplayStepping)) || (iepoch == nEpoch - 1)) {
            residual_plot->Draw("LA");
            residual_plot->GetYaxis()->UnZoom();
            canvas->Update();
         }
      }
   }
   // Cleaning
   delete[]buffer;
   delete[]dir;
   // Final Text and Graph outputs
   if (verbosity % 2)
      cout << "Training done." << endl;
   if (verbosity / 2) {
      TLegend *legend = new TLegend(.75, .80, .95, .95);
      legend->AddEntry(residual_plot->GetListOfGraphs()->At(0),
                       "Training sample", "L");
      legend->AddEntry(residual_plot->GetListOfGraphs()->At(1),
                       "Test sample", "L");
      legend->Draw();
   }
}

//______________________________________________________________________________
Double_t TMultiLayerPerceptron::Result(Int_t event, Int_t index) const
{
   // Computes the output for a given event. 
   // Look at the output neuron designed by index.
   GetEntry(event);
   TNeuron *out = (TNeuron *) (fLastLayer.At(index));
   if (out)
      return out->GetValue();
   else
      return 0;
}

//______________________________________________________________________________
Double_t TMultiLayerPerceptron::GetError(Int_t event) const
{
   // Error on the output for a given event
   GetEntry(event);
   Double_t error = 0;
   for (Int_t i = 0; i < fLastLayer.GetEntriesFast(); i++) {
      error += (((TNeuron *) fLastLayer[i])->GetError() *
                ((TNeuron *) fLastLayer[i])->GetError());
   }
   error /= 2.;
   error *= fEventWeight->EvalInstance();
   error *= fCurrentTreeWeight;
   return error;
}

//______________________________________________________________________________
Double_t TMultiLayerPerceptron::GetError(TMultiLayerPerceptron::DataSet set) const
{
   // Error on the whole dataset
   TEventList *list =
       ((set == TMultiLayerPerceptron::kTraining) ? fTraining : fTest);
   Double_t error = 0;
   Int_t i;
   if (list) {
      Int_t nEvents = list->GetN();
      for (i = 0; i < nEvents; i++) {
         error += GetError(list->GetEntry(i));
      }
   } else if (fData) {
      Int_t nEvents = (Int_t) fData->GetEntries();
      for (i = 0; i < nEvents; i++) {
         error += GetError(i);
      }
   }
   return error;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::ComputeDEDw() const
{
   // Compute the DEDw = sum on all training events of dedw for each weight
   // normalized by the number of events.
   Int_t i,j;
   Int_t nentries = fSynapses.GetEntriesFast();
   TSynapse *synapse;
   for (i=0;i<nentries;i++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(i);
      synapse->SetDEDw(0.);
   }
   TNeuron *neuron;
   nentries = fNetwork.GetEntriesFast();
   for (i=0;i<nentries;i++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(i);
      neuron->SetDEDw(0.);
   }
   Double_t eventWeight = 1.;
   if (fTraining) {
      Int_t nEvents = fTraining->GetN();
      for (i = 0; i < nEvents; i++) {
         GetEntry(fTraining->GetEntry(i));
	 eventWeight = fEventWeight->EvalInstance();
	 eventWeight *= fCurrentTreeWeight;
         nentries = fSynapses.GetEntriesFast();
         for (j=0;j<nentries;j++) {
            synapse = (TSynapse *) fSynapses.UncheckedAt(j);
            synapse->SetDEDw(synapse->GetDEDw() + (synapse->GetDeDw()*eventWeight));
         }
         nentries = fNetwork.GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuron = (TNeuron *) fNetwork.UncheckedAt(j);
            neuron->SetDEDw(neuron->GetDEDw() + (neuron->GetDeDw()*eventWeight));
         }
      }
      nentries = fSynapses.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (TSynapse *) fSynapses.UncheckedAt(j);
         synapse->SetDEDw(synapse->GetDEDw() / (Double_t) nEvents);
      }
      nentries = fNetwork.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (TNeuron *) fNetwork.UncheckedAt(j);
         neuron->SetDEDw(neuron->GetDEDw() / (Double_t) nEvents);
      }
   } else if (fData) {
      Int_t nEvents = (Int_t) fData->GetEntries();
      for (i = 0; i < nEvents; i++) {
         GetEntry(i);
	 eventWeight = fEventWeight->EvalInstance();
	 eventWeight *= fCurrentTreeWeight;
         nentries = fSynapses.GetEntriesFast();
         for (j=0;j<nentries;j++) {
            synapse = (TSynapse *) fSynapses.UncheckedAt(j);
            synapse->SetDEDw(synapse->GetDEDw() + (synapse->GetDeDw()*eventWeight));
         }
         nentries = fNetwork.GetEntriesFast();
         for (j=0;j<nentries;j++) {
            neuron = (TNeuron *) fNetwork.UncheckedAt(j);
            neuron->SetDEDw(neuron->GetDEDw() + (neuron->GetDeDw()*eventWeight));
         }
      }
      nentries = fSynapses.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (TSynapse *) fSynapses.UncheckedAt(j);
         synapse->SetDEDw(synapse->GetDEDw() / (Double_t) nEvents);
      }
      nentries = fNetwork.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (TNeuron *) fNetwork.UncheckedAt(j);
         neuron->SetDEDw(neuron->GetDEDw() / (Double_t) nEvents);
      }
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::Randomize() const
{
   // Randomize the weights
   Int_t nentries = fSynapses.GetEntriesFast();
   Int_t j;
   TSynapse *synapse;
   TNeuron *neuron;
   TTimeStamp ts;
   TRandom3 gen(ts.GetSec());
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      synapse->SetWeight(gen.Rndm() - 0.5);
   }
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      neuron->SetWeight(gen.Rndm() - 0.5);
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::AttachData()
{
   // Connects the TTree to Neurons in input and output 
   // layers. The formulas associated to each neuron are created
   // and reported to the network formula manager.
   // By default, the branch is not normalised since this would degrade 
   // performance for classification jobs.
   // Normalisation can be requested by putting '@' in front of the formula.
   Int_t j = 0;
   Int_t beg = 0;
   TString brName;
   TNeuron *neuron = NULL;
   Bool_t normalize = false;
   fManager = new TTreeFormulaManager;
   //first layer
   TString input  = TString(fStructure(0, fStructure.First(':')));
   Int_t end = input.Index(",", beg + 1);
   Int_t nentries = fFirstLayer.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      end = (end == -1 ? input.Length() : end);
      brName = TString(input(beg, end - beg));
      neuron = (TNeuron *) fFirstLayer.UncheckedAt(j);
      fManager->Add(neuron->UseBranch(fData,brName.Data()));
      beg = end + 1;
      end = input.Index(",", beg + 1);
   }
   // last layer
   TString output = TString(
           fStructure(fStructure.Last(':') + 1,
                      fStructure.Length() - fStructure.Last(':')));
   j = 0;
   beg = 0;
   end = output.Index(",", beg + 1);
   nentries = fLastLayer.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      end = (end == -1 ? output.Length() : end);
      normalize = false;
      brName = TString(output(beg, end - beg));
      neuron = (TNeuron *) fLastLayer.UncheckedAt(j);
      if (brName[0]=='@')
         normalize = true;
      fManager->Add(neuron->UseBranch(fData,brName.Data() + (normalize?1:0)));
      if(!normalize) neuron->SetNormalisation(0., 1.);
      beg = end + 1;
      end = output.Index(",", beg + 1);
   }
   if(fManager->GetMultiplicity()>1)
      Warning("BuildNetwork()","all indices in arrays must be specified.\nFirst element will be assumed.");
   fManager->Add((fEventWeight = new TTreeFormula("NNweight",fWeight.Data(),fData)));
   fManager->Sync();
}

//______________________________________________________________________________
void TMultiLayerPerceptron::BuildNetwork()
{
   // Instanciates the network from the description
   TString input  = TString(fStructure(0, fStructure.First(':')));
   TString hidden = TString(
           fStructure(fStructure.First(':') + 1,
                      fStructure.Last(':') - fStructure.First(':') - 1));
   TString output = TString(
           fStructure(fStructure.Last(':') + 1,
                      fStructure.Length() - fStructure.Last(':')));
   Int_t bll = atoi(TString(
           hidden(hidden.Last(':') + 1,
                  hidden.Length() - (hidden.Last(':') + 1))).Data());
   if (input.Length() == 0) {
      Error("BuildNetwork()","malformed structure. No input layer.");
      return;
   }
   if (output.Length() == 0) {
      Error("BuildNetwork()","malformed structure. No output layer.");
      return;
   }
   BuildFirstLayer(input);
   BuildHiddenLayers(hidden);
   BuildLastLayer(output, bll);
}

//______________________________________________________________________________
void TMultiLayerPerceptron::BuildFirstLayer(TString & input)
{
   // Instanciates the neurons in input
   // Inputs are normalised and the type is set to kOff 
   // (simple forward of the formula value)
 
   Int_t nneurons = input.CountChar(',')+1;
   TNeuron *neuron = NULL;
   Int_t i = 0;
   for (i = 0; i<nneurons; i++) {
      neuron = new TNeuron(TNeuron::kOff);
      fFirstLayer.AddLast(neuron);
      fNetwork.AddLast(neuron);
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::BuildHiddenLayers(TString & hidden)
{
   // Builds hidden layers.
   // Neurons are Sigmoids.
   Int_t beg = 0;
   Int_t end = hidden.Index(":", beg + 1);
   Int_t prevStart = 0;
   Int_t prevStop = fNetwork.GetEntriesFast();
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   Int_t i,j;
   while (end != -1) {
      Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
      for (i = 0; i < num; i++) {
         neuron = new TNeuron(TNeuron::kSigmoid);
         fNetwork.AddLast(neuron);
         for (j = prevStart; j < prevStop; j++) {
            synapse = new TSynapse((TNeuron *) fNetwork[j], neuron);
            fSynapses.AddLast(synapse);
         }
      }
      beg = end + 1;
      end = hidden.Index(":", beg + 1);
      prevStart = prevStop;
      prevStop = fNetwork.GetEntriesFast();
   }
   Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
   for (i = 0; i < num; i++) {
      neuron = new TNeuron(TNeuron::kSigmoid);
      fNetwork.AddLast(neuron);
      for (j = prevStart; j < prevStop; j++) {
         synapse = new TSynapse((TNeuron *) fNetwork[j], neuron);
         fSynapses.AddLast(synapse);
      }
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::BuildLastLayer(TString & output, Int_t prev)
{
   // Builds the output layer
   // Neurons are linear combinations of input.
   Int_t nneurons = output.CountChar(',')+1;
   Int_t prevStop = fNetwork.GetEntriesFast();
   Int_t prevStart = prevStop - prev;
   TNeuron *neuron;
   TSynapse *synapse;
   Int_t i,j;
   for (i = 0; i<nneurons; i++) {
      neuron = new TNeuron(TNeuron::kLinear);
      for (j = prevStart; j < prevStop; j++) {
         synapse = new TSynapse((TNeuron *) fNetwork[j], neuron);
         fSynapses.AddLast(synapse);
      }
      fLastLayer.AddLast(neuron);
      fNetwork.AddLast(neuron);
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::DrawResult(Int_t index, Option_t * option) const
{
   // Draws the neural net output
   // It produces an histogram with the output for the two datasets.
   // Index is the number of the desired output neuron.
   // "option" can contain:
   // - test or train to select a dataset
   // - comp to produce a X-Y comparison plot
   
   TString opt = option;
   opt.ToLower();
   TNeuron *out = (TNeuron *) (fLastLayer.At(index));
   if (!out) {
      Error("DrawResult()","no such output.");
      return;
   }
   //TCanvas *canvas = new TCanvas("NNresult", "Neural Net output");
   new TCanvas("NNresult", "Neural Net output");
   const Double_t *norm = out->GetNormalisation();
   TEventList *events = NULL;
   TString setname;
   Int_t i;
   if (opt.Contains("train")) {
      events = fTraining;
      setname = "train";
   } else if (opt.Contains("test")) {
      events = fTest;
      setname = "test";
   }
   if ((!fData) || (!events)) {
      Error("DrawResult()","no dataset.");
      return;
   }
   if (opt.Contains("comp")) {
      //comparison plot
      TString title = "Neural Net Output control. ";
      title += setname;
      setname = "MLP_" + setname + "_comp";
      TH2D *hist = ((TH2D *) gDirectory->Get(setname.Data()));
      if (!hist)
         hist = new TH2D(setname.Data(), title.Data(), 50, -1, 1, 50, -1, 1);
      hist->Reset();
      Int_t nEvents = fTraining->GetN();
      for (i = 0; i < nEvents; i++) {
         GetEntry(fTraining->GetEntry(i));
         hist->Fill(out->GetValue(), (out->GetBranch() - norm[1]) / norm[0]);
      }
      hist->Draw();
   } else {
      //output plot
      TString title = "Neural Net Output. ";
      title += setname;
      setname = "MLP_" + setname;
      TH1D *hist = ((TH1D *) gDirectory->Get(setname.Data()));
      if (!hist)
         hist = new TH1D(setname, title, 50, 1, -1);
      hist->Reset();
      Int_t nEvents = fTraining->GetN();
      for (i = 0; i < nEvents; i++)
         hist->Fill(Result(fTraining->GetEntry(i), index));
      hist->Draw();
      if (opt.Contains("train") && opt.Contains("test")) {
         events = fTraining;
         setname = "train";
         TH1D *hist = ((TH1D *) gDirectory->Get("MLP_test"));
         if (!hist)
            hist = new TH1D(setname, title, 50, 1, -1);
         hist->Reset();
         Int_t nEvents = fTest->GetN();
         for (i = 0; i < nEvents; i++)
            hist->Fill(Result(fTest->GetEntry(i), index));
         hist->Draw("same");
      }
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::DumpWeights(Option_t * filename) const
{
   // Dumps the weights to a text file.
   // Set filename to "-" (default) to dump to the standard output
   TString filen = filename;
   ostream * output;
   if (filen == "")
      return;
   if (filen == "-")
      output = &cout;
   else
      output = new ofstream(filen.Data());
   *output << "#neurons weights" << endl;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   TNeuron *neuron = NULL;
   while ((neuron = (TNeuron *) it->Next()))
      *output << neuron->GetWeight() << endl;
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   TSynapse *synapse = NULL;
   *output << "#synapses weights" << endl;
   while ((synapse = (TSynapse *) it->Next()))
      *output << synapse->GetWeight() << endl;
   delete it;
   if (filen != "-") {
      ((ofstream *) output)->close();
      delete output;
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::LoadWeights(Option_t * filename)
{
   // Loads the weights from a text file conforming to the format
   // defined by DumpWeights.
   TString filen = filename;
   char *buff = new char[100];
   Double_t w;
   if (filen == "")
      return;
   ifstream input(filen.Data());
   input.getline(buff, 100);
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   TNeuron *neuron = NULL;
   while ((neuron = (TNeuron *) it->Next())) {
      input >> w;
      neuron->SetWeight(w);
   }
   delete it;
   input.getline(buff, 100);
   input.getline(buff, 100);
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   TSynapse *synapse = NULL;
   while ((synapse = (TSynapse *) it->Next())) {
      input >> w;
      synapse->SetWeight(w);
   }
   delete it;
   delete[] buff;
}


//______________________________________________________________________________
Double_t TMultiLayerPerceptron::Evaluate(Int_t index, Double_t *params) const
{
   // Returns the Neural Net for a given set of input parameters
   // #parameters must equal #input neurons
   
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   TNeuron *neuron;
   while ((neuron = (TNeuron *) it->Next()))
      neuron->SetNewEvent();
   delete it;
   it = (TObjArrayIter *) fFirstLayer.MakeIterator();
   Int_t i=0;
   while ((neuron = (TNeuron *) it->Next()))
      neuron->ForceExternalValue(params[i++]);
   delete it;
   TNeuron *out = (TNeuron *) (fLastLayer.At(index));
   if (out)
      return out->GetValue();
   else
      return 0;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::Export(Option_t * filename, Option_t * language) const
{
   // Exports the NN as a function for any non-ROOT-dependant code
   // Supported languages are: only C++ , FORTRAN and Python (yet)
   // This feature is also usefull if you want to plot the NN as 
   // a function (TF1 or TF2).
   
   TString lg = language;
   Int_t i;
   if (lg == "C++") {
      TString classname = filename;
      TString header = filename;
      header += ".h";
      TString source = filename;
      source += ".cxx";
      ofstream headerfile(header);
      ofstream sourcefile(source);
      headerfile << "#ifndef NN" << filename << endl;
      headerfile << "#define NN" << filename << endl << endl;
      headerfile << "class " << classname << " { " << endl;
      headerfile << "public:" << endl;
      headerfile << "   " << classname << "() {}" << endl;
      headerfile << "   ~" << classname << "() {}" << endl;
      sourcefile << "#include \"" << header << "\"" << endl;
      sourcefile << "#include \"math.h\"" << endl << endl;
      headerfile << "   double value(int index";
      sourcefile << "double " << classname << "::value(int index";
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++) {
         headerfile << ",double in" << i;
         sourcefile << ",double in" << i;
      }
      headerfile << ");" << endl;
      sourcefile << ") {" << endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         sourcefile << "   input" << i << " = (in" << i << " - "
             << ((TNeuron *) fFirstLayer[i])->GetNormalisation()[1] << ")/"
             << ((TNeuron *) fFirstLayer[i])->GetNormalisation()[0] << ";" 
             << endl;
      sourcefile << "   switch(index) {" << endl;
      TNeuron *neuron;
      TObjArrayIter *it = (TObjArrayIter *) fLastLayer.MakeIterator();
      Int_t idx = 0;
      while ((neuron = (TNeuron *) it->Next()))
         sourcefile << "     case " << idx++ << ":" << endl 
                    << "         return neuron" << neuron << "();" 
                    << endl;
      sourcefile << "     default:" << endl 
                 << "         return 0.;" << endl << "   }" 
                 << endl;
      sourcefile << "}" << endl << endl;
      headerfile << "private:" << endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         headerfile << "   double input" << i << ";" << endl;
      it = (TObjArrayIter *) fNetwork.MakeIterator();
      idx = 0;
      while ((neuron = (TNeuron *) it->Next())) {
         headerfile << "   double neuron" << neuron << "();" << endl;
         sourcefile << "double " << classname << "::neuron" << neuron 
                    << "() {" << endl;
         if (!neuron->GetPre(0))
            sourcefile << "   return input" << idx++ << ";" << endl;
         else {
            sourcefile << "   double input = " << neuron->GetWeight() 
                       << ";" << endl;
            TSynapse *syn;
            Int_t n = 0;
            while ((syn = neuron->GetPre(n++)))
               sourcefile << "   input += synapse" << syn << "();" 
                          << endl;
            if (!neuron->GetPost(0))
               sourcefile << "   return input;" << endl;
            else {
               sourcefile << "   return ((1/(1+exp(-input)))*";
               sourcefile << neuron->GetNormalisation()[0] << ")+" ;
               sourcefile << neuron->GetNormalisation()[1] << ";" << endl;
            }
         }
         sourcefile << "}" << endl << endl;
      }
      delete it;
      TSynapse *synapse = NULL;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      while ((synapse = (TSynapse *) it->Next())) {
         headerfile << "   double synapse" << synapse << "();" << endl;
         sourcefile << "double " << classname << "::synapse" 
                    << synapse << "() {" << endl;
         sourcefile << "   return (neuron" << synapse->GetPre() 
                    << "()*" << synapse->GetWeight() << ");" << endl;
         sourcefile << "}" << endl << endl;
      }
      delete it;
      headerfile << "};" << endl << endl;
      headerfile << "#endif" << endl << endl;
      headerfile.close();
      sourcefile.close();
      cout << header << " and " << source << " created." << endl;
   }
   else if(lg == "FORTRAN") {
      TString implicit = "      implicit double precision (a-h,n-z)\n";
      ofstream sigmoid("sigmoid.f");
      sigmoid 	<< "      double precision FUNCTION SIGMOID(X)"	<< endl
    		<< implicit
		<< "      IF(X.GT.37.) THEN"			<< endl
    		<< "         SIGMOID = 1."			<< endl
		<< "      ELSE IF(X.LT.-709.) THEN"		<< endl
    		<< "         SIGMOID = 0."			<< endl
    		<< "      ELSE"					<< endl
    		<< "         SIGMOID = 1./(1.+EXP(-X))"		<< endl
    		<< "      ENDIF"				<< endl
    		<< "      END"					<< endl;
      sigmoid.close();
      TString source = filename;
      source += ".f";
      ofstream sourcefile(source);

      // Header
      sourcefile << "      double precision function " << filename << "(x, index)" << endl;
      sourcefile << implicit;
      sourcefile << "      double precision x(" << 
      fFirstLayer.GetEntriesFast() << ")" << endl << endl;

/*      // First layer
      sourcefile << "C --- First Layer" << endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         sourcefile << "      first" << i << " = (x(" << i+1 << ") - "
             << ((TNeuron *) fFirstLayer[i])->GetNormalisation()[1] << ")/"
             << ((TNeuron *) fFirstLayer[i])->GetNormalisation()[0] << endl;
*/
      // Last layer
      sourcefile << "C --- Last Layer" << endl;
      TNeuron *neuron;
      TObjArrayIter *it = (TObjArrayIter *) fLastLayer.MakeIterator();
      Int_t idx = 0;
      TString ifelseif = "      if (index.eq.";
      while ((neuron = (TNeuron *) it->Next())) {
         sourcefile << ifelseif.Data() << idx++ << ") then" << endl 
                    << "          " << filename << "=neuron" << neuron << "(x)" 
                    << endl;
	 ifelseif = "      else if (index.eq.";
      }
      sourcefile << "      else" << endl 
                 << "          " << filename << "=0.d0" << endl
		 << "      endif" << endl;
      sourcefile << "      end" << endl;

      // Network
      sourcefile << "C --- Hidden layers" << endl;
      it = (TObjArrayIter *) fNetwork.MakeIterator();
      idx = 0;
      while ((neuron = (TNeuron *) it->Next())) {
         sourcefile << "      double precision function neuron" << neuron << "(x)" << endl
		    << implicit;
         sourcefile << "      double precision x(" << fFirstLayer.GetEntriesFast() << ")" << endl << endl;
         if (!neuron->GetPre(0)) {
            sourcefile << "      neuron" << neuron << " = (x(" << idx+1 << ") - "
             << ((TNeuron *) fFirstLayer[idx])->GetNormalisation()[1] << "d0)/"
             << ((TNeuron *) fFirstLayer[idx])->GetNormalisation()[0] << "d0" << endl;
	     idx++;
         } else {
            sourcefile << "      neuron" << neuron << " = " << neuron->GetWeight() << "d0" << endl;
            TSynapse *syn;
            Int_t n = 0;
            while ((syn = neuron->GetPre(n++)))
               sourcefile << "      neuron" << neuron 
	    		  << " = neuron" << neuron 
			  << " + synapse" << syn << "(x)" << endl;
            if (neuron->GetPost(0)) {
               sourcefile << "      neuron" << neuron << "= (sigmoid(neuron" << neuron << ")*";
               sourcefile << neuron->GetNormalisation()[0] << "d0)+" ;
               sourcefile << neuron->GetNormalisation()[1] << "d0" << endl;
            }
         }
         sourcefile << "      end" << endl;
      }
      delete it;
      
      // Synapses
      sourcefile << "C --- Synapses" << endl;
      TSynapse *synapse = NULL;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      while ((synapse = (TSynapse *) it->Next())) {
         sourcefile << "      double precision function " << "synapse" 
                    << synapse << "(x)\n" << implicit;
         sourcefile << "      double precision x(" << fFirstLayer.GetEntriesFast() << ")" << endl << endl;
         sourcefile << "      synapse" << synapse << "=neuron" << synapse->GetPre() 
                    << "(x)*" << synapse->GetWeight() << "d0" << endl;
         sourcefile << "      end" << endl << endl;
      }
      delete it;
      sourcefile.close();
      cout << source << " created." << endl;
   }
   else if(lg == "Python") {
      TString classname = filename;
      TString pyfile = filename;
      pyfile += ".py";
      ofstream pythonfile(pyfile);
      pythonfile << "from cmath import exp" << endl << endl;
      pythonfile << "class " << classname << ":" << endl;
      pythonfile << "\tdef value(self,index";
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++) {
         pythonfile << ",in" << i;
      }
      pythonfile << "):" << endl;
      for (i = 0; i < fFirstLayer.GetEntriesFast(); i++)
         pythonfile << "\t\tself.input" << i << " = (in" << i << " - "
             << ((TNeuron *) fFirstLayer[i])->GetNormalisation()[1] << ")/"
             << ((TNeuron *) fFirstLayer[i])->GetNormalisation()[0] << endl;
      TNeuron *neuron;
      TObjArrayIter *it = (TObjArrayIter *) fLastLayer.MakeIterator();
      Int_t idx = 0;
      while ((neuron = (TNeuron *) it->Next()))
         pythonfile << "\t\tif index==" << idx++ 
             << ": return self.neuron" << neuron << "()" << endl;
      pythonfile << "\t\treturn 0." << endl;
      it = (TObjArrayIter *) fNetwork.MakeIterator();
      idx = 0;
      while ((neuron = (TNeuron *) it->Next())) {
         pythonfile << "\tdef neuron" << neuron << "(self):" << endl;
         if (!neuron->GetPre(0))
            pythonfile << "\t\treturn self.input" << idx++ << endl;
         else {
            pythonfile << "\t\tinput = " << neuron->GetWeight() << endl;
            TSynapse *syn;
            Int_t n = 0;
            while ((syn = neuron->GetPre(n++)))
               pythonfile << "\t\tinput = input + self.synapse" 
                          << syn << "()" << endl;
            if (!neuron->GetPost(0))
               pythonfile << "\t\treturn input" << endl;
            else {
               pythonfile << "\t\treturn ((1/(1+exp(-input)))*";
               pythonfile << neuron->GetNormalisation()[0] << ")+" ;
               pythonfile << neuron->GetNormalisation()[1] << endl;
            }
         }
      }
      delete it;
      TSynapse *synapse = NULL;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      while ((synapse = (TSynapse *) it->Next())) {
         pythonfile << "\tdef synapse" << synapse << "(self):" << endl;
         pythonfile << "\t\treturn (self.neuron" << synapse->GetPre() 
                    << "()*" << synapse->GetWeight() << ")" << endl;
      }
      delete it;
      pythonfile.close();
      cout << pyfile << " created." << endl;
   }
}

//______________________________________________________________________________
void TMultiLayerPerceptron::Shuffle(Int_t * index, Int_t n) const
{
   // Shuffle the Int_t index[n] in input.
   // Input: 
   //   index: the array to shuffle
   //   n: the size of the array
   // Output:
   //   index: the shuffled indexes
   // This method is used for stochastic training
   
   TTimeStamp ts;
   TRandom3 rnd(ts.GetSec());
   Int_t j, k;
   Int_t a = n - 1;
   for (Int_t i = 0; i < n; i++) {
      j = (Int_t) (rnd.Rndm() * a);
      k = index[j];
      index[j] = index[i];
      index[i] = k;
   }
   return;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::MLP_Stochastic(Double_t * buffer)
{
   // One step for the stochastic method
   // buffer should contain the previous dw vector and will be updated
   
   Int_t nEvents = fTraining->GetN();
   Int_t *index = new Int_t[nEvents];
   Int_t i,j,nentries;
   for (i = 0; i < nEvents; i++)
      index[i] = i;
   fEta *= fEtaDecay;
   Shuffle(index, nEvents);
   TNeuron *neuron;
   TSynapse *synapse;
   for (i = 0; i < nEvents; i++) {
      GetEntry(fTraining->GetEntry(index[i]));
      // First compute DeDw for all neurons: force calculation before 
      // modifying the weights.
      nentries = fFirstLayer.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (TNeuron *) fFirstLayer.UncheckedAt(j);
         neuron->GetDeDw();
      }
      Int_t cnt = 0;
      // Step for all neurons
      nentries = fNetwork.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         neuron = (TNeuron *) fNetwork.UncheckedAt(j);
         buffer[cnt] = (-fEta) * (neuron->GetDeDw() + fDelta) 
		       + fEpsilon * buffer[cnt];
         neuron->SetWeight(neuron->GetWeight() + buffer[cnt++]);
      }
      // Step for all synapses
      nentries = fSynapses.GetEntriesFast();
      for (j=0;j<nentries;j++) {
         synapse = (TSynapse *) fSynapses.UncheckedAt(j);
         buffer[cnt] = (-fEta) * (synapse->GetDeDw() + fDelta) 
                       + fEpsilon * buffer[cnt];
         synapse->SetWeight(synapse->GetWeight() + buffer[cnt++]);
      }
   }
   delete[]index;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::MLP_Batch(Double_t * buffer)
{
   // One step for the batch (stochastic) method. 
   // DEDw should have been updated before calling this.
   
   fEta *= fEtaDecay;
   Int_t cnt = 0;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   TNeuron *neuron = NULL;
   // Step for all neurons
   while ((neuron = (TNeuron *) it->Next())) {
      buffer[cnt] = (-fEta) * (neuron->GetDEDw() + fDelta) 
                    + fEpsilon * buffer[cnt];
      neuron->SetWeight(neuron->GetWeight() + buffer[cnt++]);
   }
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   TSynapse *synapse = NULL;
   // Step for all synapses
   while ((synapse = (TSynapse *) it->Next())) {
      buffer[cnt] = (-fEta) * (synapse->GetDEDw() + fDelta) 
                    + fEpsilon * buffer[cnt];
      synapse->SetWeight(synapse->GetWeight() + buffer[cnt++]);
   }
   delete it;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::MLP_Line(Double_t * origin, Double_t * dir, Double_t dist)
{
   // Sets the weights to a point along a line
   // Weights are set to [origin + (dist * dir)].
   
   Int_t idx = 0;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   while ((neuron = (TNeuron *) it->Next())) {
      neuron->SetWeight(origin[idx] + (dir[idx++] * dist));
   }
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   while ((synapse = (TSynapse *) it->Next())) {
      synapse->SetWeight(origin[idx] + (dir[idx++] * dist));
   }
   delete it;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SteepestDir(Double_t * dir)
{
   // Sets the search direction to steepest descent.
   Int_t idx = 0;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   TObjArrayIter *it = (TObjArrayIter *) fNetwork.MakeIterator();
   while ((neuron = (TNeuron *) it->Next()))
      dir[idx++] = -neuron->GetDEDw();
   delete it;
   it = (TObjArrayIter *) fSynapses.MakeIterator();
   while ((synapse = (TSynapse *) it->Next()))
      dir[idx++] = -synapse->GetDEDw();
   delete it;
}

//______________________________________________________________________________
bool TMultiLayerPerceptron::LineSearch(Double_t * direction, Double_t * buffer)
{
   // Search along the line defined by direction.
   // buffer is not used but is updated with the new dw 
   // so that it can be used by a later stochastic step.
   // It returns true if the line search fails.
   
   Int_t idx = 0;
   Int_t j,nentries;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   // store weights before line search 
   Double_t *origin = new Double_t[fNetwork.GetEntriesFast() +
                                   fSynapses.GetEntriesFast()];
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      origin[idx++] = neuron->GetWeight();
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      origin[idx++] = synapse->GetWeight();
   }
   // try to find a triplet (alpha1, alpha2, alpha3) such that 
   // Error(alpha1)>Error(alpha2)<Error(alpha3)
   Double_t err1 = GetError(kTraining);
   Double_t alpha1 = 0.;
   Double_t alpha2 = fLastAlpha;
   if (alpha2 < 0.01)
      alpha2 = 0.01;
   if (alpha2 > 2.0)
      alpha2 = 2.0;
   Double_t alpha3 = alpha2;
   MLP_Line(origin, direction, alpha2);
   Double_t err2 = GetError(kTraining);
   Double_t err3 = err2;
   Bool_t bingo = false;
   Int_t icount;
   if (err1 > err2) {
      for (icount = 0; icount < 100; icount++) {
         alpha3 *= fTau;
         MLP_Line(origin, direction, alpha3);
         err3 = GetError(kTraining);
         if (err3 > err2) {
            bingo = true;
            break;
         }
         alpha1 = alpha2;
         err1 = err2;
         alpha2 = alpha3;
         err2 = err3;
      }
      if (!bingo) {
         MLP_Line(origin, direction, 0.);
         delete[]origin;
         return true;
      }
   } else {
      for (icount = 0; icount < 100; icount++) {
         alpha2 /= fTau;
         MLP_Line(origin, direction, alpha2);
         err2 = GetError(kTraining);
         if (err1 > err2) {
            bingo = true;
            break;
         }
         alpha3 = alpha2;
         err3 = err2;
      }
      if (!bingo) {
         MLP_Line(origin, direction, 0.);
         delete[]origin;
         fLastAlpha = 0.05;
         return true;
      }
   }
   // Sets the weights to the bottom of parabola
   fLastAlpha = 0.5 * (alpha1 + alpha3 -
                (err3 - err1) / ((err3 - err2) / (alpha3 - alpha2) 
                - (err2 - err1) / (alpha2 - alpha1)));
   fLastAlpha = fLastAlpha < 10000 ? fLastAlpha : 10000;
   MLP_Line(origin, direction, fLastAlpha);
   GetError(kTraining);
   // Stores weight changes (can be used by a later stochastic step)
   idx = 0;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      buffer[idx] = neuron->GetWeight() - origin[idx++];
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      buffer[idx] = synapse->GetWeight() - origin[idx++];
   }
   return false;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::ConjugateGradientsDir(Double_t * dir, Double_t beta)
{
   // Sets the search direction to conjugate gradient direction
   // beta should be:               
   //  ||g_{(t+1)}||^2 / ||g_{(t)}||^2                   (Fletcher-Reeves)
   //  g_{(t+1)} (g_{(t+1)}-g_{(t)}) / ||g_{(t)}||^2     (Ribiere-Polak)
   
   Int_t idx = 0;
   Int_t j,nentries;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      dir[idx] = -neuron->GetDEDw() + beta * dir[idx];
      idx++;
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      dir[idx] = -synapse->GetDEDw() + beta * dir[idx];
      idx++;
   }
}

//______________________________________________________________________________
bool TMultiLayerPerceptron::GetBFGSH(TMatrixD & BFGSH, TMatrixD & gamma, TMatrixD & delta)
{
   // Computes the hessian matrix using the BFGS update algorithm.
   // from gamma (g_{(t+1)}-g_{(t)}) and delta (w_{(t+1)}-w_{(t)}).
   // It returns true if such a direction could not be found 
   // (if gamma and delta are orthogonal).
   
   TMatrixD gd(gamma, TMatrixD::kTransposeMult, delta);
   if ((Double_t) gd[0][0] == 0.)
      return true;
   TMatrixD Hg(BFGSH, TMatrixD::kMult, gamma);
   TMatrixD tmp(gamma, TMatrixD::kTransposeMult, BFGSH);
   TMatrixD gHg(gamma, TMatrixD::kTransposeMult, Hg);
   Double_t a = 1 / (Double_t) gd[0][0];
   Double_t f = 1 + ((Double_t) gHg[0][0] * a);
   TMatrixD res( TMatrixD(delta, TMatrixD::kMult, 
                TMatrixD(TMatrixD::kTransposed, delta)));
   res *= f;
   res -= (TMatrixD(delta, TMatrixD::kMult, tmp) + 
           TMatrixD(Hg, TMatrixD::kMult, 
                   TMatrixD(TMatrixD::kTransposed, delta)));
   res *= a;
   BFGSH += res;
   return false;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::SetGammaDelta(TMatrixD & gamma, TMatrixD & delta,
                                          Double_t * buffer)
{
   // Sets the gamma (g_{(t+1)}-g_{(t)}) and delta (w_{(t+1)}-w_{(t)}) vectors
   // Gamma is computed here, so ComputeDEDw cannot have been called before, 
   // and delta is a direct translation of buffer into a TMatrixD.
   
   Int_t els = fNetwork.GetEntriesFast() + fSynapses.GetEntriesFast();
   Int_t idx = 0;
   Int_t j,nentries;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      gamma[idx++][0] = -neuron->GetDEDw();
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      gamma[idx++][0] = -synapse->GetDEDw();
   }
   for (Int_t i = 0; i < els; i++)
      delta[i] = buffer[i];
   //delta.SetElements(buffer,"F");
   ComputeDEDw();
   idx = 0;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      gamma[idx++][0] += neuron->GetDEDw();
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      gamma[idx++][0] += synapse->GetDEDw();
   }
}

//______________________________________________________________________________
Double_t TMultiLayerPerceptron::DerivDir(Double_t * dir)
{
   // scalar product between gradient and direction
   // = derivative along direction
   
   Int_t idx = 0;
   Int_t j,nentries;
   Double_t output = 0;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      output += neuron->GetDEDw() * dir[idx++];
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      output += synapse->GetDEDw() * dir[idx++];
   }
   return output;
}

//______________________________________________________________________________
void TMultiLayerPerceptron::BFGSDir(TMatrixD & BFGSH, Double_t * dir)
{
   // Computes the direction for the BFGS algorithm as the product
   // between the Hessian estimate (BFGSH) and the dir.
   
   Int_t els = fNetwork.GetEntriesFast() + fSynapses.GetEntriesFast();
   TMatrixD dedw(els, 1);
   Int_t idx = 0;
   Int_t j,nentries;
   TNeuron *neuron = NULL;
   TSynapse *synapse = NULL;
   nentries = fNetwork.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      neuron = (TNeuron *) fNetwork.UncheckedAt(j);
      dedw[idx++][0] = neuron->GetDEDw();
   }
   nentries = fSynapses.GetEntriesFast();
   for (j=0;j<nentries;j++) {
      synapse = (TSynapse *) fSynapses.UncheckedAt(j);
      dedw[idx++][0] = synapse->GetDEDw();
   }
   TMatrixD direction(BFGSH, TMatrixD::kMult, dedw);
   for (Int_t i = 0; i < els; i++)
      dir[i] = -direction[i][0];
   //direction.GetElements(dir,"F");
}

//______________________________________________________________________________
void TMultiLayerPerceptron::Draw(const Option_t*)
{
  // Draws the network structure.
  // Neurons are depicted by a blue disk, and synapses by
  // lines connecting neurons.
  // The line width is proportionnal to the weight.

#define NeuronSize 2.5

   Int_t nLayers = fStructure.CountChar(':')+1;
   Float_t xStep = 1./(nLayers+1.);
   Int_t layer;
   for(layer=0; layer< nLayers-1; layer++) {
      Float_t nNeurons_this = 0;
      if(layer==0) {
         TString input      = TString(fStructure(0, fStructure.First(':')));
         nNeurons_this = input.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
	 while (end != -1) {
           Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
	   cnt++;
           beg = end + 1;
           end = hidden.Index(":", beg + 1);
	   if(layer==cnt) nNeurons_this = num;
         }
	 Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
	 cnt++;
	 if(layer==cnt) nNeurons_this = num;
      }
      Float_t nNeurons_next = 0;
      if(layer==nLayers-2) {
         TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
         nNeurons_next = output.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
	 while (end != -1) {
           Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
	   cnt++;
           beg = end + 1;
           end = hidden.Index(":", beg + 1);
	   if(layer+1==cnt) nNeurons_next = num;
         }
	 Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
	 cnt++;
	 if(layer+1==cnt) nNeurons_next = num;
      }
      Float_t yStep_this = 1./(nNeurons_this+1.);
      Float_t yStep_next = 1./(nNeurons_next+1.);
      TObjArrayIter* it = (TObjArrayIter *) fSynapses.MakeIterator();
      TSynapse *theSynapse = NULL;
      Float_t maxWeight = 0;
      while ((theSynapse = (TSynapse *) it->Next()))
         maxWeight = maxWeight < theSynapse->GetWeight() ? theSynapse->GetWeight() : maxWeight;
      delete it;
      it = (TObjArrayIter *) fSynapses.MakeIterator();
      for(Int_t neuron1=0; neuron1<nNeurons_this; neuron1++) {
         for(Int_t neuron2=0; neuron2<nNeurons_next; neuron2++) {
            TLine* synapse = new TLine(xStep*(layer+1),yStep_this*(neuron1+1),xStep*(layer+2),yStep_next*(neuron2+1));
	    synapse->Draw();
	    theSynapse = (TSynapse *) it->Next();
	    synapse->SetLineWidth(Int_t((theSynapse->GetWeight()/maxWeight)*10.));
	    synapse->SetLineStyle(1);
	    if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.5) synapse->SetLineStyle(2);
	    if(((TMath::Abs(theSynapse->GetWeight())/maxWeight)*10.)<0.25) synapse->SetLineStyle(3);
         }
      }
      delete it;
   }
   for(layer=0; layer< nLayers; layer++) {
      Float_t nNeurons = 0;
      if(layer==0) {
         TString input      = TString(fStructure(0, fStructure.First(':')));
         nNeurons = input.CountChar(',')+1;
      }
      else if(layer==nLayers-1) {
         TString output = TString(fStructure(fStructure.Last(':') + 1,fStructure.Length() - fStructure.Last(':')));
         nNeurons = output.CountChar(',')+1;
      }
      else {
         Int_t cnt=0;
         TString hidden = TString(fStructure(fStructure.First(':') + 1,fStructure.Last(':') - fStructure.First(':') - 1));
         Int_t beg = 0;
         Int_t end = hidden.Index(":", beg + 1);
	 while (end != -1) {
           Int_t num = atoi(TString(hidden(beg, end - beg)).Data());
	   cnt++;
           beg = end + 1;
           end = hidden.Index(":", beg + 1);
	   if(layer==cnt) nNeurons = num;
         }
	 Int_t num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
	 cnt++;
	 if(layer==cnt) nNeurons = num;
      }
      Float_t yStep = 1./(nNeurons+1.);
      for(Int_t neuron=0; neuron<nNeurons; neuron++) {
         TMarker* m = new TMarker(xStep*(layer+1),yStep*(neuron+1),20);
         m->SetMarkerColor(4);
         m->SetMarkerSize(NeuronSize);
         m->Draw();
      }
   }
   TString input      = TString(fStructure(0, fStructure.First(':')));
   Int_t beg = 0;
   Int_t end = input.Index(",", beg + 1);
   TString brName;
   Int_t cnt = 0;
   Float_t yStep = 1./(input.CountChar(',')+2.);
   while (end != -1) {
      brName = TString(input(beg, end - beg));
      beg = end + 1;
      end = input.Index(",", beg + 1);
      cnt++;
      TText* label = new TText(0.5*xStep,yStep*cnt,brName.Data());
      label->Draw();
   }
   brName = TString(input(beg, input.Length() - beg));
   cnt++;
   TText* label = new TText(0.5*xStep,yStep*cnt,brName.Data());
   label->Draw();
}
