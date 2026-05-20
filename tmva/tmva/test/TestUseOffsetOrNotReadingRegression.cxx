#include "TMVA/Reader.h"

#include <gtest/gtest.h>

#include <iostream>

constexpr auto methodString = R"(
<?xml version="1.0"?>
<MethodSetup Method="MLP::MLP">
  <GeneralInfo>
    <Info name="TMVA Release" value="4.1.0 [262400]"/>
    <Info name="ROOT Release" value="5.28/00 [334848]"/>
    <Info name="Creator" value="tianjp"/>
    <Info name="Date" value="Mon Jul  6 21:53:20 2015"/>
    <Info name="Host" value="Linux cw104.cc.kek.jp 2.6.18-164.el5 #1 SMP Tue Aug 18 15:51:48 EDT 2009 x86_64 x86_64 x86_64 GNU/Linux"/>
    <Info name="Dir" value="/gpfs/fs01/ilc/tianjp/analysis/PostDBD/IsolatedLeptonTagging/training/macros_e1e1h_gg_qqqq_250"/>
    <Info name="Training events" value="182321"/>
    <Info name="TrainingTime" value="801.579"/>
    <Info name="AnalysisType" value="Classification"/>
  </GeneralInfo>
  <Options>
    <Option name="NCycles" modified="Yes">500</Option>
    <Option name="HiddenLayers" modified="Yes">N+5</Option>
    <Option name="NeuronType" modified="Yes">tanh</Option>
    <Option name="RandomSeed" modified="No">1</Option>
    <Option name="EstimatorType" modified="No">MSE</Option>
    <Option name="NeuronInputType" modified="No">sum</Option>
    <Option name="V" modified="Yes">False</Option>
    <Option name="VerbosityLevel" modified="No">Default</Option>
    <Option name="VarTransform" modified="Yes">N</Option>
    <Option name="H" modified="Yes">True</Option>
    <Option name="CreateMVAPdfs" modified="No">False</Option>
    <Option name="IgnoreNegWeightsInTraining" modified="No">False</Option>
    <Option name="TrainingMethod" modified="No">BP</Option>
    <Option name="LearningRate" modified="No">2.000000e-02</Option>
    <Option name="DecayRate" modified="No">1.000000e-02</Option>
    <Option name="TestRate" modified="Yes">10</Option>
    <Option name="EpochMonitoring" modified="Yes">True</Option>
    <Option name="Sampling" modified="No">1.000000e+00</Option>
    <Option name="SamplingEpoch" modified="No">1.000000e+00</Option>
    <Option name="SamplingImportance" modified="No">1.000000e+00</Option>
    <Option name="SamplingTraining" modified="No">True</Option>
    <Option name="SamplingTesting" modified="No">False</Option>
    <Option name="ResetStep" modified="No">50</Option>
    <Option name="Tau" modified="No">3.000000e+00</Option>
    <Option name="BPMode" modified="No">sequential</Option>
    <Option name="BatchSize" modified="No">-1</Option>
    <Option name="ConvergenceImprove" modified="No">1.000000e-30</Option>
    <Option name="ConvergenceTests" modified="No">-1</Option>
    <Option name="UseRegulator" modified="No">False</Option>
    <Option name="UpdateLimit" modified="No">10</Option>
    <Option name="CalculateErrors" modified="No">False</Option>
  </Options>
  <Variables NVar="9">
    <Variable VarIndex="0" Expression="coneec" Label="coneec" Title="coneec" Unit="" Internal="coneec" Type="c" Min="0" Max="106.682"/>
    <Variable VarIndex="1" Expression="coneen" Label="coneen" Title="coneen" Unit="" Internal="coneen" Type="c" Min="0" Max="115.829"/>
    <Variable VarIndex="2" Expression="momentum" Label="momentum" Title="momentum" Unit="" Internal="momentum" Type="m" Min="5.00001" Max="91.6776"/>
    <Variable VarIndex="3" Expression="coslarcon" Label="coslarcon" Title="coslarcon" Unit="" Internal="coslarcon" Type="c" Min="0.950006" Max="1"/>
    <Variable VarIndex="4" Expression="energyratio" Label="energyratio" Title="energyratio" Unit="" Internal="energyratio" Type="e" Min="0.0410161" Max="1"/>
    <Variable VarIndex="5" Expression="ratioecal" Label="ratioecal" Title="ratioecal" Unit="" Internal="ratioecal" Type="r" Min="0.9" Max="1"/>
    <Variable VarIndex="6" Expression="ratiototcal" Label="ratiototcal" Title="ratiototcal" Unit="" Internal="ratiototcal" Type="r" Min="0.500015" Max="1.29995"/>
    <Variable VarIndex="7" Expression="nsigd0" Label="nsigd0" Title="nsigd0" Unit="" Internal="nsigd0" Type="n" Min="-49.9731" Max="49.9946"/>
    <Variable VarIndex="8" Expression="nsigz0" Label="nsigz0" Title="nsigz0" Unit="" Internal="nsigz0" Type="n" Min="-4.99833" Max="4.99963"/>
  </Variables>
  <Spectators NSpec="0"/>
  <Transformations NTransformations="1">
    <Transform Name="Normalize" NVariables="9" NTargets="0">
      <Class ClassIndex="0">
        <Variables>
          <Variable VarIndex="0" Min="0.0000000000000000e+00" Max="6.9274139404296875e+01"/>
          <Variable VarIndex="1" Min="0.0000000000000000e+00" Max="7.7186988830566406e+01"/>
          <Variable VarIndex="2" Min="5.0062704086303711e+00" Max="9.1677558898925781e+01"/>
          <Variable VarIndex="3" Min="9.5000582933425903e-01" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="4" Min="4.8792138695716858e-02" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="5" Min="9.0177327394485474e-01" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="6" Min="5.0208520889282227e-01" Max="1.2992891073226929e+00"/>
          <Variable VarIndex="7" Min="-4.9855728149414062e+01" Max="4.9925727844238281e+01"/>
          <Variable VarIndex="8" Min="-4.8339676856994629e+00" Max="4.9416265487670898e+00"/>
        </Variables>
        <Targets/>
      </Class>
      <Class ClassIndex="1">
        <Variables>
          <Variable VarIndex="0" Min="0.0000000000000000e+00" Max="1.0668167877197266e+02"/>
          <Variable VarIndex="1" Min="0.0000000000000000e+00" Max="1.1582881164550781e+02"/>
          <Variable VarIndex="2" Min="5.0000071525573730e+00" Max="8.7144081115722656e+01"/>
          <Variable VarIndex="3" Min="9.5031201839447021e-01" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="4" Min="4.1016053408384323e-02" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="5" Min="9.0000003576278687e-01" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="6" Min="5.0001466274261475e-01" Max="1.2999539375305176e+00"/>
          <Variable VarIndex="7" Min="-4.9973102569580078e+01" Max="4.9994590759277344e+01"/>
          <Variable VarIndex="8" Min="-4.9983320236206055e+00" Max="4.9996280670166016e+00"/>
        </Variables>
        <Targets/>
      </Class>
      <Class ClassIndex="2">
        <Variables>
          <Variable VarIndex="0" Min="0.0000000000000000e+00" Max="1.0668167877197266e+02"/>
          <Variable VarIndex="1" Min="0.0000000000000000e+00" Max="1.1582881164550781e+02"/>
          <Variable VarIndex="2" Min="5.0000071525573730e+00" Max="9.1677558898925781e+01"/>
          <Variable VarIndex="3" Min="9.5000582933425903e-01" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="4" Min="4.1016053408384323e-02" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="5" Min="9.0000003576278687e-01" Max="1.0000000000000000e+00"/>
          <Variable VarIndex="6" Min="5.0001466274261475e-01" Max="1.2999539375305176e+00"/>
          <Variable VarIndex="7" Min="-4.9973102569580078e+01" Max="4.9994590759277344e+01"/>
          <Variable VarIndex="8" Min="-4.9983320236206055e+00" Max="4.9996280670166016e+00"/>
        </Variables>
        <Targets/>
      </Class>
    </Transform>
  </Transformations>
  <MVAPdfs/>
  <Weights>
    <Layout NLayers="3">
      <Layer Index="0" NNeurons="10">
        <Neuron NSynapses="14">
          7.2211987450405757e-01 2.4919653039859249e+00 1.3569862383030229e+00 4.2065367866214148e-02 -1.9271628356502275e+00 -1.1405745802043241e+00 -1.1413246267097445e+00 3.2630530566686000e-01 -5.6983114288315073e-01 4.4474245965062725e-01 -1.6334240010343701e+00 -8.4507998536895557e-01 -1.5414148728694572e+00 2.3620545594532936e+00 
        </Neuron>
        <Neuron NSynapses="14">
          -1.6575654498562458e+00 -1.3122948283385854e+00 1.9627573290577752e-01 2.1737799174449628e+00 -9.0578961461805629e-01 1.9258924248246687e+00 -1.3951230297023887e+00 -3.2502774837317455e-01 1.5475512304544354e+00 -1.8490381459988794e-01 -1.5672975226934840e+00 -6.0685032652367976e-01 5.9558644148858675e-01 -1.1009139553244240e+00 
        </Neuron>
        <Neuron NSynapses="14">
          -3.4188307254959098e+00 -2.7575858314151902e+00 4.5966166548861698e-01 2.3294278177286931e+00 5.4541036503066531e-01 -1.8080042193959227e+00 -2.1227141463839613e-01 -3.0543880109772465e+00 9.1605640015888257e-01 1.3789241982571276e+00 -1.5717925718526158e+00 1.3754240454239963e+00 3.9374114310304509e+00 -1.7738171007000065e+00 
        </Neuron>
        <Neuron NSynapses="14">
          -1.5968693403844750e-01 1.1145951553045108e+00 -1.4159003834085657e+00 6.0297145584196056e-02 1.3878455542715133e+00 1.1419499207152726e+00 1.6507363747990624e+00 1.4241292266979935e+00 4.3941464907216776e-01 1.0092870995233869e+00 -1.0347391687230838e+00 9.5825997318322087e-01 -1.1137302702468004e+00 -3.8053149432000538e-01 
        </Neuron>
        <Neuron NSynapses="14">
          -1.3811066391765725e+00 -1.0269703129468069e+00 8.9846642582134439e-01 2.2218697836708325e+00 -1.5266537383204533e+00 2.8961822618272919e+00 -1.6315566808179788e-01 3.3473095884137960e-01 2.5836423437977603e+00 -4.6491624787119384e-01 -2.1476174570634607e-01 -8.0282518413909251e-02 7.0471686178358639e-01 -2.6742452194283319e+00 
        </Neuron>
        <Neuron NSynapses="14">
          -1.2106948783778220e+00 -1.4506896596639407e+00 2.6193927929434069e-01 1.1830554386743977e+00 -8.0956619570406740e-01 -3.6489365948718511e-01 -7.8864804111718767e-01 -1.3139709769522938e+00 2.3910957407361660e-01 -8.4606225345855335e-01 1.3894627946236020e+00 8.2091452957469746e-01 1.0220840655263141e+00 -1.3488425576354810e+00 
        </Neuron>
        <Neuron NSynapses="14">
          -1.0794167389105749e+00 -8.3184491375656222e-01 6.0617346665919569e-01 -2.2799473592738509e-01 -6.3285446800925849e-01 -2.5666328823913759e-01 5.8195566341021174e-02 -4.0331231853650723e-01 -2.9309063015359790e-02 -5.1293420267422585e-01 -6.4850025095157293e-01 1.0476187945251085e+00 5.3517332347439095e-01 -4.4432892891189618e-01 
        </Neuron>
        <Neuron NSynapses="14">
          -7.9009317668969889e-02 -6.8009231355905098e-01 -1.5469989590353250e+00 -3.1154256505354971e-01 6.3494543476336851e-01 3.5305148148059018e-01 -9.3316483412409157e-01 3.5115382304814879e-01 -1.1308456476053141e+00 2.0732719780857942e-01 -4.3111482323736690e-02 4.4833590440688398e-01 -1.1594531256662099e+00 5.3719332359605654e-01 
        </Neuron>
        <Neuron NSynapses="14">
          1.1472352164802804e+00 3.7789223209064349e-01 -3.2852491864110289e-01 1.5798173079015827e+00 1.6826917554858048e-01 1.8001410524567910e-01 7.0644662311944506e-01 3.2590394334824142e-02 -8.7952957871731338e-02 7.1632808587872826e-02 -2.4871170475433782e-01 1.1557134530360138e+00 1.0608848360241897e-01 -1.0580695982475036e-01 
        </Neuron>
        <Neuron NSynapses="14">
          -3.8956689904730113e-01 1.5514886363254041e-01 -2.3078431135259190e+00 7.6793678325873160e-01 1.0674159390952971e+00 -3.3413931452256405e+00 1.7582679017401026e+00 -8.0724537984762257e-01 -1.0795441444523637e-01 3.1071221124309907e+00 2.4277090725555195e+00 -1.6969367654155174e+00 1.1302060581143833e+00 3.1013478926727012e+00 
        </Neuron>
      </Layer>
      <Layer Index="1" NNeurons="15">
        <Neuron NSynapses="1">
          2.8073276420564353e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -2.5083341123546704e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -7.9749835368894828e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -2.4219594048714227e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -5.8660140860897225e-01 
        </Neuron>
        <Neuron NSynapses="1">
          2.3192690630283436e-01 
        </Neuron>
        <Neuron NSynapses="1">
          3.6420742767100944e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -1.6804529636560728e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -3.0201581984332387e-01 
        </Neuron>
        <Neuron NSynapses="1">
          6.5316912677068972e-01 
        </Neuron>
        <Neuron NSynapses="1">
          5.4536798547068941e-03 
        </Neuron>
        <Neuron NSynapses="1">
          1.0603005543655107e-02 
        </Neuron>
        <Neuron NSynapses="1">
          2.8679691975339600e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -5.9800482589005310e-01 
        </Neuron>
        <Neuron NSynapses="1">
          -5.0874954760531321e-01 
        </Neuron>
      </Layer>
      <Layer Index="2" NNeurons="1">
        <Neuron NSynapses="0"/>
      </Layer>
    </Layout>
  </Weights>
</MethodSetup>
)";

TEST(TestUseOffsetOrNot, ReadingOldFile)
{
   Float_t coneec = 0., coneen = 0., momentum = 0., coslarcon = 0., energyratio = 0., ratioecal = 0., ratiototcal = 0.,
           nsigd0 = 0., nsigz0 = 0.;

   // Create reader for electron weights
   TMVA::Reader electronReader("!Color:!Silent");
   electronReader.AddVariable("coneec", &coneec);
   electronReader.AddVariable("coneen", &coneen);
   electronReader.AddVariable("momentum", &momentum);
   electronReader.AddVariable("coslarcon", &coslarcon);
   electronReader.AddVariable("energyratio", &energyratio);
   electronReader.AddVariable("ratioecal", &ratioecal);
   electronReader.AddVariable("ratiototcal", &ratiototcal);
   electronReader.AddVariable("nsigd0", &nsigd0);
   electronReader.AddVariable("nsigz0", &nsigz0);

   const auto methodType = TMVA::Types::Instance().GetMethodType("MLP");

   EXPECT_NO_THROW(electronReader.BookMVA(methodType, methodString));
}
