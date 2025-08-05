# \file
# \ingroup tutorial_roostats
# RooStats Model Inspector
#
# Usage:
# The usage is the same as the StandardXxxDemo.py Python Scripts.
# The macro expects a root file containing a workspace with a ModelConfig and a dataset
#
# ~~~{.py}
# $ ipython3
# %run ModelInspector.py
# #ModelInspector(fileName, workspaceName, modelConfigName, dataSetName)
# ~~~
# Fails with normal python.
# Drag the sliders to adjust the parameters of the model.
# the min and max range of the sliders are used to define the upper & lower variation
# the pointer position of the slider is the central blue curve.
#
# Click the FIT button to
#
# To Do:
#  - Check a better way to exit without crashing.
#  - check boxes to specify which nuisance parameters used in making variation
#  - a button to make the profile inspector plots
#  - a check button to use MINOS errors
#  - have fit button show the covariance matrix from the fit
#  - a button to make the log likelihood plots
#  - a dialog to open the desired file
#  - ability to see the signal and background contributions?
#
# \macro_code
#
#  - Version 1, October 2011
#     - based on tutorial macro by Bertrand Bellenot, Ilka Antcheva
#  - Version 2, November 2011
#     - fixes from Bertrand Bellenot for scrolling window for many parameters
#  - Version 3, April 2024
#     - translation from C++ macros to python3
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)

import sys
import ROOT

from enum import Enum


class ETestCommandIdentifiers(Enum):
    HId1 = 1
    HId2 = 2
    HId3 = 3
    HCId1 = 4
    HCId2 = 5
    HSId1 = 6


ETestCmdId = ETestCommandIdentifiers


class ModelInspectorGUI(ROOT.TGMainFrame):
    # private:
    # -fCanvas 	 = TRootEmbeddedCanvas()
    # -fLcan 	 = TGLayoutHints()
    # -fFitFcn 	 = TF1()
    # -fPlot 	 = RooPlot()
    # -fWS 	 = RooWorkspace()
    # -fFile 	 = TFile()
    # -fMC 	 = ModelConfig()
    # -fData 	 = super(RooAbsData)
    # -fFitRes 	 = RooFitResult()
    # -
    # -fSliderList 	 = TList()
    # -fFrameList 	 = TList()
    # -fPlotList = vector("RooPlot *")()
    # -fSliderMap =    ROOT.map("TGTripleHSlider *", " TString " )() # it is an instance
    # -
    # -TSliderMap =    ROOT.map(TGTripleHSlider , char )   # it is a type
    # -fLabelMap =    ROOT.map("TGTripleHSlider*" , "TGLabel *" )()
    # -
    # -fFitButton 	 = TGButton()
    # -fExitButton 	 = TGTextButton()
    # -
    # -# BB: a TGCanvas and a vertical frame are needed for using scrollbars
    # -fCan = 		 TGCanvas()
    # -fVFrame = 		 TGVerticalFrame()
    # -
    # -fHframe0 = fHframe1 = fHframe2 = 		 TGHorizontalFrame()
    # -fBly = fBfly1 = fBfly2 = fBfly3 = 		 TGLayoutHints()
    # -fHslider1 = 		 TGTripleHSlider()
    # -fTbh1 = fTbh2 = fTbh3 = 		 TGTextBuffer()
    # -fCheck1 = fCheck2 = 		 TGCheckButton()

    # public:
    # def __init__(RooWorkspace, ModelConfig, RooAbsData): pass
    # def __del__() : pass

    # def CloseWindow() : pass
    # def DoText( str ): pass
    # def DoSlider(): pass
    # def DoSlider( str ): pass
    # def DoFit(): pass
    # def DoExit(): pass
    # def HandleButtons(): pass

    # ______________________________________________________________________________
    def __del__(self):
        # Clean up

        self.Cleanup()

    # ______________________________________________________________________________
    def CloseWindow(self):
        # Called when window is closed via the window manager.

        del self
        pass

    # ______________________________________________________________________________
    def DoText(self):
        # Handle text entry widgets.

        te = ROOT.BindObject(self.gTQSender, ROOT.TGTextEntry)
        Id = te.WidgetId()
        if Id == ETestCmdId.HId1.value:
            fHslider1.SetPosition(atof(self.fTbh1.GetString()), self.fHslider1.GetMaxPosition())
        elif Id == ETestCmdId.HId2.value:
            fHslider1.SetPointerPosition(atof(self.fTbh2.GetString()))
        elif Id == ETestCmdId.HId3.value:
            fHslider1.SetPosition(self.fHslider1.GetMinPosition(), atof(self.fTbh1.GetString()))

        self.DoSlider()

    # ______________________________________________________________________________
    def DoFit(self):
        self.fFitRes = self.fMC.GetPdf().fitTo(self.fData, Save=True)

        for it in self.fSliderMap:
            param = self.fWS.var(it.second)
            param = self.fFitRes.floatParsFinal().find(it.second.Data())
            it.first.SetPosition(param.getVal() - param.getError(), param.getVal() + param.getError())
            it.first.SetPointerPosition(param.getVal())

        self.DoSlider()

    # ______________________________________________________________________________
    # def DoSlider(text = ""):

    #   print(f".", text)
    #

    # ______________________________________________________________________________
    def DoSlider(self):
        # Handle slider widgets.

        # char buf[32];

        simPdf = ROOT.nullptr
        numCats = 0
        if str(self.fMC.GetPdf().ClassName()) == "RooSimultaneous":
            simPdf = self.fMC.GetPdf()
            channelCat = simPdf.indexCat()
            numCats = channelCat.numTypes()
        else:
            pass

        ###############
        if not simPdf:
            ###############
            # if not SimPdf
            ###############

            # pre loop
            # map<TGTripleHSlider , const char >.iterator it
            it = ROOT.map(TGTripleHSlider, char).iterator  # it # unnecessary

            # try :
            #   del self.fPlot
            # except :
            #   pass
            # self.fPlot = (self.fMC.GetObservables().first()).frame()
            self.fPlot = (self.fMC.GetObservables().first()).frame()
            self.fData.plotOn(self.fPlot)
            normCount = Double_t()

            # high loop
            # it0 = self.fSliderMap.begin()
            for it in self.fSliderMap:
                name = it.second
                self.fWS.var(name).setVal(it.first.GetMaxPosition())
                param = self.fWS.var(name)
                self.fLabelMap[it.first].SetText(
                    ROOT.Form(
                        "{:s} = {:.3}f [{:.3}f,{:.3}f]".format(
                            param.GetName(),
                            it.first.GetPointerPosition(),
                            it.first.GetMinPosition(),
                            it.first.GetMaxPosition(),
                        )
                    )
                )

            normCount = self.fMC.GetPdf().expectedEvents(self.fMC.GetObservables())
            self.fMC.GetPdf().plotOn(
                self.fPlot, ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent), LineColor="r"
            )

            # low loop
            # it0 = self.fSliderMap.begin()
            for it in self.fSliderMap:
                name = it.second
                self.fWS.var(name).setVal(it.first.GetMinPosition())

            normCount = self.fMC.GetPdf().expectedEvents(self.fMC.GetObservables())
            self.fMC.GetPdf().plotOn(
                self.fPlot, ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent), LineColor="g"
            )

            # central loop
            # it0 = self.fSliderMap.begin()
            for it in self.fSliderMap:
                name = it.second
                self.fWS.var(name).setVal(it.first.GetPointerPosition())

            normCount = self.fMC.GetPdf().expectedEvents(self.fMC.GetObservables())
            self.fMC.GetPdf().plotOn(
                self.fPlot, ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent), LineColor="b"
            )
            self.fPlot.Draw()

            self.fCanvas.GetCanvas().Modified()
            self.fCanvas.GetCanvas().Update()
            #########################/
        else:
            #########################/
            # else ( indentation belongs to "if simpdf")
            #########################/
            channelCat = simPdf.indexCat()
            #    TIterator* iter = simPdf->indexCat().typeIterator() ;
            frameIndex = 0
            global gchannelCat
            gchannelCat = channelCat
            for tt in channelCat:
                catName = tt.first

                frameIndex += 1
                self.fCanvas.GetCanvas().cd(frameIndex)

                # pre loop
                pdftmp = simPdf.getPdf(str(catName))
                obstmp = pdftmp.getObservables(self.fMC.GetObservables())
                global ggpdftmp, gobstmp
                ggpdftmp = pdftmp
                gobstmp = obstmp
                # return
                obs = obstmp.first()
                # fplotlist is a template, plotlist is the actual vector<RooPlot> with dim  numCats
                global gframeIndex
                gframeIndex = frameIndex
                global gfPlotList
                gfPlotList = self.fPlotList
                self.fPlot = self.fPlotList[frameIndex - 1]
                if self.fPlot:
                    del self.fPlot
                self.fPlot = obs.frame()

                self.fPlotList[frameIndex - 1] = self.fPlot
                # plotlist[(frameIndex - 1)] = fPlot

                msglevel = ROOT.RooMsgService.instance().globalKillBelow()
                ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
                # """
                self.fData.plotOn(
                    self.fPlot,
                    MarkerSize=1,
                    Cut=ROOT.Form("{:s}=={:s}::{:s}".format(channelCat.GetName(), channelCat.GetName(), str(catName))),
                    DataError="None",
                )
                # """

                # self.fData.plotOn(self.fPlot)

                ROOT.RooMsgService.instance().setGlobalKillBelow(msglevel)

                # normCount = Double_t()

                # high loop
                # it0 = self.fSliderMap.begin()
                for it in self.fSliderMap:
                    name = it.second
                    self.fWS.var(name).setVal(it.first.GetMaxPosition())
                    param = self.fWS.var(name)  # RooRealVar
                    self.fLabelMap[it.first].SetText(
                        ROOT.Form(
                            "{:s} = {:.3f} [{:.3f},{:.3f}]".format(
                                param.GetName(),
                                it.first.GetPointerPosition(),
                                it.first.GetMinPosition(),
                                it.first.GetMaxPosition(),
                            )
                        )
                    )
                # normCount = pdftmp.expectedEvents(obs)
                normCount = pdftmp.expectedEvents(obstmp)
                # normCount = pdftmp.expectedEvents(RooArgSet(obs))
                pdftmp.plotOn(
                    self.fPlot,
                    ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                    LineColor="r",
                    LineWidth=2,
                )

                # low loop
                # it0 = self.fSliderMap.begin()
                for it in self.fSliderMap:
                    name = it.second
                    self.fWS.var(name).setVal(it.first.GetMinPosition())
                    param = self.fWS.var(name)
                    self.fLabelMap[it.first].SetText(
                        ROOT.Form(
                            "{:s} = {:.3f} [{:.3f},{:.3f}]".format(
                                param.GetName(),
                                it.first.GetPointerPosition(),
                                it.first.GetMinPosition(),
                                it.first.GetMaxPosition(),
                            )
                        )
                    )

                # normCount = pdftmp.expectedEvents(RooArgSet(obs))
                normCount = pdftmp.expectedEvents(obstmp)
                pdftmp.plotOn(
                    self.fPlot,
                    ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                    LineColor="g",
                    LineWidth=2,
                )

                # central loop
                # it0 = self.fSliderMap.begin()
                for it in self.fSliderMap:
                    name = it.second
                    self.fWS.var(name).setVal(it.first.GetPointerPosition())
                    param = self.fWS.var(name)
                    self.fLabelMap[it.first].SetText(
                        ROOT.Form(
                            "{:s} = {:.3f} [{:.3f},{:.3f}]".format(
                                param.GetName(),
                                it.first.GetPointerPosition(),
                                it.first.GetMinPosition(),
                                it.first.GetMaxPosition(),
                            )
                        )
                    )

                # normCount = pdftmp.expectedEvents(RooArgSet(obs))
                normCount = pdftmp.expectedEvents(obstmp)
                if not self.fFitRes:
                    pass
                    global gnormCount
                    gnormCount = normCount
                    global gpdftmp
                    gpdftmp = pdftmp
                    pdftmp.plotOn(
                        self.fPlot,
                        ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                        LineColor="b",
                        LineWidth=2,
                    )
                    # pdftmp.plotOn(self.fPlot)
                else:
                    pdftmp.plotOn(
                        self.fPlot,
                        ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                        ROOT.RooFit.VisualizeError(self.fFitRes, self.fMC.GetNuisanceParameters()),
                        FillColor="y",
                    )
                    pdftmp.plotOn(
                        self.fPlot,
                        ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                        LineColor="b",
                        LineWidth=2,
                    )
                    msglevel = ROOT.RooMsgService.instance().globalKillBelow()
                    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
                    self.fData.plotOn(
                        self.fPlot,
                        MarkerSize=1,
                        Cut=ROOT.Form(
                            "{:s}=={:s}::{:s}".format(channelCat.GetName(), channelCat.GetName(), str(catName))
                        ),
                        DataError="None",
                    )

                    ROOT.RooMsgService.instance().setGlobalKillBelow(msglevel)

                self.fPlot.Draw()

            self.fCanvas.GetCanvas().Modified()
            self.fCanvas.GetCanvas().Update()
            ##############/
            # end if(simPdf)
            # return

    # ______________________________________________________________________________
    def HandleButtons(self):

        # Handle different buttons.

        # btn = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )
        # doesnt' work properly; since we are inside a function a new instance of ROOT is created.
        # then, the ROOT.gTQSender doesn't have any information.
        # self.gTQSender is a stored variable saved after super().__init__(ROOT.gClient.GetRoot(), ...)

        btn = ROOT.BindObject(self.gTQSender, ROOT.TGTextButton)
        Id = btn.WidgetId()
        match (Id):
            case ETestCmdId.HCId1.value:
                self.fHslider1.SetConstrained(self.fCheck1.GetState())
                pass
            case ETestCmdId.HCId2.value:
                self.fHslider1.SetRelative(self.fCheck2.GetState())
                pass
            case _:
                pass

    def DoExit(self):

        print("Exit application...")
        self.gApplication.Terminate(0)
        del self
        sys.exit()

    # ______________________________________________________________________________
    def __init__(self, w, mc, data):
        # ----------------------------------members---------------------
        # --- Principal Members ---
        self.fWS = w
        self.fMC = mc
        self.fData = data
        # --- Principal Members ---
        self.fCanvas = ROOT.TRootEmbeddedCanvas()
        self.fLcan = ROOT.TGLayoutHints()
        self.fFitFcn = ROOT.TF1()
        self.fPlot = ROOT.RooPlot()
        self.fFile = ROOT.TFile()
        self.fFitRes = ROOT.RooFitResult()
        # -
        self.fSliderList = ROOT.TList()
        self.fFrameList = ROOT.TList()
        self.fPlotList = ROOT.std.vector("RooPlot *")()
        self.fSliderMap = ROOT.std.map("TGTripleHSlider *", " TString ")()  # it is an instance
        # -
        # -TSliderMap =    ROOT.map(TGTripleHSlider , char )   # it is a type
        self.fLabelMap = ROOT.std.map("TGTripleHSlider*", "TGLabel *")()
        # -
        self.fFitButton = ROOT.TGButton()
        self.fExitButton = ROOT.TGTextButton()
        # -
        # -# BB: a TGCanvas and a vertical frame are needed for using scrollbars
        self.fCan = ROOT.TGCanvas()
        self.fVFrame = ROOT.TGVerticalFrame()
        # -
        self.fHframe0 = self.fHframe1 = self.fHframe2 = ROOT.TGHorizontalFrame()
        self.fBly = self.fBfly1 = self.fBfly2 = self.fBfly3 = ROOT.TGLayoutHints()
        self.fHslider1 = ROOT.TGTripleHSlider()
        self.fTbh1 = self.fTbh2 = self.fTbh3 = ROOT.TGTextBuffer()
        self.fCheck1 = self.fCheck2 = ROOT.TGCheckButton()
        # -------------------------------------------end of members---------------------
        # debugging
        # global gself
        # gself = self
        #
        # super(TGMainFrame, self).__init__(gClient.GetRoot(), 500, 500)
        # Initialize TGMainFrame and saving its pointers
        super().__init__(ROOT.gClient.GetRoot(), 500, 500)
        self.gTQSender = ROOT.gTQSender
        self.gApplication = ROOT.gApplication

        ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumericIntegration)

        simPdf = ROOT.nullptr
        numCats = 1
        # if (strcmp(fMC.GetPdf().ClassName(), "RooSimultaneous") == 0) : non-pythonic syntax
        if self.fMC.GetPdf().ClassName() == "RooSimultaneous":  # simple, pythonic syntax
            print(f"Is a simultaneous PDF")
            simPdf = self.fMC.GetPdf()
            channelCat = simPdf.indexCat()
            print(f" with {channelCat.numTypes()} categories")
            numCats = channelCat.numTypes()
        else:
            print(f"Is not a simultaneous PDF")

        self.fFitRes = ROOT.nullptr
        self.SetCleanup(ROOT.kDeepCleanup)

        # Create an embedded canvas and add it to the main frame with center at x and y
        # and with 30 pixel margin all around
        self.fCanvas = ROOT.TRootEmbeddedCanvas("Canvas", self, 600, 400)
        self.fLcan = ROOT.TGLayoutHints(ROOT.kLHintsExpandX | ROOT.kLHintsExpandY, 10, 10, 10, 10)
        self.AddFrame(self.fCanvas, self.fLcan)
        self.fPlotList.resize(numCats)
        # plotlist = self.fPlotList(numCats) # instead we create an instance of the template
        if numCats > 1:
            self.fCanvas.GetCanvas().Divide(numCats)
            for i in range(numCats):
                self.fCanvas.GetCanvas().cd(i + 1).SetBorderMode(0)
                self.fCanvas.GetCanvas().cd(i + 1).SetGrid()

        # return
        self.fHframe0 = ROOT.TGHorizontalFrame(self, 0, 0, 0)

        self.fCheck1 = ROOT.TGCheckButton(self.fHframe0, "&Constrained", ETestCmdId.HCId1.value)
        self.fCheck2 = ROOT.TGCheckButton(self.fHframe0, "&Relative", ETestCmdId.HCId2.value)
        self.fCheck1.SetState(ROOT.kButtonUp)
        self.fCheck2.SetState(ROOT.kButtonUp)
        self.fCheck1.SetToolTipText("Pointer position constrained to slider sides")
        self.fCheck2.SetToolTipText("Pointer position relative to slider position")

        self.fHframe0.Resize(200, 50)

        self.fHframe2 = ROOT.TGHorizontalFrame(self, 0, 0, 0)

        dp_DoFit = ROOT.TPyDispatcher(self.DoFit)
        self.fFitButton = ROOT.TGTextButton(self.fHframe2, "&Fit")
        self.fFitButton.SetFont("Helvetica")
        self.fFitButton.Connect("Clicked()", "TPyDispatcher", dp_DoFit, "Dispatch()")

        dp_DoExit = ROOT.TPyDispatcher(self.DoExit)
        self.fExitButton = ROOT.TGTextButton(self.fHframe2, "&Exit")
        self.fExitButton.SetFont("Helvetica")
        # self.fExitButton.Connect( "Clicked()", "TPyDispatcher", dp_DoExit , "Dispatch()")
        # doesn't work properly. Break segmentation violation. Full crash.
        self.fExitButton.SetCommand('TPython::Exec( "raise SystemExit" )')

        # dp_CloseWindow = TPyDispatcher( self.CloseWindow)
        # self.Connect("CloseWindow()", "TPyDispatcher", dp_CloseWindow, "Dispatch()")
        self.DontCallClose()

        dp_HandleButtons = ROOT.TPyDispatcher(self.HandleButtons)
        self.fCheck1.Connect("Clicked()", "TPyDispatcher", dp_HandleButtons, "Dispatch()")
        self.fCheck2.Connect("Clicked()", "TPyDispatcher", dp_HandleButtons, "Dispatch()")

        self.fHframe2.Resize(100, 25)

        # --- layout for buttons: top align, equally expand horizontally
        self.fBly = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsExpandX, 5, 5, 5, 5)

        # --- layout for the frame: place at bottom, right aligned
        self.fBfly1 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsCenterX, 5, 5, 5, 5)
        self.fBfly2 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsLeft, 5, 5, 5, 5)
        self.fBfly3 = ROOT.TGLayoutHints(ROOT.kLHintsTop | ROOT.kLHintsRight, 5, 5, 5, 5)

        self.fHframe2.AddFrame(self.fFitButton, self.fBfly2)
        self.fHframe2.AddFrame(self.fExitButton, self.fBfly3)

        self.AddFrame(self.fHframe0, self.fBly)
        self.AddFrame(self.fHframe2, self.fBly)

        # Loop over POI & NP, create slider
        # need maps of NP->slider? or just slider->NP
        parameters = ROOT.RooArgSet()
        parameters.add(self.fMC.GetParametersOfInterest())
        parameters.add(self.fMC.GetNuisanceParameters())
        # it = parameters.createIterator() # unnecessary
        param = ROOT.nullptr
        # BB: This is the part needed in order to have scrollbars
        self.fCan = ROOT.TGCanvas(self, 100, 100, ROOT.kFixedSize)
        self.AddFrame(self.fCan, ROOT.TGLayoutHints(ROOT.kLHintsExpandY | ROOT.kLHintsExpandX))
        self.fVFrame = ROOT.TGVerticalFrame(self.fCan.GetViewPort(), 10, 10)
        self.fCan.SetContainer(self.fVFrame)
        # And that's it!
        # Obviously, the parent of other subframes is now fVFrame instead of "self"...

        # while (param := it.Next()): #unnecessary
        for param in parameters:
            print(f"Adding Slider for ", param.GetName())
            hframek = ROOT.TGHorizontalFrame(self.fVFrame, 0, 0, 0)

            hlabel = ROOT.TGLabel(
                hframek, ROOT.Form("{:s} = {:.3f} +{:.3f}".format(param.GetName(), param.getVal(), param.getError()))
            )

            hsliderk = ROOT.TGTripleHSlider(
                hframek,
                0,
                ROOT.kDoubleScaleBoth,
                ETestCmdId.HSId1.value,
                ROOT.kHorizontalFrame,
                ROOT.TGFrame.GetDefaultFrameBackground(),
                False,
                False,
                False,
                False,
            )

            dp_DoSlider = ROOT.TPyDispatcher(self.DoSlider)
            hsliderk.Connect("PointerPositionChanged()", "TPyDispatcher", dp_DoSlider, "Dispatch()")
            hsliderk.Connect("PositionChanged()", "TPyDispatcher", dp_DoSlider, "Dispatch()")
            hsliderk.SetRange(param.getMin(), param.getMax())

            hframek.Resize(200, 25)
            self.fSliderList.Add(hsliderk)
            self.fFrameList.Add(hframek)

            hsliderk.SetPosition(param.getVal() - param.getError(), param.getVal() + param.getError())
            hsliderk.SetPointerPosition(param.getVal())

            hframek.AddFrame(hlabel, self.fBly)  #
            hframek.AddFrame(hsliderk, self.fBly)  #
            self.fVFrame.AddFrame(hframek, self.fBly)
            self.fSliderMap[hsliderk] = param.GetName()
            self.fLabelMap[hsliderk] = hlabel

        # Set main frame name, map sub windows (buttons), initialize layout
        # algorithm via Resize() and map main frame
        self.SetWindowName("RooFit/RooStats Model Inspector")
        self.MapSubwindows()
        self.Resize(self.GetDefaultSize())
        self.MapWindow()

        self.DoSlider()


# ----------------------------------------------------------------------------------------------------
def ModelInspector(infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"):

    # -------------------------------------------------------
    # First part is just to access a user-defined file
    # or create the standard example file if it doesn't exist

    filename = ""
    if infile == "":
        filename = "results/example_combined_GaussExample_model.root"
        fileExist = not ROOT.gSystem.AccessPathName(filename)  # note opposite return code
        # if file does not exists generate with histfactory
        if not fileExist:
            # Normally this would be run on the command line
            print(f"will run standard hist2workspace example")
            ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
            ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
            print(f"\n\n---------------------")
            print(f"Done creating example input")
            print(f"---------------------\n\n")

    else:
        filename = infile

    # Bad behaviour of variable, workspace, modelconfig. They get unset after its first call(being whatever)
    # if we declare pointer to the file, workspace, modelconfig, so everything seems to work-out fine.
    Declare = ROOT.gInterpreter.Declare
    Declare(
        """using namespace std;
              using namespace RooFit;
              using namespace RooStats;
              """
    )
    ##################################################
    # Try to open the file:
    # Not to use: file = TFile.Open(filename, "READ" )
    Declare(f'TFile *file = TFile::Open("{filename}");')
    file = ROOT.file

    # if input file was specified but not found, quit
    try:
        file.GetName()
    except ReferenceError:
        print("file wasn't load properly and is a nullptr")
        print(f"\nInput file {filename} was not found")
        return

    # -------------------------------------------------------
    # Tutorial starts here
    # -------------------------------------------------------
    # get the workspace out of the file
    # Not to use :w = file.Get(workspaceName) # RooWorkspace
    ProcessLine = ROOT.gInterpreter.ProcessLine
    ProcessLine(f'RooWorkspace *w = (RooWorkspace *)file->Get("{workspaceName}");')
    w = ROOT.w
    try:
        w.GetName()
    except ReferenceError:
        print(f"Workspace:{workspaceName} wasn't load properly and is a nullptr")
        return

    # get the modelConfig out of the file
    # Not to use: mc = w.obj(modelConfigName) # ModelConfig
    ProcessLine(f'ModelConfig *mc = (ModelConfig *)w->obj("{modelConfigName}");')
    mc = ROOT.mc

    # get the data out of the workspace
    # Not to use: data = w.data(dataName) # RooAbsData
    ProcessLine(f'RooAbsData *DATA = w->data("{dataName}");')
    data = ROOT.DATA

    # make sure ingredients are found
    try:
        mc.GetName()
        print("name", mc.GetPdf())
        mc.GetPdf()
        data.GetName()
    except ReferenceError:
        print(f"ModelConfig:{modelConfigName} wasn't load properly and is a nullptr")
        return
    try:
        data.GetName()
    except ReferenceError:
        print(f"data:{dataName} wasn't load properly and is a nullptr")
        return

    print("running ModelInspectorGUI...")
    ModelInspectorGUI(w, mc, data)


ModelInspector(infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData")
