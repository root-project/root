# \file
# \ingroup tutorial_roostats
# \notebook -js
# StandardHistFactoryPlotsWithCategories
#
#  This is a standard demo that can be used with any ROOT file
#  prepared in the standard way.  You specify:
#  - name for input ROOT file
#  - name of workspace inside ROOT file that holds model and data
#  - name of ModelConfig that specifies details for calculator tools
#  - name of dataset
#
#  With default parameters the macro will attempt to run the
#  standard hist2workspace example and read the ROOT file
#  that it produces.
#
#  The macro will scan through all the categories in a simPdf find the corresponding
#  observable.  For each category, it will loop through each of the nuisance parameters
#  and plot
#  - the data
#  - the nominal model (blue)
#  - the +Nsigma (red)
#  - the -Nsigma (green)
#
#  You can specify how many sigma to vary by changing nSigmaToVary.
#  You can also change the signal rate by changing muVal.
#
#  The script produces a lot plots, you can merge them by doing:
#
#  gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=merged.pdf `ls *pdf`
# ~~~
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)


import ROOT


def StandardHistFactoryPlotsWithCategories(
    infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"
):

    nSigmaToVary = 5.0
    muVal = 0
    doFit = False

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

    # Try to open the file
    file = ROOT.TFile.Open(filename)

    # if input file was specified but not found, quit
    if not file:
        print(f"StandardRooStatsDemoMacro: Input file {filename} is not found")
        return

    # -------------------------------------------------------
    # Tutorial starts here
    # -------------------------------------------------------

    # get the workspace out of the file
    w = file.Get(workspaceName)
    if not w:
        print(f"workspace not found")
        return

    # get the modelConfig out of the file
    mc = w.obj(modelConfigName)

    # get the modelConfig out of the file
    data = w.data(dataName)

    # make sure ingredients are found
    if not data or not mc:
        w.Print()
        print(f"data or ModelConfig was not found")
        return

    # -------------------------------------------------------
    # now use the profile inspector

    obs = mc.GetObservables().first()
    frameList = []

    firstPOI = mc.GetParametersOfInterest().first()

    firstPOI.setVal(muVal)
    #  firstPOI.setConstant()
    if doFit:
        mc.GetPdf().fitTo(data)

    # -------------------------------------------------------

    mc.GetNuisanceParameters().Print("v")
    nPlotsMax = 1000
    print(f" check expectedData by category")
    simData = ROOT.kNone
    simPdf = ROOT.kNone
    if str(mc.GetPdf().ClassName()) == "RooSimultaneous":
        print(f"Is a simultaneous PDF")
        simPdf = mc.GetPdf()
    else:
        print(f"Is not a simultaneous PDF")

    if doFit:
        channelCat = simPdf.indexCat()
        catName = channelCat.begin().first
        pdftmp = (mc.GetPdf()).getPdf(str(catName))
        obstmppdftmp.getObservables(mc.GetObservables())
        obs = obstmp.first()
        frame = obs.frame()
        print("{0}=={0}::{1}".format(channelCat.GetName(), catName))
        print(catName, " ", channelCat.getLabel())
        data.plotOn(
            frame,
            MarkerSize=1,
            Cut="{0}=={0}::{1}".format(channelCat.GetName(), catName),
            DataError="None",
        )

        normCount = data.sumEntries("{0}=={0}::{1}".format(channelCat.GetName(), catName))

        pdftmp.plotOn(frame, Normalization(normCount, ROOT.RooAbsReal.NumEvent), LineWidth=2)
        frame.Draw()
        print("expected events = ", mc.GetPdf().expectedEvents(data.get()))
        return

    nPlots = 0
    if not simPdf:

        for var in mc.GetNuisanceParameters():
            frame = obs.frame()
            frame.SetYTitle(var.GetName())
            data.plotOn(frame, MarkerSize(1))
            var.setVal(0)
            mc.GetPdf().plotOn(frame, LineWidth(1))
            var.setVal(1)
            mc.GetPdf().plotOn(frame, LineColor(kRed), LineStyle(kDashed), LineWidth(1))
            var.setVal(-1)
            mc.GetPdf().plotOn(frame, LineColor(kGreen), LineStyle(kDashed), LineWidth(1))
            frameList.append(frame)
            var.setVal(0)

    else:
        channelCat = simPdf.indexCat()
        for tt in channelCat:

            if nPlots == nPlotsMax:
                break

            catName = tt.first

            print("on type ", catName, " ")
            # Get pdf associated with state from simpdf
            pdftmp = simPdf.getPdf(str(catName))

            # Generate observables defined by the pdf associated with this state
            obstmp = pdftmp.getObservables(mc.GetObservables())
            #      obstmp.Print()

            obs = obstmp.first()

            for var in mc.GetNuisanceParameters():

                if nPlots >= nPlotsMax:
                    break

                c2 = ROOT.TCanvas("c2")
                frame = obs.frame()
                frame.SetName(f"frame{nPlots}")
                frame.SetYTitle(var.GetName())

                cut = "{0}=={0}::{1}".format(channelCat.GetName(), catName)
                print(cut)
                print(catName, " ", channelCat.getLabel())
                data.plotOn(
                    frame,
                    MarkerSize=1,
                    Cut=cut,
                    DataError="None",
                )

                normCount = data.sumEntries(cut)

                if str(var.GetName()) == "Lumi":
                    print(f"working on lumi")
                    var.setVal(w.var("nominalLumi").getVal())
                    var.Print()
                else:
                    var.setVal(0)

                # w.allVars().Print("v")
                # mc.GetNuisanceParameters().Print("v")
                # pdftmp.plotOn(frame,LineWidth(2))
                # mc.GetPdf().plotOn(frame,LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
                # pdftmp.plotOn(frame,LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
                global gobs
                gobs = obs
                global gpdftmp
                gpdftmp = pdftmp

                # notes: obs is RooRealVar / obstmp is RooArgSet
                #       pdftmp.expectedEvents receives RooArgSet as an argument
                #       in C++ automatic conversion is possible.
                #       in python the conversion is not possible.
                # C-code : normCount = pdftmp->expectedEvents(*obs);
                # Python : normCount = pdftmp.expectedEvents(obs) #doesn't work properly
                # RooArgSet(obs) doesn´t reproduce well the results
                # instead, we have to use obstmp
                # normCount = pdftmp.expectedEvents(RooArgSet(obstmp)) #doesn´t work properly
                normCount = pdftmp.expectedEvents(obstmp)
                pdftmp.plotOn(frame, ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent), LineWidth=2)

                if str(var.GetName()) == "Lumi":
                    print(f"working on lumi")
                    var.setVal(w.var("nominalLumi").getVal() + 0.05)
                    var.Print()
                else:
                    var.setVal(nSigmaToVary)

                # pdftmp.plotOn(frame,LineColor(kRed),LineStyle(kDashed),LineWidth(2))
                # mc.GetPdf().plotOn(frame,LineColor(kRed),LineStyle(kDashed),LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
                # pdftmp.plotOn(frame,LineColor(kRed),LineStyle(kDashed),LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
                normCount = pdftmp.expectedEvents(obstmp)
                pdftmp.plotOn(
                    frame,
                    ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                    LineWidth=2,
                    LineColor="r",
                    LineStyle="--",
                )

                if str(var.GetName()) == "Lumi":
                    print(f"working on lumi")
                    var.setVal(w.var("nominalLumi").getVal() - 0.05)
                    var.Print()
                else:
                    var.setVal(-nSigmaToVary)

                # pdftmp.plotOn(frame,LineColor(kGreen),LineStyle(kDashed),LineWidth(2))
                # mc.GetPdf().plotOn(frame,LineColor(kGreen),LineStyle(kDashed),LineWidth(2),Slice(channelCat,catName.c_str()),ProjWData(data))
                # pdftmp.plotOn(frame,LineColor(kGreen),LineStyle(kDashed),LineWidth(2),Slice(channelCat,catName.c_str()),ProjWData(data))
                normCount = pdftmp.expectedEvents(obstmp)
                pdftmp.plotOn(
                    frame,
                    ROOT.RooFit.Normalization(normCount, ROOT.RooAbsReal.NumEvent),
                    LineWidth=2,
                    LineColor="g",
                    LineStyle="--",
                )

                # set them back to normal
                if str(var.GetName()) == "Lumi":
                    print(f"working on lumi")
                    var.setVal(w.var("nominalLumi").getVal())
                    var.Print()
                else:
                    var.setVal(0)

                frameList.append(frame)

                # quit making plots
                nPlots += 1

                frame.Draw()
                c2.Update()
                c2.Draw()
                c2.SaveAs(f"StandardHistFactoryPlotsWithCategories.1.{catName}_{obs.GetName()}_{var.GetName()}.png")
                del c2

    # -------------------------------------------------------

    # now make plots
    c1 = ROOT.TCanvas("c1", "ProfileInspectorDemo", 800, 200)
    nFrames = len(frameList)
    if nFrames > 4:
        nx = int(sqrt(nFrames))
        ny = ROOT.TMath.CeilNint(nFrames / nx)
        nx = ROOT.TMath.CeilNint(sqrt(nFrames))
        c1.Divide(ny, nx)
    else:
        c1.Divide(nFrames)
    for i in range(nFrames):
        c1.cd(i + 1)
        frameList[i].Draw()
        c1.Update()
    c1.Draw()
    c1.SaveAs("StandardHistFactoryPlotsWithCategories.2.pdf")

    file.Close()


StandardHistFactoryPlotsWithCategories(
    infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"
)
