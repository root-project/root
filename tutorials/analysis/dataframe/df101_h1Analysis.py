import ROOT

ROOT.EnableImplicitMT(4)


# Declare filters on RVec objects and JIT with Numba
@ROOT.Numba.Declare(["int", "int", "RVecI"], "bool")
def ik_ipi_nhitrp_cut(ik, ipi, nhitrp):
    return nhitrp[ik - 1] * nhitrp[ipi - 1] > 1


@ROOT.Numba.Declare(["int", "RVecF", "RVecF"], "bool")
def ik_rstart_rend_cut(ik, rstart, rend):
    return rend[ik - 1] - rstart[ik - 1] > 22


@ROOT.Numba.Declare(["int", "RVecF", "RVecF"], "bool")
def ipi_rstart_rend_cut(ipi, rstart, rend):
    return rend[ipi - 1] - rstart[ipi - 1] > 22


@ROOT.Numba.Declare(["int", "RVecF"], "bool")
def ik_nlhk_cut(ik, nlhk):
    return nlhk[ik - 1] > 0.1


@ROOT.Numba.Declare(["int", "RVecF"], "bool")
def ipi_nlhpi_cut(ipi, nlhpi):
    return nlhpi[ipi - 1] > 0.1


@ROOT.Numba.Declare(["int", "RVecF"], "bool")
def ipis_nlhpi_cut(ipis, nlhpi):
    return nlhpi[ipis - 1] > 0.1


def select(rdf):
    return (
        rdf.Filter("TMath::Abs(md0_d - 1.8646) < 0.04")
        .Filter("ptds_d > 2.5")
        .Filter("TMath::Abs(etads_d) < 1.5")
        .Filter("Numba::ik_ipi_nhitrp_cut(ik, ipi, nhitrp)")
        .Filter("Numba::ik_rstart_rend_cut(ik, rstart, rend)")
        .Filter("Numba::ipi_rstart_rend_cut(ipi, rstart, rend)")
        .Filter("Numba::ik_nlhk_cut(ik, nlhk)")
        .Filter("Numba::ipi_nlhpi_cut(ipi, nlhpi)")
        .Filter("Numba::ipis_nlhpi_cut(ipis, nlhpi)")
    )


dxbin = (0.17 - 0.13) / 40
condition = "x > 0.13957"
xp3 = "(x - [3]) * (x - [3])"


def FitAndPlotHdmd(hdmd: ROOT.TH1):
    ROOT.gStyle.SetOptFit()
    c1 = ROOT.TCanvas("c1", "h1analysis analysis", 10, 10, 800, 600)

    hdmd.GetXaxis().SetTitleOffset(1.4)

    hdraw = hdmd.DrawClone()

    # Fit histogram hdmd with function f5 using the loglikelihood option
    formula = f"{dxbin} * ([0] * pow(x - 0.13957, [1]) + [2] / 2.5066 / [4] * exp(-{xp3} / 2 / [4] / [4]))"
    f5 = ROOT.TF1("f5", f"{condition} ? {formula} : 0", 0.139, 0.17, 5)
    f5.SetParameters(1000000, 0.25, 2000, 0.1454, 0.001)
    hdraw.Fit(f5, "lr")

    c1.Update()

    return


def FitAndPlotH2(h2: ROOT.TH2):
    # Create the canvas for tau d0
    c2 = ROOT.TCanvas("c2", "tauD0", 100, 100, 800, 600)

    c2.SetGrid()
    c2.SetBottomMargin(0.15)

    # Project slices of 2-d histogram h2 along X , then fit each slice
    # with function f2 and make a histogram for each fit parameter
    # Note that the generated histograms are added to the list of objects
    # in the current directory.
    sigma = 0.0012
    formula = f"{dxbin} * ([0] * pow(x - 0.13957, 0.25) + [1] / 2.5066 / {sigma} * exp(-{xp3} / 2 / {sigma} / {sigma}))"
    print(f"TWO: {condition} ? {formula} : 0")

    f2 = ROOT.TF1("f2", f"{condition} ? {formula} : 0", 0.139, 0.17, 2)
    f2.SetParameters(10000, 10)
    h2.FitSlicesX(f2, 0, -1, 1, "qln")

    # See TH2::FitSlicesX documentation why h2_1 name is used
    h2_1 = ROOT.gDirectory.Get("h2_1")
    h2_1.SetDirectory(ROOT.nullptr)
    h2_1.GetXaxis().SetTitle("#tau [ps]")
    h2_1.SetMarkerStyle(21)
    h2_1.Draw()

    c2.Update()

    line = ROOT.TLine(0, 0, 0, c2.GetUymax())
    line.Draw()

    return


chain = ROOT.TChain("h42")
chain.Add("root://eospublic.cern.ch//eos/root-eos/h1/dstarmb.root")
chain.Add("root://eospublic.cern.ch//eos/root-eos/h1/dstarp1a.root")
chain.Add("root://eospublic.cern.ch//eos/root-eos/h1/dstarp1b.root")
chain.Add("root://eospublic.cern.ch//eos/root-eos/h1/dstarp2.root")

df = ROOT.RDataFrame(chain)
selected = select(df)
# Note: The title syntax is "<Title>;<Label x axis>;<Label y axis>"
hdmdARP = selected.Histo1D(("hdmd", "Dm_d;m_{K#pi#pi} - m_{K#pi}[GeV/c^{2}]", 40, 0.13, 0.17), "dm_d")
selected_added_branch = selected.Define("h2_y", "rpd0_t / 0.029979f * 1.8646f / ptd0_d")
h2ARP = selected_added_branch.Histo2D(("h2", "ptD0 vs Dm_d", 30, 0.135, 0.165, 30, -3, 6), "dm_d", "h2_y")

FitAndPlotHdmd(hdmdARP)
FitAndPlotH2(h2ARP)
