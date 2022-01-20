import unittest
import ROOT

msg = ROOT.RooMsgService.instance()
msg.setGlobalKillBelow(ROOT.RooFit.WARNING)


class TestHS3HistFactoryJSON(unittest.TestCase):
    def measurement(self, inputFileName="test_hs3_histfactory_json_input.root"):
        ROOT.gROOT.SetBatch(True)
        meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")
        meas.SetOutputFilePrefix("./results/example_")
        meas.SetPOI("mu")
        meas.AddConstantParam("Lumi")
        meas.SetLumi(1.0)
        meas.SetLumiRelErr(0.10)
        meas.SetExportOnly(False)
        meas.SetBinHigh(2)
        #        meas.AddConstantParam("syst1")
        chan = ROOT.RooStats.HistFactory.Channel("channel1")
        chan.SetData("data", inputFileName)
        chan.SetStatErrorConfig(0.01, "Poisson")
        sig = ROOT.RooStats.HistFactory.Sample("signal", "signal", inputFileName)
        sig.AddOverallSys("syst1", 0.95, 1.05)
        sig.AddNormFactor("mu", 1, -3, 5)
        chan.AddSample(sig)
        background1 = ROOT.RooStats.HistFactory.Sample("background1", "background1", inputFileName)
        background1.ActivateStatError("background1_statUncert", inputFileName)
        background1.AddOverallSys("syst2", 0.95, 1.05)
        chan.AddSample(background1)
        background2 = ROOT.RooStats.HistFactory.Sample("background2", "background2", inputFileName)
        background2.ActivateStatError()
        background2.AddOverallSys("syst3", 0.95, 1.05)
        chan.AddSample(background2)
        meas.AddChannel(chan)
        meas.CollectHistograms()
        return meas

    def toJSON(self, meas, fname):
        tool = ROOT.RooStats.HistFactory.JSONTool(meas)
        tool.PrintJSON(fname)

    def toWS(self, meas):
        w = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(meas)
        return w

    def test_create(self):
        meas = self.measurement()
        self.toJSON(meas, "hf.json")

    def importToWS(self, infile, wsname):
        ws = ROOT.RooWorkspace(wsname)
        tool = ROOT.RooJSONFactoryWSTool(ws)
        tool.importJSON(infile)
        return ws

    def test_closure(self):
        meas = self.measurement()
        self.toJSON(meas, "hf.json")
        ws = self.toWS(meas)
        ws_from_js = self.importToWS("hf.json", "ws1")

        noset = ROOT.RooArgSet()

        mc = ws["ModelConfig"]
        assert mc

        mc_from_js = ws_from_js["ModelConfig"]
        assert mc_from_js

        pdf = mc.GetPdf()
        assert pdf

        pdf_from_js = ws_from_js[meas.GetName()]
        assert pdf_from_js

        data = ws["obsData"]
        assert data

        data_from_js = ws_from_js["obsData"]
        assert data_from_js

        pdf.fitTo(data, Strategy=1, Minos=mc.GetParametersOfInterest(), GlobalObservables=mc.GetGlobalObservables())

        pdf_from_js.fitTo(
            data_from_js,
            Strategy=1,
            Minos=mc_from_js.GetParametersOfInterest(),
            GlobalObservables=mc_from_js.GetGlobalObservables(),
        )

        from math import isclose

        assert isclose(
            ws.var("mu").getVal(),
            ws_from_js.var("mu").getVal(),
            rel_tol=1e-4,
            abs_tol=1e-4,
        )
        assert isclose(
            ws.var("mu").getError(),
            ws_from_js.var("mu").getError(),
            rel_tol=1e-4,
            abs_tol=1e-4,
        )

    def test_closure_loop(self):
        meas = self.measurement()
        ws = self.toWS(meas)

        tool = ROOT.RooJSONFactoryWSTool(ws)
        js = tool.exportJSONtoString()

        newws = ROOT.RooWorkspace("new")
        newtool = ROOT.RooJSONFactoryWSTool(newws)
        newtool.importJSONfromString(js)

        mc = ws["ModelConfig"]
        assert mc

        newmc = newws["ModelConfig"]
        assert newmc

        pdf = mc.GetPdf()
        assert pdf

        newpdf = newmc.GetPdf()
        assert newpdf

        data = ws["obsData"]
        assert data

        newdata = newws["obsData"]
        assert newdata

        pdf.fitTo(
            data,
            Strategy=1,
            Minos=mc.GetParametersOfInterest(),
            GlobalObservables=mc.GetGlobalObservables(),
            PrintLevel=-1,
        )

        newpdf.fitTo(
            newdata,
            Strategy=1,
            Minos=newmc.GetParametersOfInterest(),
            GlobalObservables=newmc.GetGlobalObservables(),
            PrintLevel=-1,
        )

        from math import isclose

        assert isclose(
            ws.var("mu").getVal(),
            newws.var("mu").getVal(),
            rel_tol=1e-4,
            abs_tol=1e-4,
        )
        assert isclose(
            ws.var("mu").getError(),
            newws.var("mu").getError(),
            rel_tol=1e-4,
            abs_tol=1e-4,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
