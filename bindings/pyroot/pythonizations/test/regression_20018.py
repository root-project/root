import unittest

import ROOT


class Regression20018(unittest.TestCase):
    """
    Regression test for https://github.com/root-project/root/issues/20018

    Calling ``TColor.DefinedColors(1)`` puts the global color bookkeeping into
    the "always store colors" mode (``gLastDefinedColors = -1``). When a canvas
    is afterwards serialized to JSON for the web display - which is exactly what
    JupyROOT does to show a canvas inline in a notebook - ``TCanvas::Streamer``
    used to unconditionally call ``fPrimitives->Add(...)`` to attach the list of
    colors. During the web-canvas snapshot the pad primitives are temporarily
    detached (``fPrimitives`` is set to ``nullptr``, see
    ``TWebCanvas::CreatePadSnapshot``), so this dereferenced a null pointer and
    crashed with a segmentation violation.
    """

    def test_definedcolors_web_canvas_json(self):
        # TWebCanvas (and thus CreateCanvasJSON) is only available when ROOT is
        # built with the web display (root7/webgui). Without it, JupyROOT uses a
        # different code path that is not affected by this issue.
        if not hasattr(ROOT, "TWebCanvas") or not hasattr(ROOT.TWebCanvas, "CreateCanvasJSON"):
            self.skipTest("ROOT was built without the web display (webgui)")

        ROOT.gROOT.SetBatch(True)

        # This is what triggers the bug: force the "always store colors" mode.
        ROOT.TColor.DefinedColors(1)

        c = ROOT.TCanvas("c_regression_20018", "Basic ROOT Plot", 800, 600)
        h = ROOT.TH1F("h_regression_20018", "Example Histogram;X axis;Entries", 100, 0, 10)
        h.FillRandom("gaus", 1000)
        h.Draw()
        c.Update()

        # Used to segfault in TCanvas::Streamer due to a null fPrimitives.
        json = ROOT.TWebCanvas.CreateCanvasJSON(c, 23, True)
        self.assertGreater(len(json.Data()), 0)


if __name__ == "__main__":
    unittest.main()
