import unittest

import ROOT


class SnapshotTests(unittest.TestCase):
    # Regression described in https://github.com/root-project/root/issues/20320#issuecomment-3553697692
    # This was caused by an iterator invalidation when snapshots with JIT-ted filters is used
    def test_snapshot(self):
        df = ROOT.RDataFrame(10)
        for var in ["pt", "eta", "phi", "pdgId", "mass", "tightId", "pfIsoId", "deltaEtaSC", "cutBased"]:
            df = df.Define("Muon_%s" % var, "ROOT::RVec<float>(2, 1.)")
            df = df.Define("Electron_%s" % var, "ROOT::RVec<float>(2, 1.)")
        for var in ["pt", "eta", "phi", "pdgId", "mass"]:
            for var2 in []:
                df = df.Define("Muon_good_%s" % var2, "ROOT::RVec<float>(2, 1.)")
            df = df.Define(
                "Muon_good_%s" % var,
                "Muon_%s[abs(Muon_eta) < 2.4 && Muon_pt > 0 && Muon_tightId && Muon_pfIsoId >= 0]" % var,
            )
        for var in ["pt", "eta", "phi", "pdgId", "mass"]:
            df = df.Define(
                "Electron_good_%s" % var,
                "Electron_%s[!(abs(Electron_eta+Electron_deltaEtaSC)>0 && abs(Electron_eta+Electron_deltaEtaSC)< 0) && abs(Electron_eta)<2.4 && Electron_pt > 0 && Electron_cutBased > 0]"
                % var,
            )

        df = df.Define("Muon_IDSF", "1+0.01*(Muon_pt-40)")
        df = df.Vary(
            "Muon_IDSF",
            "ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>({1+0.005*(Muon_pt-40), 1+0.02*(Muon_pt-40)})",
            ["down", "up"],
            "muon_unc",
        )
        df = df.Define("Electron_IDSF", "1+0.01*(Electron_pt-40)")
        df = df.Vary(
            "Electron_IDSF",
            "ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>({1+0.005*(Electron_pt-40), 1+0.02*(Electron_pt-40)})",
            ["down", "up"],
            "electron_unc",
        )

        df = df.Filter("(Muon_good_pt.size() + Electron_good_pt.size()) > 0")

        comprAlgo = getattr(ROOT.RCompressionSetting.EAlgorithm, "kZLIB")
        opts = ROOT.RDF.RSnapshotOptions("RECREATE", comprAlgo, 0, 0, 99, False)
        opts.fIncludeVariations = True

        snapshot = df.Snapshot("Events", "output.root", ["Electron_IDSF", "Muon_IDSF"], opts)
        self.assertIsNotNone(snapshot)

        with ROOT.TFile.Open("output.root") as f:
            tree = f.Get("Events")
            self.assertIsNotNone(tree)
            self.assertEqual(tree.GetEntries(), 10)


if __name__ == "__main__":
    unittest.main()
