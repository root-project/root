import unittest

import ROOT


class TestTMVA(unittest.TestCase):
    """
    Test for Factory Constructor and Bookmethod pythonizations.
    """

    outputFile = ROOT.TFile.Open("TMVA.root", "RECREATE")
    factory = ROOT.TMVA.Factory("tmva003", outputFile, "!V:!DrawProgressBar:AnalysisType=Classification")

    def test_factory_constructor(self):
        outputFile = self.outputFile
        factory = self.factory
        factory_test = ROOT.TMVA.Factory(
            "tmva003", outputFile, V=False, DrawProgressBar=False, AnalysisType="Classification"
        )
        self.assertEqual(factory.GetOptions(), factory_test.GetOptions())

    def test_factory_bookmethod(self):
        filename = "http://root.cern.ch/files/tmva_class_example.root"
        data = ROOT.TFile.Open(filename)
        signal = data.Get("TreeS")
        background = data.Get("TreeB")

        # Add variables and register the trees with the dataloader
        dataloader = ROOT.TMVA.DataLoader("tmva003_BDT")
        variables = ["var1", "var2", "var3", "var4"]

        for var in variables:
            dataloader.AddVariable(var)

        dataloader.AddSignalTree(signal, 1.0)
        dataloader.AddBackgroundTree(background, 1.0)
        dataloader.PrepareTrainingAndTestTree("", "")

        # Train a TMVA method
        factory = self.factory
        factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", V=False, H=False, NTrees=300, MaxDepth=2)

    def test_preparetrainingandtesttree(self):
        dataloader = ROOT.TMVA.DataLoader("dataset")
        mycuts = ROOT.TCut("")
        mycutb = ROOT.TCut("")
        dataloader.AddVariable("myvar1 := var1+var2", "F")
        dataloader.AddVariable("myvar2 := var1-var2", "Expression 2", "", "F")
        dataloader.AddVariable("var3", "Variable 3", "units", "F")
        dataloader.AddVariable("var4", "Variable 4", "units", "F")
        dataloader.PrepareTrainingAndTestTree(
            mycuts,
            mycutb,
            nTrain_Signal=1000,
            nTrain_Background=1000,
            SplitMode="Random",
            NormMode="NumEvents",
            V=False,
        )

    def test_constructor(self):
        outputFile = ROOT.TFile.Open("TMVARegCv.root", "RECREATE")
        inputFile = ROOT.TFile.Open("http://root.cern.ch/files/tmva_reg_example.root", "CACHEREAD")
        dataloader = ROOT.TMVA.DataLoader("dataset")
        dataloader.AddVariable("var1", "Variable 1", "units", "F")
        dataloader.AddVariable("var2", "Variable 2", "units", "F")
        dataloader.AddTarget("fvalue")
        regTree = inputFile.Get("TreeR")
        dataloader.AddRegressionTree(regTree, 1.0)
        dataloader.PrepareTrainingAndTestTree("", nTest_Regression=1, SplitMode="Block", NormMode="NumEvents", V=False)

        analysisType = ROOT.TString("Regression")
        useRandomSplitting = False
        splitType = "Random" if useRandomSplitting else "Deterministic"
        splitExpr = "int(fabs([eventID]))%int([NumFolds])" if not useRandomSplitting else ""
        cv = ROOT.TMVA.CrossValidation(
            "TMVACrossValidationRegression",
            dataloader,
            outputFile,
            V=False,
            Silent=False,
            ModelPersistence=True,
            FoldFileOutput=False,
            AnalysisType=analysisType.Data(),
            SplitType=splitType,
            NumFolds="2",
            SplitExpr=splitExpr,
        )
