import unittest
import ROOT
import os

RDataFrame = ROOT.ROOT.RDataFrame
RDatasetSpec = ROOT.RDF.Experimental.RDatasetSpec
REntryRange = ROOT.RDF.Experimental.RDatasetSpec.REntryRange

class RDatasetSpecTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # Necessary because in the new Cppyy if we want to instantiate
        # a templated method with a type written as a string (e.g. 'float')
        # we need to pass it in square brackets, otherwise it can be
        # (mis)interpreted as string parameter and the method itself is
        # called with 'float' as a parameter.
        # For example the Take() method mentioned multiple times in this test
        # has to be called e.g. with:
        # Take['float']()
        # instead of:
        # Take('float')()
        cls.legacy_pyroot = os.environ.get('LEGACY_PYROOT') == 'on'

        # reuse the code from the C++ unit tests to create some files
        code = ''' {
        auto dfWriter0 = ROOT::RDataFrame(5).Define("z", [](ULong64_t e) { return e + 100; }, {"rdfentry_"});
        dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree", "PYspecTestFile2.root", {"z"});
        dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree", "PYspecTestFile3.root", {"z"});
        dfWriter0.Range(4, 5).Snapshot<ULong64_t>("subTree", "PYspecTestFile4.root", {"z"});
        dfWriter0.Range(0, 2).Snapshot<ULong64_t>("subTree1", "PYspecTestFile5.root", {"z"});
        dfWriter0.Range(2, 4).Snapshot<ULong64_t>("subTree2", "PYspecTestFile6.root", {"z"});
        dfWriter0.Snapshot<ULong64_t>("anotherTree", "PYspecTestFile7.root", {"z"});
        }'''
        ROOT.gInterpreter.Calc(code)

    def tearDown(self):
        for i in range(2, 8):
            ROOT.gSystem.Unlink("PYspecTestFile" + str(i) + ".root")

    # a test involving all 3 constructors, all possible valid ranges, all possible friends
    def test_General(self):
        ranges = [REntryRange(), REntryRange(1, 4), REntryRange(2,4), REntryRange(100), REntryRange(1, 100),
                  REntryRange(2, 2), REntryRange(7, 7)]
        expectedRess = [[100, 101, 102, 103, 104], [101, 102, 103], [102, 103], [100, 101, 102, 103, 104],
                        [101, 102, 103, 104], [], []]
        specs = []
        
        for r in ranges:
            # for each range add all constructors
            specs.append([])
            specs[-1].append(RDatasetSpec("anotherTree", "PYspecTestFile7.root", r))
            specs[-1].append(RDatasetSpec("subTree", ["PYspecTestFile2.root",
                                                 "PYspecTestFile3.root",
                                                 "PYspecTestFile4.root"], r))
            specs[-1].append(RDatasetSpec([("subTree1", "PYspecTestFile5.root"),
                                      ("subTree2", "PYspecTestFile6.root"),
                                      ("subTree", "PYspecTestFile4.root")], r))

            # for each spec constructor add all possible combination of friends
            for s in specs[-1]:
                s.AddFriend("anotherTree", "PYspecTestFile7.root", "friendTree")
                s.AddFriend("subTree", ["PYspecTestFile2.root",
                                        "PYspecTestFile3.root",
                                        "PYspecTestFile4.root"], "friendChain1")
                s.AddFriend([("subTree1", "PYspecTestFile5.root"),
                             ("subTree2", "PYspecTestFile6.root"),
                             ("subTree", "PYspecTestFile4.root")], "friendChainN")

        for i in range(len(ranges)):
            for s in specs[i]:
                rdf = RDataFrame(s)

                if self.legacy_pyroot:
                    resP = rdf.Take("ULong64_t")("z")
                    fr1P = rdf.Take("ULong64_t")("friendTree.z")
                    fr2P = rdf.Take("ULong64_t")("friendChain1.z")
                    fr3P = rdf.Take("ULong64_t")("friendChainN.z")
                else:
                    resP = rdf.Take["ULong64_t"]("z")
                    fr1P = rdf.Take["ULong64_t"]("friendTree.z")
                    fr2P = rdf.Take["ULong64_t"]("friendChain1.z")
                    fr3P = rdf.Take["ULong64_t"]("friendChainN.z")

                res = resP.GetValue()
                fr1 = fr1P.GetValue()
                fr2 = fr2P.GetValue()
                fr3 = fr3P.GetValue()

                for j in range(len(expectedRess[i])):
                    self.assertEqual(len(res), len(expectedRess[i]))
                    self.assertEqual(res[j], expectedRess[i][j])
                    self.assertEqual(fr1[j], expectedRess[i][j])
                    self.assertEqual(fr2[j], expectedRess[i][j])
                    self.assertEqual(fr3[j], expectedRess[i][j])

if __name__ == '__main__':
    unittest.main()
