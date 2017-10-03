import unittest
import ROOT

TDataFrame = ROOT.ROOT.Experimental.TDataFrame

class Cache(unittest.TestCase):
    @classmethod
    def setUp(cls):
      code = ''' {
      const char* treeName = "t";
      const char* fileName = "fileName.root";
      TFile f(fileName, "RECREATE");
      TTree t(treeName, treeName);
      float arr[4];
      t.Branch("arr", arr, "arr[4]/F");
      for (auto i : ROOT::TSeqU(4)) {
         for (auto j : ROOT::TSeqU(4)) {
            arr[j] = i + j;
         }
         t.Fill();
      }
      t.Write();
      }'''
      ROOT.gInterpreter.Calc(code)

    def test_TakeArrays(self):
        tdf = TDataFrame("t", "fileName.root")
        ColType_t = "std::array_view<float>"
        v = tdf.Take(ColType_t)("arr").GetVal()
        d = tdf.Take(ColType_t+", std::deque("+ColType_t+")")("arr").GetVal()
        # commented out until we do not understand iteration
        #l = tdf.Take(ColType_t+", std::list("+ColType_t+")")("arr").GetVal()


        ifloat = 0.
        for i in xrange(4):
            vv = v[i];
            dv = d[i];
            for j in xrange(4):
                ref = ifloat + j;
                self.assertEqual(ref, vv[j]);
                self.assertEqual(ref, dv[j]);

    def test_Carrays(self):
       tdf = TDataFrame("t", "fileName.root")
       cache = tdf.Cache("arr")
       arr = cache.Take('std::vector<float>')("arr")
       for e in arr:
           for i in xrange(4):
               self.assertEqual(float(i), e[i]);

if __name__ == '__main__':
    unittest.main()
