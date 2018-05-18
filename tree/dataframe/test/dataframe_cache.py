import unittest
import ROOT

RDataFrame = ROOT.ROOT.RDataFrame

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
        tdf = RDataFrame("t", "fileName.root")
        ColType_t = "std::span<float>"
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
       tdf = RDataFrame("t", "fileName.root")
       cache = tdf.Cache("arr")
       arr = cache.Take('std::vector<float>')("arr")
       for e in arr:
           for i in xrange(4):
               self.assertEqual(float(i), e[i]);

    def test_EntryAndSlotColumns(self):
       tdf = RDataFrame(8)
       c = tdf.Filter("tdfentry_ % 2 == 0").Define("myEntry","tdfentry_").Cache()
       ds_entries = c.Take('UInt64_t')('tdfentry_')
       ref_ds_entries = [0,1,2,3]
       for e, eref in zip (ds_entries, ref_ds_entries):
           self.assertEqual(e, eref)
       old_entries = c.Take('UInt64_t')('tdfentry_')
       ref_old_entries = [0,2,4,6]
       for e, eref in zip (old_entries, ref_old_entries):
           self.assertEqual(e, eref)

if __name__ == '__main__':
    unittest.main()
