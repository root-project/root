import unittest
import ROOT
from ROOT import RDF

class RDataFrameHistoProfile(unittest.TestCase):
    '''
    Tests for the pythonization of HistoXD and ProfileXD methods.
    '''

    col = 'rdfentry_'
    coltype = 'ULong64_t'
    title = 'HistoProfile Test'
    nentries = 2

    method_to_args = {
        'Histo1D' : (('mh1d', title, 2, 0, 2), (col,), (coltype,)),
        'Histo2D' : (('mh2d', title, 2, 0, 2, 2, 0, 2), (col,)*2, (coltype,)*2),
        'Histo3D' : (('mh3d', title, 2, 0, 2, 2, 0, 2, 2, 0, 2), (col,)*3, (coltype,)*3),
        'Profile1D' : (('mp1d', title, 2, 0, 2), (col,)*2, (coltype,)*2),
        'Profile2D' : (('mp2d', title, 2, 0, 2, 2, 0, 2), (col,)*3, (coltype,)*3),
    }

    def test_tuple_to_model(self):
        '''
        Test conversion of model constructor arguments, passed as a tuple, to
        an actual model object.
        '''
        df = ROOT.RDataFrame(self.nentries)
        for method, args in self.method_to_args.items():
            model_args, col_args, _ = args
            # Test model arg as tuple
            res = getattr(df, method)(model_args, *col_args).GetValue()
            self.assertEqual(res.GetEntries(), self.nentries)
            self.assertEqual(res.GetTitle(), self.title)

    def test_template_instantiation(self):
        '''
        Test that the pythonized methods support being subscripted (explicit
        template instantiation).
        '''
        df = ROOT.RDataFrame(self.nentries)
        for method, args in self.method_to_args.items():
            model_args, col_args, col_types = args
            # Test instantiation + model arg as tuple
            res = getattr(df, method)[col_types](model_args, *col_args).GetValue()
            self.assertEqual(res.GetEntries(), self.nentries)
            self.assertEqual(res.GetTitle(), self.title)


if __name__ == '__main__':
    unittest.main()
