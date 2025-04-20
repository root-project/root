import unittest

import ROOT

# Development testing:

# c = ROOT.TCanvas()
# rng = np.random.RandomState(10)
# a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
# hist1 = ROOT.CreateHisto(name = "1D uniform binning", title = "helo world", bins = [200], data = a)
# hist1.Draw()
# c.SaveAs("1d_uniform.png")

# c = ROOT.TCanvas()
# var_bins = [np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])**2]

# hist2 = ROOT.CreateHisto(name = "1D variable binning", title = "helo world", bins = var_bins, data = a)

# hist2.Draw("1D Variable")
# c.SaveAs("1d_variable.png")


# c = ROOT.TCanvas()
# n = 10000
# x = np.linspace(1, 10, n)
# y = 2*np.log(x) + np.random.rand(n)
# a = np.stack((x, y), axis = 1)
# hist2 = ROOT.CreateHisto(name = "2D uniform binning", title = "helo world", bins = (100, 100), data = a)
# hist2.Draw()

# nbinx = hist2.GetXaxis()
# c.SaveAs("2d_uniform.png")

# c = ROOT.TCanvas()
# n = 10000

# var_bins = np.array(np.linspace(0, 3.16, 100))**2
# hist2 = ROOT.CreateHisto(name = "2D variable binning",
#                           title = "helo world",
#                           bins = (var_bins, 10),
#                           data = a)

# hist2.Draw()
# c.SaveAs("2d_variable.png")

# c = ROOT.TCanvas()
# n = 10000
# x = np.linspace(1, 10, n)                 
# y = 2 * np.log(x) + np.random.rand(n)      
# z = np.sin(x) + np.random.rand(n) * 0.5   

# a = np.stack((x, y, z), axis=1)

# hist3 = ROOT.CreateHisto(
#     name="3D uniform binning",
#     title="3D Example Histogram",
#     bins=(50, 50, 50),
#     data=a
# )

# hist3.Draw()
# c.SaveAs("3d_uniform.png")

# c = ROOT.TCanvas()
# n = 10000
# x = np.linspace(1, 10, n)                 
# y = 2 * np.log(x) + np.random.rand(n)      
# z = np.sin(x) + np.random.rand(n) * 0.5

# a = np.stack((x, y, z), axis=1)

# var_bins_x = np.array(np.linspace(0, 3.16, 50))**2
# var_bins_y = np.array(np.linspace(0, 20, 50))**2
# var_bins_z = np.array(np.linspace(0, 1.2, 50))**2

# hist3 = ROOT.CreateHisto(
#     name="3D variable binning",
#     title="helo world",
#     # bins=(50, 50, 50),
#     bins=(var_bins_x, var_bins_y, var_bins_z),
#     data=a
# )

# hist3.Draw()
# c.SaveAs("3d_variable.png")


class FillWithNumpyArray(unittest.TestCase):
    """
    Test for the FillN method of TH1 and subclasses, which fills
    the histogram with a numpy array.
    """

    # Tests
    def test_CreateTH1(self):
        import numpy as np
        # Create sample data'

        # c = ROOT.TCanvas()
        # rng = np.random.RandomState(10)
        # a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
        # hist1 = ROOT.CreateHisto(name = "1D uniform binning", title = "helo world", bins = [200], data = a)
        # hist1.Draw()
        # c.SaveAs("1d_uniform.png")


        data = np.array([1., 2, 2, 3, 3, 3, 4, 4, 5])
        # Create histograms
        nbins = 5
        min_val = 0
        max_val = 10
        hist_valid= ROOT.TH1F("1D Uniform Hist", "Validation Histogram", nbins, min_val, max_val)
        hist_numpy = ROOT.CreateHisto("1D Uniform Hist", "Numpy Histogram", np.array(np.linspace(0, 10, 5)), data = data)
        # Fill histograms
        hist_valid.FillN(len(data), data, np.ones(len(data)))

        # Test if the histograms have the same content
        for i in range(nbins):
            self.assertAlmostEqual(first = verbose_hist.GetBinContent(i), second = simple_hist.GetBinContent(i))

        # Test filling with weights
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        verbose_hist.FillN(len(data), data, weights)
        simple_hist.Fill(data, weights)
        for i in range(nbins):
            self.assertAlmostEqual(verbose_hist.GetBinContent(i), simple_hist.GetBinContent(i))
            
        # Test filling with weights with a different length
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            simple_hist.Fill(data, weights)

if __name__ == '__main__':
    unittest.main()
