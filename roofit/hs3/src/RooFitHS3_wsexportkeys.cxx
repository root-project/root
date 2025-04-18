namespace {

auto RooFitHS3_wsexportkeys = R"({
    "RooAddition": {
        "type": "sum",
        "proxies": {
            "set": "summands"
        }
    },
    "RooArgusBG": {
        "type": "ARGUS_dist",
        "proxies": {
            "m": "mass",
            "m0": "resonance",
            "c": "slope",
            "p": "power"
        }
    },
    "RooBernstein": {
        "type": "bernstein_poly_dist",
        "proxies": {
            "coefList": "coefficients",
            "x": "x"
        }
    },
    "RooBifurGauss": {
        "type": "bifurkated_gaussian_dist",
        "proxies": {
            "x": "x",
            "mean": "mean",
            "sigmaL": "sigmaL",
            "sigmaR": "sigmaR"
        }
    },
    "RooCBShape": {
        "type": "crystalball_dist",
        "proxies": {
            "alpha": "alpha",
            "m": "m",
            "m0": "m0",
            "n": "n",
            "sigma": "sigma"
        }
    },
    "RooCrystalBall": {
        "type": "crystalball_doublesided_dist",
        "proxies": {
            "alphaL": "alpha_L",
            "alphaR": "alpha_R",
            "nL": "n_L",
            "nR": "n_R",
            "x": "m",
            "x0": "m0",
            "sigmaL": "sigma_L",
            "sigmaR": "sigma_R"
        }
    },
    "RooGamma": {
        "type": "gamma_dist",
        "proxies": {
            "x": "x",
            "gamma": "gamma",
            "beta": "beta",
            "mu": "mu"
        }
    },
    "RooGaussian": {
        "type": "gaussian_dist",
        "proxies": {
            "x": "x",
            "mean": "mean",
            "sigma": "sigma"
        }
    },
    "ParamHistFunc": {
        "type": "step",
        "proxies": {
            "dataVars": "variables",
            "paramSet": "parameters"
        }
    },
    "RooLandau": {
        "type": "landau_dist",
        "proxies": {
            "x": "x",
            "mean": "mean",
            "sigma": "sigma"
        }
    },
    "RooPowerSum": {
        "type": "power_sum_dist",
        "proxies": {
            "coefList": "coefficients",
            "expList": "exponents",
            "x": "x"
        }
    },
    "RooProdPdf": {
        "type": "product_dist",
        "proxies": {
            "pdfs": "factors"
        }
    },
    "RooProduct": {
        "type": "product",
        "proxies": {
            "compRSet": "factors",
            "compCSet": "factors"
        }
    },
    "RooUniform": {
        "type": "uniform_dist",
        "proxies": {
            "x": "x"
        }
    }
})";

} // namespace
