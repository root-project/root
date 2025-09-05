namespace {

auto RooFitHS3_wsfactoryexpressions = R"({
    "ARGUS_dist": {
        "class": "RooArgusBG",
        "arguments": [
            "mass",
            "resonance",
            "slope",
            "power"
        ]
    },
    "bernstein_poly_dist": {
        "class": "RooBernstein",
        "arguments": [
            "x",
            "coefficients"
        ]
    },
    "bifurkated_gaussian_dist": {
        "class": "RooBifurGauss",
        "arguments": [
            "x",
            "mean",
            "sigmaL",
            "sigmaR"
        ]
    },
    "crystalball_dist": {
        "class": "RooCBShape",
        "arguments": [
            "m",
            "m0",
            "sigma",
            "alpha",
            "n"
        ]
    },
    "chebychev_dist": {
        "class": "RooChebychev",
        "arguments": [
            "x",
            "coefficients"
        ]
    },
    "gamma_dist": {
        "class": "RooGamma",
        "arguments": [
            "x",
            "gamma",
            "beta",
            "mu"
        ]
    },
    "gaussian_dist": {
        "class": "RooGaussian",
        "arguments": [
            "x",
            "mean",
            "sigma"
        ]
    },
    "normal_dist": {
        "class": "RooGaussian",
        "arguments": [
            "x",
            "mean",
            "sigma"
        ]
    },
    "interpolation0d": {
        "class": "RooStats::HistFactory::FlexibleInterpVar",
        "arguments": [
            "vars",
            "nom",
            "low",
            "high"
        ]
    },
    "landau_dist": {
        "class": "RooLandau",
        "arguments": [
            "x",
            "mean",
            "sigma"
        ]
    },
    "power_sum_dist": {
        "class": "RooPowerSum",
        "arguments": [
            "x",
            "coefficients",
            "exponents"
        ]
    },
    "product": {
        "class": "RooProduct",
        "arguments": [
            "factors"
        ]
    },
    "product_dist": {
        "class": "RooProdPdf",
        "arguments": [
            "factors"
        ]
    },
    "step": {
        "class": "ParamHistFunc",
        "arguments": [
            "variables",
            "parameters"
        ]
    },
    "sum": {
        "class": "RooAddition",
        "arguments": [
            "summands"
        ]
    },
    "uniform_dist": {
        "class": "RooUniform",
        "arguments": [
            "x"
        ]
    },
    "crystalball_doublesided_dist": {
        "class": "RooCrystalBall",
        "arguments": [
            "m", "m0", "sigma_L", "sigma_R", "alpha_L", "n_L", "alpha_R", "n_R"
        ]
    }
})";

} // namespace
