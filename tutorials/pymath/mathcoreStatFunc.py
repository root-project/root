# \file
# \ingroup tutorial_math
# \notebook
# Example-script showing some bidimensional probability density functions implemented in ROOT.
#
# The macro shows four of them(there are many others in ROOT):
#    Cauchy, Chi-Squred, Gaussian, tDistribution ;
# which are bidimensional distributions.
#
# In order to run the macro type:
#
# ~~~{.cpp}
#   root [0] .x mathcoreStatFunc.C
# ~~~
#
# \macro_image
# \macro_code
#
# \author Andras Zsenei

import ROOT

TF2 = ROOT.TF2
TSystem = ROOT.TSystem
TCanvas = ROOT.TCanvas


def mathcoreStatFunc():
   
   # All ROOT-objects must be defined as globally. 
   # Otherwise they will be destroyed by python after
   # its execution. This way we ensure to manipulate 
   # TCanvas-object dynamically such that in root[]. 
   # Take into account not only canvases must be defined
   # globally but legends, functions, ... Whatever stuff
   # you would like to include in your canvas must be saved,
   # i.e. --defined globally--.
   # 
   # Numbers, strings, are not important.

   global f1a, f2a, f3a, f4a
   f1a =  TF2("f1a","ROOT::Math::cauchy_pdf(x, y)",0,10,0,10)
   f2a =  TF2("f2a","ROOT::Math::chisquared_pdf(x,y)",0,20, 0,20)
   f3a =  TF2("f3a","ROOT::Math::gaussian_pdf(x,y)",0,10,0,5)
   f4a =  TF2("f4a","ROOT::Math::tdistribution_pdf(x,y)",0,10,0,5)

   #Note:
   #      The the second argument in TF2 requires C-format.
   #      Otherwise, will raise error.
   #     
   #      For your information:
   #      There is a list above with all available distributions 
   #      in root version > 6.30.
   #      
   #      Use 'help(ROOT.Math._the_distribution_you_want_to_know_more_)' 
   #      for its proper argument implementation. Some are one-dimensional
   #      other bi-dimensional.  
   #      Of course, this list doesnt't include hyperbolic-trigonometric functions.
   #      or Special Functions like Legendre, Hermite, ... except for those normalized.
   #      
   #      ROOT::Math::airy_Ai
   #      ROOT::Math::airy_Ai_deriv
   #      ROOT::Math::airy_Bi
   #      ROOT::Math::airy_Bi_deriv
   #      ROOT::Math::airy_zero_Ai
   #      ROOT::Math::airy_zero_Ai_deriv
   #      ROOT::Math::airy_zero_Bi
   #      ROOT::Math::airy_zero_Bi_deriv
   #      ROOT::Math::assoc_laguerre
   #      ROOT::Math::assoc_legendre
   #      ROOT::Math::beta
   #      ROOT::Math::beta_cdf
   #      ROOT::Math::beta_cdf_c
   #      ROOT::Math::beta_pdf
   #      ROOT::Math::beta_quantile
   #      ROOT::Math::beta_quantile_c
   #      ROOT::Math::bigaussian_pdf
   #      ROOT::Math::binomial_cdf
   #      ROOT::Math::binomial_cdf_c
   #      ROOT::Math::binomial_pdf
   #      ROOT::Math::breitwigner_cdf
   #      ROOT::Math::breitwigner_cdf_c
   #      ROOT::Math::breitwigner_pdf
   #      ROOT::Math::breitwigner_quantile
   #      ROOT::Math::breitwigner_quantile_c
   #      ROOT::Math::cauchy_cdf
   #      ROOT::Math::cauchy_cdf_c
   #      ROOT::Math::cauchy_pdf
   #      ROOT::Math::cauchy_quantile
   #      ROOT::Math::cauchy_quantile_c
   #      ROOT::Math::chisquared_cdf
   #      ROOT::Math::chisquared_cdf_c
   #      ROOT::Math::chisquared_pdf
   #      ROOT::Math::chisquared_quantile
   #      ROOT::Math::chisquared_quantile_c
   #      ROOT::Math::comp_ellint_1
   #      ROOT::Math::comp_ellint_2
   #      ROOT::Math::comp_ellint_3
   #      ROOT::Math::conf_hyperg
   #      ROOT::Math::conf_hypergU
   #      ROOT::Math::crystalball_cdf
   #      ROOT::Math::crystalball_cdf_c
   #      ROOT::Math::crystalball_function
   #      ROOT::Math::crystalball_integral
   #      ROOT::Math::crystalball_pdf
   #      ROOT::Math::cyl_bessel_i
   #      ROOT::Math::cyl_bessel_j
   #      ROOT::Math::cyl_bessel_k
   #      ROOT::Math::cyl_neumann
   #      ROOT::Math::ellint_1
   #      ROOT::Math::ellint_2
   #      ROOT::Math::ellint_3
   #      ROOT::Math::erf
   #      ROOT::Math::erfc
   #      ROOT::Math::etaMax_impl
   #      ROOT::Math::exp
   #      ROOT::Math::expint
   #      ROOT::Math::expint_n
   #      ROOT::Math::expm1
   #      ROOT::Math::exponential_cdf
   #      ROOT::Math::exponential_cdf_c
   #      ROOT::Math::exponential_pdf
   #      ROOT::Math::exponential_quantile
   #      ROOT::Math::exponential_quantile_c
   #      ROOT::Math::fdistribution_cdf
   #      ROOT::Math::fdistribution_cdf_c
   #      ROOT::Math::fdistribution_pdf
   #      ROOT::Math::fdistribution_quantile
   #      ROOT::Math::fdistribution_quantile_c
   #      ROOT::Math::floor
   #      ROOT::Math::gamma_cdf
   #      ROOT::Math::gamma_cdf_c
   #      ROOT::Math::gamma_pdf
   #      ROOT::Math::gamma_quantile
   #      ROOT::Math::gamma_quantile_c
   #      ROOT::Math::gaussian_cdf
   #      ROOT::Math::gaussian_cdf_c
   #      ROOT::Math::gaussian_pdf
   #      ROOT::Math::gaussian_quantile
   #      ROOT::Math::gaussian_quantile_c
   #      ROOT::Math::hyperg
   #      ROOT::Math::inc_beta
   #      ROOT::Math::inc_gamma
   #      ROOT::Math::inc_gamma_c
   #      ROOT::Math::laguerre
   #      ROOT::Math::lambert_W0
   #      ROOT::Math::lambert_Wm1
   #      ROOT::Math::landau_cdf
   #      ROOT::Math::landau_cdf_c
   #      ROOT::Math::landau_pdf
   #      ROOT::Math::landau_quantile
   #      ROOT::Math::landau_quantile_c
   #      ROOT::Math::landau_xm1
   #      ROOT::Math::landau_xm2
   #      ROOT::Math::legendre
   #      ROOT::Math::lgamma
   #      ROOT::Math::log
   #      ROOT::Math::log1p
   #      ROOT::Math::lognormal_cdf
   #      ROOT::Math::lognormal_cdf_c
   #      ROOT::Math::lognormal_pdf
   #      ROOT::Math::lognormal_quantile
   #      ROOT::Math::lognormal_quantile_c
   #      ROOT::Math::negative_binomial_cdf
   #      ROOT::Math::negative_binomial_cdf_c
   #      ROOT::Math::negative_binomial_pdf
   #      ROOT::Math::noncentral_chisquared_pdf
   #      ROOT::Math::normal_cdf
   #      ROOT::Math::normal_cdf_c
   #      ROOT::Math::normal_pdf
   #      ROOT::Math::normal_quantile
   #      ROOT::Math::normal_quantile_c
   #      ROOT::Math::poisson_cdf
   #      ROOT::Math::poisson_cdf_c
   #      ROOT::Math::poisson_pdf
   #      ROOT::Math::riemann_zeta
   #      ROOT::Math::sph_bessel
   #      ROOT::Math::sph_legendre
   #      ROOT::Math::sph_neumann
   #      ROOT::Math::tdistribution_cdf
   #      ROOT::Math::tdistribution_cdf_c
   #      ROOT::Math::tdistribution_pdf
   #      ROOT::Math::tdistribution_quantile
   #      ROOT::Math::tdistribution_quantile_c
   #      ROOT::Math::tgamma
   #      ROOT::Math::uniform_cdf
   #      ROOT::Math::uniform_cdf_c
   #      ROOT::Math::uniform_pdf
   #      ROOT::Math::uniform_quantile
   #      ROOT::Math::uniform_quantile_c
   #      ROOT::Math::vavilov_accurate_cdf
   #      ROOT::Math::vavilov_accurate_cdf_c
   #      ROOT::Math::vavilov_accurate_pdf
   #      ROOT::Math::vavilov_accurate_quantile
   #      ROOT::Math::vavilov_accurate_quantile_c
   #      ROOT::Math::vavilov_fast_cdf
   #      ROOT::Math::vavilov_fast_cdf_c
   #      ROOT::Math::vavilov_fast_pdf
   #      ROOT::Math::vavilov_fast_quantile
   #      ROOT::Math::vavilov_fast_quantile_c
   #      ROOT::Math::wigner_3j
   #      ROOT::Math::wigner_6j
   #      ROOT::Math::wigner_9j



   # Setting-up Canvas 
   global c1
   c1 =  TCanvas("c1","c1",800,650)
   
   c1.Divide(2,2)

   # Drawing on Canvas
   
   c1.cd(1) 
   f1a.SetLineWidth(1)
   f1a.Draw("surf1")

   c1.cd(2) 
   f2a.SetLineWidth(1)
   f2a.Draw("surf1")

   c1.cd(3) 
   f3a.SetLineWidth(1)
   f3a.Draw("surf1")

   c1.cd(4) 
   f4a.SetLineWidth(1)
   f4a.Draw("surf1")
   


if __name__ == "__main__":
   mathcoreStatFunc()
