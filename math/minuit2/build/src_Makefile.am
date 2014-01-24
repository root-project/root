## Process this file with automake to produce Makefile.in .
#
# Author: Paul_Kunz@slac.stanford.edu
#

# The following is set; otherwise it would follow GNU conventions.
AUTOMAKE_OPTIONS = foreign

# The convenience library to be built.
lib_LTLIBRARIES = libMinuit2.la

libMinuit2_la_SOURCES =				\
         AnalyticalGradientCalculator.cxx	\
         BasicMinimumError.cxx			\
         CombinedMinimumBuilder.cxx		\
         DavidonErrorUpdator.cxx		\
         FumiliBuilder.cxx                      \
         FumiliErrorUpdator.cxx                 \
         FumiliGradientCalculator.cxx           \
         FumiliMinimizer.cxx                    \
         FumiliStandardChi2FCN.cxx              \
	 FumiliStandardMaximumLikelihoodFCN.cxx \
         HessianGradientCalculator.cxx		\
         InitialGradientCalculator.cxx		\
         LaEigenValues.cxx			\
         LaInnerProduct.cxx			\
         LaInverse.cxx				\
         LaOuterProduct.cxx			\
         LaSumOfElements.cxx			\
         LaVtMVSimilarity.cxx			\
         Minuit2Minimizer.cxx			\
         MnApplication.cxx			\
         MnContours.cxx				\
         MnCovarianceSqueeze.cxx		\
         MnEigen.cxx				\
         MnFcn.cxx				\
         MnFunctionCross.cxx			\
         MnFumiliMinimize.cxx			\
         MnGlobalCorrelationCoeff.cxx		\
         MnHesse.cxx				\
         MnLineSearch.cxx			\
         MnMachinePrecision.cxx			\
         MnMinos.cxx				\
         MnParabolaFactory.cxx			\
         MnParameterScan.cxx			\
         MnPlot.cxx				\
         MnPosDef.cxx				\
         MnPrint.cxx				\
         MnScan.cxx				\
         MnSeedGenerator.cxx			\
         MnStrategy.cxx				\
         MnTiny.cxx				\
         MnUserFcn.cxx				\
         MnUserParameterState.cxx		\
         MnUserParameters.cxx			\
         MnUserTransformation.cxx		\
         ModularFunctionMinimizer.cxx		\
	 MPIProcess.cxx                         \
         NegativeG2LineSearch.cxx		\
         Numerical2PGradientCalculator.cxx	\
         ParametricFunction.cxx                 \
         ScanBuilder.cxx			\
         SimplexBuilder.cxx			\
         SimplexParameters.cxx			\
         SimplexSeedGenerator.cxx		\
         SinParameterTransformation.cxx		\
         SqrtLowParameterTransformation.cxx	\
         SqrtUpParameterTransformation.cxx	\
         VariableMetricBuilder.cxx		\
         VariableMetricEDMEstimator.cxx		\
         MinimizerOptions.cxx		\
         GenAlgoOptions.cxx		\
         mnbins.cxx				\
         mndasum.cxx				\
         mndaxpy.cxx				\
         mnddot.cxx				\
         mndscal.cxx				\
         mndspmv.cxx				\
         mndspr.cxx				\
         mnteigen.cxx				\
         mnlsame.cxx				\
	 mntplot.cxx				\
	 mnvert.cxx				\
         mnxerbla.cxx                        

INCLUDES =					\
	-I$(top_srcdir)/inc

# For platforms where automake dependency generation doesn't work.
depend: $(libMinuit2_la_SOURCES)
	$(top_srcdir)/config/makedepend $(DEFS) $(INCLUDES) $(CPPFLAGS) $?

# An extra clean target for the Sun CC compiler.
clean-local:
	-rm -rf SunWS_cache
