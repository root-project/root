// @(#)root/splot:$Id$
// Author: Muriel Pivk, Anna Kreshuk    10/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#include "TSPlot.h"
#include "TVirtualFitter.h"
#include "TH1.h"
#include "TTreePlayer.h"
#include "TTreeFormula.h"
#include "TTreeFormulaManager.h"
#include "TSelectorDraw.h"
#include "TBrowser.h"
#include "TClass.h"
#include "TMath.h"

extern void Yields(Int_t &, Double_t *, Double_t &f, Double_t *x, Int_t iflag);

ClassImp(TSPlot)

//____________________________________________________________________
//Begin_Html <!--
/* -->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<p>
<b><font size="+2">Overview</font></b>

</p><p>
A common method used in High Energy Physics to perform measurements is
the maximum Likelihood method, exploiting discriminating variables to
disentangle signal from background. The crucial point for such an
analysis to be reliable is to use an exhaustive list of sources of
events combined with an accurate description of all the Probability
Density Functions (PDF).
</p><p>To assess the validity of the fit, a convincing quality check
is to explore further the data sample by examining the distributions of
control variables. A control variable can be obtained for instance by
removing one of the discriminating variables before performing again
the maximum Likelihood fit: this removed variable is a control
variable. The expected distribution of this control variable, for
signal, is to be compared to the one extracted, for signal, from the
data sample. In order to be able to do so, one must be able to unfold
from the distribution of the whole data sample.
</p><p>The TSPlot method allows to reconstruct the distributions for
the control variable, independently for each of the various sources of
events, without making use of any <em>a priori</em> knowledge on <u>this</u>
variable. The aim is thus to use the knowledge available for the
discriminating variables to infer the behaviour of the individual
sources of events with respect to the control variable.
</p><p>
TSPlot is optimal if the control variable is uncorrelated with the discriminating variables.

</p><p>
A detail description of the formalism itself, called <!-- MATH
 $\hbox{$_s$}{\cal P}lot$
 -->
<img src="gif/sPlot_img5.png" alt="$\hbox{$_s$}{\cal P}lot$" align="middle" border="0" height="34" width="48">, is given in&nbsp;[<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/node1.html#bib:sNIM">1</a>].

</p><p>
<b><font size="+2">The method</font></b>

</p><p>
The <!-- MATH
 $\hbox{$_s$}{\cal P}lot$
 -->
<img src="gif/sPlot_img5.png" alt="$\hbox{$_s$}{\cal P}lot$" align="middle" border="0" height="34" width="48"> technique is developped in the above context of a maximum Likelihood method making use of discriminating variables.

</p><p>One considers a data sample in which are merged several species
of events. These species represent various signal components and
background components which all together account for the data sample.
The different terms of the log-Likelihood are:
</p><ul>
<li><img src="gif/sPlot_img6.png" alt="$N$" align="bottom" border="0" height="17" width="22">: the total number of events in the data sample,
</li>
<li><!-- MATH
 ${\rm N}_{\rm s}$
 -->
<img src="gif/sPlot_img7.png" alt="${\rm N}_{\rm s}$" align="middle" border="0" height="34" width="25">: the number of species of events populating the data sample,
</li>
<li><img src="gif/sPlot_img8.png" alt="$N_i$" align="middle" border="0" height="34" width="25">: the number of events expected on the average for the <img src="gif/sPlot_img9.png" alt="$i^{\rm th}$" align="bottom" border="0" height="20" width="25"> species,
</li>
<li><!-- MATH
 ${\rm f}_i(y_e)$
 -->
<img src="gif/sPlot_img10.png" alt="${\rm f}_i(y_e)$" align="middle" border="0" height="37" width="47">: the value of the PDFs of the discriminating variables <img src="gif/sPlot_img11.png" alt="$y$" align="middle" border="0" height="33" width="15"> for the <img src="gif/sPlot_img12.png" alt="$i^{th}$" align="bottom" border="0" height="20" width="25"> species and for event <img src="gif/sPlot_img13.png" alt="$e$" align="bottom" border="0" height="17" width="13">,
</li>
<li><img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">: the set of control variables which, by definition, do not appear in the expression of the Likelihood function <img src="gif/sPlot_img15.png" alt="${\cal L}$" align="bottom" border="0" height="18" width="18">.
</li>
</ul>
The extended log-Likelihood reads:
<br>
<div align="right">

<!-- MATH
 \begin{equation}
{\cal L}=\sum_{e=1}^{N}\ln \Big\{ \sum_{i=1}^{{\rm N}_{\rm s}}N_i{\rm f}_i(y_e) \Big\} -\sum_{i=1}^{{\rm N}_{\rm s}}N_i ~.
\end{equation}
 -->
<table align="center" width="100%">
<tbody><tr valign="middle"><td align="center" nowrap="nowrap"><a name="eq:eLik"></a><img src="gif/sPlot_img16.png" alt="\begin{displaymath}
{\cal L}=\sum_{e=1}^{N}\ln \Big\{ \sum_{i=1}^{{\rm N}_{\rm s}}N_i{\rm f}_i(y_e) \Big\} -\sum_{i=1}^{{\rm N}_{\rm s}}N_i ~.
\end{displaymath}" border="0" height="59" width="276"></td>
<td align="right" width="10">
(1)</td></tr>
</tbody></table>
<br clear="all"></div><p></p>
From this expression, after maximization of <img src="gif/sPlot_img15.png" alt="${\cal L}$" align="bottom" border="0" height="18" width="18"> with respect to the <img src="gif/sPlot_img8.png" alt="$N_i$" align="middle" border="0" height="34" width="25"> parameters, a weight can be computed for every event and each species, in order to obtain later the true distribution <!-- MATH
 ${\hbox{\bf {M}}}_i(x)$
 -->
<img src="gif/sPlot_img17.png" alt="${\hbox{\bf {M}}}_i(x)$" align="middle" border="0" height="37" width="56"> of variable <img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">. If <img src="gif/sPlot_img18.png" alt="${\rm n}$" align="bottom" border="0" height="17" width="15"> is one of the <!-- MATH
 ${\rm N}_{\rm s}$
 -->
<img src="gif/sPlot_img7.png" alt="${\rm N}_{\rm s}$" align="middle" border="0" height="34" width="25"> species present in the data sample, the weight for this species is defined by:
<br>
<div align="right">

<!-- MATH
 \begin{equation}
\begin{Large}\fbox{$
{_s{\cal P}}_{\rm n}(y_e)={\sum_{j=1}^{{\rm N}_{\rm s}} \hbox{\bf V}_{{\rm n}j}{\rm f}_j(y_e)\over\sum_{k=1}^{{\rm N}_{\rm s}}N_k{\rm f}_k(y_e) } $}\end{Large} ~,
\end{equation}
 -->
<table align="center" width="100%">
<tbody><tr valign="middle"><td align="center" nowrap="nowrap"><a name="eq:weightxnotiny"></a><img src="gif/sPlot_img19.png" alt="\begin{displaymath}
\begin{Large}
\fbox{$
{_s{\cal P}}_{\rm n}(y_e)={\sum_{j=1}^...
...um_{k=1}^{{\rm N}_{\rm s}}N_k{\rm f}_k(y_e) } $}\end{Large} ~,
\end{displaymath}" border="0" height="76" width="279"></td>
<td align="right" width="10">
(2)</td></tr>
</tbody></table>
<br clear="all"></div><p></p>
where <!-- MATH
 $\hbox{\bf V}_{{\rm n}j}$
 -->
<img src="gif/sPlot_img20.png" alt="$\hbox{\bf V}_{{\rm n}j}$" align="middle" border="0" height="34" width="35">
is the covariance matrix resulting from the Likelihood maximization.
This matrix can be used directly from the fit, but this is numerically
less accurate than the direct computation:
<br>
<div align="right">

<!-- MATH
 \begin{equation}
\hbox{\bf V}^{-1}_{{\rm n}j}~=~
{\partial^2(-{\cal L})\over\partial N_{\rm n}\partial N_j}~=~
\sum_{e=1}^N {{\rm f}_{\rm n}(y_e){\rm f}_j(y_e)\over(\sum_{k=1}^{{\rm N}_{\rm s}}N_k{\rm f}_k(y_e))^2} ~.
\end{equation}
 -->
<table align="center" width="100%">
<tbody><tr valign="middle"><td align="center" nowrap="nowrap"><a name="eq:VarianceMatrixDirect"></a><img src="gif/sPlot_img21.png" alt="\begin{displaymath}
\hbox{\bf V}^{-1}_{{\rm n}j}~=~
{\partial^2(-{\cal L})\over\...
...y_e)\over(\sum_{k=1}^{{\rm N}_{\rm s}}N_k{\rm f}_k(y_e))^2} ~.
\end{displaymath}" border="0" height="58" width="360"></td>
<td align="right" width="10">
(3)</td></tr>
</tbody></table>
<br clear="all"></div><p></p>
The distribution of the control variable&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15"> obtained by histogramming the weighted events reproduces, on average, the true distribution <!-- MATH
 ${\hbox{\bf {M}}}_{\rm n}(x)$
 -->
<img src="gif/sPlot_img22.png" alt="${\hbox{\bf {M}}}_{\rm n}(x)$" align="middle" border="0" height="37" width="59">.

<p>
The class TSPlot allows to reconstruct the true distribution <!-- MATH
 ${\hbox{\bf {M}}}_{\rm n}(x)$
 -->
<img src="gif/sPlot_img22.png" alt="${\hbox{\bf {M}}}_{\rm n}(x)$" align="middle" border="0" height="37" width="59"> of a control variable&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15"> for each of the <!-- MATH
 ${\rm N}_{\rm s}$
 -->
<img src="gif/sPlot_img7.png" alt="${\rm N}_{\rm s}$" align="middle" border="0" height="34" width="25"> species from the sole knowledge of the PDFs of the discriminating variables <img src="gif/sPlot_img23.png" alt="${\rm f}_i(y)$" align="middle" border="0" height="37" width="40">. The plots obtained thanks to the TSPlot class are called <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57">.

</p><p>
<b><font size="+2">Some properties and checks</font></b>

</p><p>
Beside reproducing the true distribution, <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57"> bear remarkable properties:

</p><ul>
<li>
Each <img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">-distribution is properly normalized:
<br>
<div align="right">

<!-- MATH
 \begin{equation}
\sum_{e=1}^{N} {_s{\cal P}}_{\rm n}(y_e)~=~N_{\rm n}~.
\end{equation}
 -->
<table align="center" width="100%">
<tbody><tr valign="middle"><td align="center" nowrap="nowrap"><a name="eq:NormalizationOK"></a><img src="gif/sPlot_img24.png" alt="\begin{displaymath}
\sum_{e=1}^{N} {_s{\cal P}}_{\rm n}(y_e)~=~N_{\rm n}~.
\end{displaymath}" border="0" height="58" width="158"></td>
<td align="right" width="10">
(4)</td></tr>
</tbody></table>
<br clear="all"></div><p></p>
</li>
<li>
For any event:
<br>
<div align="right">

<!-- MATH
 \begin{equation}
\sum_{l=1}^{{\rm N}_{\rm s}} {_s{\cal P}}_l(y_e) ~=~1 ~.
\end{equation}
 -->
<table align="center" width="100%">
<tbody><tr valign="middle"><td align="center" nowrap="nowrap"><a name="eq:numberconservation"></a><img src="gif/sPlot_img25.png" alt="\begin{displaymath}
\sum_{l=1}^{{\rm N}_{\rm s}} {_s{\cal P}}_l(y_e) ~=~1 ~.
\end{displaymath}" border="0" height="59" width="140"></td>
<td align="right" width="10">
(5)</td></tr>
</tbody></table>
<br clear="all"></div><p></p>
That is to say that, summing up the <!-- MATH
 ${\rm N}_{\rm s}$
 -->
<img src="gif/sPlot_img7.png" alt="${\rm N}_{\rm s}$" align="middle" border="0" height="34" width="25"> <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57">, one recovers the data sample distribution in&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">, and summing up the number of events entering in a <!-- MATH
 $\hbox{$_s$}{\cal P}lot$
 -->
<img src="gif/sPlot_img5.png" alt="$\hbox{$_s$}{\cal P}lot$" align="middle" border="0" height="34" width="48"> for a given species, one recovers the yield of the species, as provided by the fit. The property&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#eq:NormalizationOK">4</a> is implemented in the TSPlot class as a check.
</li>
<li>the sum of the statistical uncertainties per bin
<br>
<div align="right">

<!-- MATH
 \begin{equation}
\sigma[N_{\rm n}\  _s\tilde{\rm M}_{\rm n}(x) {\delta x}]~=~\sqrt{\sum_{e \subset {\delta x}} ({_s{\cal P}}_{\rm n})^2} ~.
\end{equation}
 -->
<table align="center" width="100%">
<tbody><tr valign="middle"><td align="center" nowrap="nowrap"><a name="eq:ErrorPerBin"></a><img src="gif/sPlot_img26.png" alt="\begin{displaymath}
\sigma[N_{\rm n}\ _s\tilde{\rm M}_{\rm n}(x) {\delta x}]~=~\sqrt{\sum_{e \subset {\delta x}} ({_s{\cal P}}_{\rm n})^2} ~.
\end{displaymath}" border="0" height="55" width="276"></td>
<td align="right" width="10">
(6)</td></tr>
</tbody></table>
<br clear="all"></div><p></p>
reproduces the statistical uncertainty on the yield <img src="gif/sPlot_img27.png" alt="$N_{\rm n}$" align="middle" border="0" height="34" width="28">, as provided by the fit: <!-- MATH
 $\sigma[N_{\rm n}]\equiv\sqrt{\hbox{\bf V}_{{\rm n}{\rm n}}}$
 -->
<img src="gif/sPlot_img28.png" alt="$\sigma[N_{\rm n}]\equiv\sqrt{\hbox{\bf V}_{{\rm n}{\rm n}}}$" align="middle" border="0" height="40" width="123">.
Because of that and since the determination of the yields is optimal
when obtained using a Likelihood fit, one can conclude that the<!-- MATH
 $\hbox{$_s$}{\cal P}lot$
 -->
 <img src="gif/sPlot_img5.png" alt="$\hbox{$_s$}{\cal P}lot$" align="middle" border="0" height="34" width="48"> technique is itself an optimal method to reconstruct distributions of control variables.
</li>
</ul>

<p>
<b><font size="+2">Different steps followed by TSPlot</font></b>

</p><p>

</p><ol>
<li>A maximum Likelihood fit is performed to obtain the yields <img src="gif/sPlot_img8.png" alt="$N_i$" align="middle" border="0" height="34" width="25"> of the various species.
The fit relies on discriminating variables&nbsp;<img src="gif/sPlot_img11.png" alt="$y$" align="middle" border="0" height="33" width="15"> uncorrelated with a control variable&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">:
the later is therefore totally absent from the fit.
</li>
<li>The weights <img src="gif/sPlot_img29.png" alt="${_s{\cal P}}$" align="middle" border="0" height="34" width="27"> are calculated using Eq.&nbsp;(<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#eq:weightxnotiny">2</a>) where the covariance matrix is taken from Minuit.
</li>
<li>Histograms of&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15"> are filled by weighting the events with <img src="gif/sPlot_img29.png" alt="${_s{\cal P}}$" align="middle" border="0" height="34" width="27">.
</li>
<li>Error bars per bin are given by Eq.&nbsp;(<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#eq:ErrorPerBin">6</a>).
</li>
</ol>
The <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57"> reproduce the true distributions of the species in the control variable&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">, within the above defined statistical uncertainties.

<p>
<b><font size="+2">Illustrations</font></b>

</p><p>
To illustrate the technique, one considers an example derived from the analysis where <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57">
have been first used (charmless B decays). One is dealing with a data
sample in which two species are present: the first is termed signal and
the second background. A maximum Likelihood fit is performed to obtain
the two yields <img src="gif/sPlot_img30.png" alt="$N_1$" align="middle" border="0" height="34" width="27"> and <img src="gif/sPlot_img31.png" alt="$N_2$" align="middle" border="0" height="34" width="27">. The fit relies on two discriminating variables collectively denoted&nbsp;<img src="gif/sPlot_img11.png" alt="$y$" align="middle" border="0" height="33" width="15"> which are chosen within three possible variables denoted <img src="gif/sPlot_img1.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39">, <img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35"> and <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20">.
The variable which is not incorporated in&nbsp;<img src="gif/sPlot_img11.png" alt="$y$" align="middle" border="0" height="33" width="15"> is used as the control variable&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15">. The six distributions of the three variables are assumed to be the ones depicted in Fig.&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#fig:pdfs">1</a>.

</p><p>

</p><div align="center"><a name="fig:pdfs"></a><a name="106"></a>
<table>
<caption align="bottom"><strong>Figure 1:</strong>
Distributions of the three discriminating variables available to perform the Likelihood fit:
<img src="gif/sPlot_img32.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39">, <img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35">, <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20">.
Among the three variables, two are used to perform the fit while one is
kept out of the fit to serve the purpose of a control variable. The
three distributions on the top (resp. bottom) of the figure correspond
to the signal (resp. background). The unit of the vertical axis is
chosen such that it indicates the number of entries per bin, if one
slices the histograms in 25 bins.</caption>
<tbody><tr><td><img src="gif/sPlot_img33.png" alt="\begin{figure}\begin{center}
\mbox{{\psfig{file=pdfmesNIM.eps,width=0.33\linewi...
...th}}
{\psfig{file=pdffiNIM.eps,width=0.33\linewidth}}}
\end{center}\end{figure}" border="0" height="162" width="544"></td></tr>
</tbody></table>
</div>

<p>
A data sample being built through a Monte Carlo simulation based on the distributions shown in Fig.&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#fig:pdfs">1</a>, one obtains the three distributions of Fig.&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#fig:pdfstot">2</a>. Whereas the distribution of&nbsp;<img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35"> clearly indicates the presence of the signal, the distribution of <img src="gif/sPlot_img1.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39"> and <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20"> are less obviously populated by signal.

</p><p>

</p><div align="center"><a name="fig:pdfstot"></a><a name="169"></a>
<table>
<caption align="bottom"><strong>Figure 2:</strong>
Distributions of the three discriminating variables for signal plus
background. The three distributions are the ones obtained from a data
sample obtained through a Monte Carlo simulation based on the
distributions shown in Fig.&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#fig:pdfs">1</a>.  The data sample consists of 500 signal events and 5000 background events.</caption>
<tbody><tr><td><img src="gif/sPlot_img34.png" alt="\begin{figure}\begin{center}
\mbox{{\psfig{file=genmesTOTNIM.eps,width=0.33\lin...
...}
{\psfig{file=genfiTOTNIM.eps,width=0.33\linewidth}}}
\end{center}\end{figure}" border="0" height="160" width="545"></td></tr>
</tbody></table>
</div>

<p>
Chosing <img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35"> and <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20"> as discriminating variables to determine <img src="gif/sPlot_img30.png" alt="$N_1$" align="middle" border="0" height="34" width="27"> and <img src="gif/sPlot_img31.png" alt="$N_2$" align="middle" border="0" height="34" width="27"> through a maximum Likelihood fit, one builds, for the control variable <img src="gif/sPlot_img1.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39"> which is unknown to the fit, the two <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57"> for signal and background shown in Fig.&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#fig:messPlots">3</a>. One observes that the <!-- MATH
 $\hbox{$_s$}{\cal P}lot$
 -->
<img src="gif/sPlot_img5.png" alt="$\hbox{$_s$}{\cal P}lot$" align="middle" border="0" height="34" width="48">
for signal reproduces correctly the PDF even where the latter vanishes,
although the error bars remain sizeable. This results from the almost
complete cancellation between positive and negative weights: the sum of
weights is close to zero while the sum of weights squared is not. The
occurence of negative weights occurs through the appearance of the
covariance matrix, and its negative components, in the definition of
Eq.&nbsp;(<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#eq:weightxnotiny">2</a>).

</p><p>
A word of caution is in order with respect to the error bars. Whereas
their sum in quadrature is identical to the statistical uncertainties
of the yields determined by the fit, and if, in addition, they are
asymptotically correct, the error bars should be handled with care for
low statistics and/or for too fine binning. This is because the error
bars do not incorporate two known properties of the PDFs: PDFs are
positive definite and can be non-zero in a given x-bin, even if in the
particular data sample at hand, no event is observed in this bin. The
latter limitation is not specific to<!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
 <img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57">,
rather it is always present when one is willing to infer the PDF at the
origin of an histogram, when, for some bins, the number of entries does
not guaranty the applicability of the Gaussian regime. In such
situations, a satisfactory practice is to attach allowed ranges to the
histogram to indicate the upper and lower limits of the PDF value which
are consistent with the actual observation, at a given confidence
level.
</p><p>

</p><div align="center"><a name="fig:messPlots"></a><a name="127"></a>
<table>
<caption align="bottom"><strong>Figure 3:</strong>
The <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57"> (signal on the left, background on the right) obtained for <img src="gif/sPlot_img32.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39"> are represented as dots with error bars. They are obtained from a fit using only information from <img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35"> and <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20">.</caption>
<tbody><tr><td><img src="gif/sPlot_img35.png" alt="\begin{figure}\begin{center}
\mbox{\psfig{file=mass-sig-sPlot.eps,width=0.48\li...
... \psfig{file=mass-bkg-sPlot.eps,width=0.48\linewidth}}
\end{center}\end{figure}" border="0" height="181" width="539"></td></tr>
</tbody></table>
</div>

<p>
Chosing <img src="gif/sPlot_img1.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39"> and <img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35"> as discriminating variables to determine <img src="gif/sPlot_img30.png" alt="$N_1$" align="middle" border="0" height="34" width="27"> and <img src="gif/sPlot_img31.png" alt="$N_2$" align="middle" border="0" height="34" width="27"> through a maximum Likelihood fit, one builds, for the control variable <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20"> which is unknown to the fit, the two <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57"> for signal and background shown in Fig.&nbsp;<a href="http://www.slac.stanford.edu/%7Epivk/sPlot/sPlot_ROOT/sPlot_ROOT.html#fig:FisPlots">4</a>. In the <!-- MATH
 $\hbox{$_s$}{\cal P}lot$
 -->
<img src="gif/sPlot_img5.png" alt="$\hbox{$_s$}{\cal P}lot$" align="middle" border="0" height="34" width="48"> for signal one observes that error bars are the largest in the&nbsp;<img src="gif/sPlot_img14.png" alt="$x$" align="bottom" border="0" height="17" width="15"> regions where the background is the largest.

</p><p>

</p><div align="center"><a name="fig:FisPlots"></a><a name="136"></a>
<table>
<caption align="bottom"><strong>Figure 4:</strong>
The <!-- MATH
 $\hbox{$_s$}{\cal P}lots$
 -->
<img src="gif/sPlot_img4.png" alt="$\hbox {$_s$}{\cal P}lots$" align="middle" border="0" height="34" width="57"> (signal on the left, background on the right) obtained for <img src="gif/sPlot_img3.png" alt="${\cal F}$" align="bottom" border="0" height="18" width="20"> are represented as dots with error bars. They are obtained from a fit using only information from <img src="gif/sPlot_img32.png" alt="${m_{\rm ES}}$" align="middle" border="0" height="33" width="39"> and <img src="gif/sPlot_img2.png" alt="$\Delta E$" align="bottom" border="0" height="17" width="35">.</caption>
<tbody><tr><td><img src="gif/sPlot_img36.png" alt="\begin{figure}\begin{center}
\mbox{\psfig{file=fisher-sig-sPlot.eps,width=0.48\...
...psfig{file=fisher-bkg-sPlot.eps,width=0.48\linewidth}}
\end{center}\end{figure}" border="0" height="180" width="539"></td></tr>
</tbody></table>
</div>

<p>
The results above can be obtained by running the tutorial TestSPlot.C
</p>
<!--*/
//-->End_Html


//____________________________________________________________________
TSPlot::TSPlot() :
 fTree(0),
 fTreename(0),
 fVarexp(0),
 fSelection(0)
{
   // default constructor (used by I/O only)
   fNx = 0;
   fNy=0;
   fNevents = 0;
   fNSpecies=0;
   fNumbersOfEvents=0;
}

//____________________________________________________________________
TSPlot::TSPlot(Int_t nx, Int_t ny, Int_t ne, Int_t ns, TTree *tree) :
 fTreename(0),
 fVarexp(0),
 fSelection(0)

{
   //normal TSPlot constructor
   // nx :  number of control variables
   // ny :  number of discriminating variables
   // ne :  total number of events
   // ns :  number of species
   // tree: input data

   fNx = nx;
   fNy=ny;
   fNevents = ne;
   fNSpecies=ns;

   fXvar.ResizeTo(fNevents, fNx);
   fYvar.ResizeTo(fNevents, fNy);
   fYpdf.ResizeTo(fNevents, fNSpecies*fNy);
   fSWeights.ResizeTo(fNevents, fNSpecies*(fNy+1));
   fTree = tree;
   fNumbersOfEvents = 0;
}

//____________________________________________________________________
TSPlot::~TSPlot()
{
   // destructor

   if (fNumbersOfEvents)
      delete [] fNumbersOfEvents;
   if (!fXvarHists.IsEmpty())
      fXvarHists.Delete();
   if (!fYvarHists.IsEmpty())
      fYvarHists.Delete();
   if (!fYpdfHists.IsEmpty())
      fYpdfHists.Delete();
}

//____________________________________________________________________
void TSPlot::Browse(TBrowser *b)
{
   //To browse the histograms

   if (!fSWeightsHists.IsEmpty()) {
      TIter next(&fSWeightsHists);
      TH1D* h = 0;
      while ((h = (TH1D*)next()))
         b->Add(h,h->GetName());
   }

   if (!fYpdfHists.IsEmpty()) {
      TIter next(&fYpdfHists);
      TH1D* h = 0;
      while ((h = (TH1D*)next()))
         b->Add(h,h->GetName());
   }
   if (!fYvarHists.IsEmpty()) {
      TIter next(&fYvarHists);
      TH1D* h = 0;
      while ((h = (TH1D*)next()))
         b->Add(h,h->GetName());
   }
   if (!fXvarHists.IsEmpty()) {
      TIter next(&fXvarHists);
      TH1D* h = 0;
      while ((h = (TH1D*)next()))
         b->Add(h,h->GetName());
   }
   b->Add(&fSWeights, "sWeights");
}


//____________________________________________________________________
void TSPlot::SetInitialNumbersOfSpecies(Int_t *numbers)
{
//Set the initial number of events of each species - used
//as initial estimates in minuit

   if (!fNumbersOfEvents)
      fNumbersOfEvents = new Double_t[fNSpecies];
   for (Int_t i=0; i<fNSpecies; i++)
      fNumbersOfEvents[i]=numbers[i];
}

//____________________________________________________________________
void TSPlot::MakeSPlot(Option_t *option)
{
//Calculates the sWeights
//The option controls the print level
//"Q" - no print out
//"V" - prints the estimated #of events in species - default
//"VV" - as "V" + the minuit printing + sums of weights for control


   if (!fNumbersOfEvents){
      Error("MakeSPlot","Initial numbers of events in species have not been set");
      return;
   }
   Int_t i, j, ispecies;

   TString opt = option;
   opt.ToUpper();
   opt.ReplaceAll("VV", "W");

   //make sure that global fitter is minuit
   char s[]="TFitter";
   if (TVirtualFitter::GetFitter()){
      Int_t strdiff=strcmp(TVirtualFitter::GetFitter()->IsA()->GetName(), s);
      if (strdiff!=0)
         delete TVirtualFitter::GetFitter();
   }


   TVirtualFitter *minuit = TVirtualFitter::Fitter(0, 2);
   fPdfTot.ResizeTo(fNevents, fNSpecies);

   //now let's do it, excluding different yvars
   //for iplot = -1 none is excluded
   for (Int_t iplot=-1; iplot<fNy; iplot++){
      for (i=0; i<fNevents; i++){
         for (ispecies=0; ispecies<fNSpecies; ispecies++){
            fPdfTot(i, ispecies)=1;
            for (j=0; j<fNy; j++){
               if (j!=iplot)
                  fPdfTot(i, ispecies)*=fYpdf(i, ispecies*fNy+j);
            }
         }
      }
      minuit->Clear();
      minuit->SetFCN(Yields);
      Double_t arglist[10];
      //set the print level
      if (opt.Contains("Q")||opt.Contains("V")){
         arglist[0]=-1;
      }
      if (opt.Contains("W"))
         arglist[0]=0;
      minuit->ExecuteCommand("SET PRINT", arglist, 1);

      minuit->SetObjectFit(&fPdfTot); //a tricky way to get fPdfTot matrix to fcn
      for (ispecies=0; ispecies<fNSpecies; ispecies++)
         minuit->SetParameter(ispecies, "", fNumbersOfEvents[ispecies], 1, 0, 0);

      minuit->ExecuteCommand("MIGRAD", arglist, 0);
      for (ispecies=0; ispecies<fNSpecies; ispecies++){
         fNumbersOfEvents[ispecies]=minuit->GetParameter(ispecies);
         if (!opt.Contains("Q"))
            printf("estimated #of events in species %d = %f\n", ispecies, fNumbersOfEvents[ispecies]);
      }
      if (!opt.Contains("Q"))
         printf("\n");
      Double_t *covmat = minuit->GetCovarianceMatrix();
      SPlots(covmat, iplot);

      if (opt.Contains("W")){
         Double_t *sumweight = new Double_t[fNSpecies];
         for (i=0; i<fNSpecies; i++){
            sumweight[i]=0;
            for (j=0; j<fNevents; j++)
               sumweight[i]+=fSWeights(j, (iplot+1)*fNSpecies + i);
            printf("checking sum of weights[%d]=%f\n", i, sumweight[i]);
         }
         printf("\n");
         delete [] sumweight;
      }
   }
}

//____________________________________________________________________
void TSPlot::SPlots(Double_t *covmat, Int_t i_excl)
{
//Computes the sWeights from the covariance matrix

   Int_t i, ispecies, k;
   Double_t numerator, denominator;
   for (i=0; i<fNevents; i++){
      denominator=0;
      for (ispecies=0; ispecies<fNSpecies; ispecies++)
         denominator+=fNumbersOfEvents[ispecies]*fPdfTot(i, ispecies);
      for (ispecies=0; ispecies<fNSpecies; ispecies++){
         numerator=0;
         for (k=0; k<fNSpecies; k++)
            numerator+=covmat[ispecies*fNSpecies+k]*fPdfTot(i, k);
         fSWeights(i, (i_excl+1)*fNSpecies + ispecies)=numerator/denominator;
      }
   }

}

//____________________________________________________________________
void TSPlot::GetSWeights(TMatrixD &weights)
{
//Returns the matrix of sweights

   if (weights.GetNcols()!=fNSpecies*(fNy+1) || weights.GetNrows()!=fNevents)
      weights.ResizeTo(fNevents, fNSpecies*(fNy+1));
   weights = fSWeights;
}

//____________________________________________________________________
void TSPlot::GetSWeights(Double_t *weights)
{
//Returns the matrix of sweights. It is assumed that the array passed in the argurment is big enough

   for (Int_t i=0; i<fNevents; i++){
      for (Int_t j=0; j<fNSpecies; j++){
         weights[i*fNSpecies+j]=fSWeights(i, j);
      }
   }
}

//____________________________________________________________________
void TSPlot::FillXvarHists(Int_t nbins)
{
//Fills the histograms of x variables (not weighted) with nbins

   Int_t i, j;

   if (!fXvarHists.IsEmpty()){
      if (((TH1D*)fXvarHists.First())->GetNbinsX()!=nbins)
         fXvarHists.Delete();
      else
         return;
   }

   //make the histograms
   char name[10];
   for (i=0; i<fNx; i++){
      snprintf(name,10, "x%d", i);
      TH1D *h = new TH1D(name, name, nbins, fMinmax(0, i), fMinmax(1, i));
      for (j=0; j<fNevents; j++)
         h->Fill(fXvar(j, i));
      fXvarHists.Add(h);
   }

}

//____________________________________________________________________
TObjArray* TSPlot::GetXvarHists()
{
//Returns the array of histograms of x variables (not weighted)
//If histograms have not already
//been filled, they are filled with default binning 100.

   Int_t nbins = 100;
   if (fXvarHists.IsEmpty())
      FillXvarHists(nbins);
   else if (((TH1D*)fXvarHists.First())->GetNbinsX()!=nbins)
      FillXvarHists(nbins);
   return &fXvarHists;
}

//____________________________________________________________________
TH1D *TSPlot::GetXvarHist(Int_t ixvar)
{
//Returns the histogram of variable #ixvar
//If histograms have not already
//been filled, they are filled with default binning 100.

   Int_t nbins = 100;
   if (fXvarHists.IsEmpty())
      FillXvarHists(nbins);
   else if (((TH1D*)fXvarHists.First())->GetNbinsX()!=nbins)
      FillXvarHists(nbins);

   return (TH1D*)fXvarHists.UncheckedAt(ixvar);
}

//____________________________________________________________________
void TSPlot::FillYvarHists(Int_t nbins)
{
//Fill the histograms of y variables

   Int_t i, j;

   if (!fYvarHists.IsEmpty()){
      if (((TH1D*)fYvarHists.First())->GetNbinsX()!=nbins)
         fYvarHists.Delete();
      else
         return;
   }

   //make the histograms
   char name[10];
   for (i=0; i<fNy; i++){
      snprintf(name,10, "y%d", i);
      TH1D *h=new TH1D(name, name, nbins, fMinmax(0, fNx+i), fMinmax(1, fNx+i));
      for (j=0; j<fNevents; j++)
         h->Fill(fYvar(j, i));
      fYvarHists.Add(h);
   }
}

//____________________________________________________________________
TObjArray* TSPlot::GetYvarHists()
{
//Returns the array of histograms of y variables. If histograms have not already
//been filled, they are filled with default binning 100.

   Int_t nbins = 100;
   if (fYvarHists.IsEmpty())
      FillYvarHists(nbins);
   else if (((TH1D*)fYvarHists.First())->GetNbinsX()!=nbins)
      FillYvarHists(nbins);
   return &fYvarHists;
}

//____________________________________________________________________
TH1D *TSPlot::GetYvarHist(Int_t iyvar)
{
//Returns the histogram of variable iyvar.If histograms have not already
//been filled, they are filled with default binning 100.

   Int_t nbins = 100;
   if (fYvarHists.IsEmpty())
      FillYvarHists(nbins);
   else if (((TH1D*)fYvarHists.First())->GetNbinsX()!=nbins)
      FillYvarHists(nbins);
   return (TH1D*)fYvarHists.UncheckedAt(iyvar);
}

//____________________________________________________________________
void TSPlot::FillYpdfHists(Int_t nbins)
{
//Fills the histograms of pdf-s of y variables with binning nbins

   Int_t i, j, ispecies;

   if (!fYpdfHists.IsEmpty()){
      if (((TH1D*)fYpdfHists.First())->GetNbinsX()!=nbins)
         fYpdfHists.Delete();
      else
         return;
   }

   char name[30];
   for (ispecies=0; ispecies<fNSpecies; ispecies++){
      for (i=0; i<fNy; i++){
         snprintf(name,30, "pdf_species%d_y%d", ispecies, i);
         //TH1D *h = new TH1D(name, name, nbins, ypdfmin[ispecies*fNy+i], ypdfmax[ispecies*fNy+i]);
         TH1D *h = new TH1D(name, name, nbins, fMinmax(0, fNx+fNy+ispecies*fNy+i), fMinmax(1, fNx+fNy+ispecies*fNy+i));
         for (j=0; j<fNevents; j++)
            h->Fill(fYpdf(j, ispecies*fNy+i));
         fYpdfHists.Add(h);
      }
   }
}

//____________________________________________________________________
TObjArray* TSPlot::GetYpdfHists()
{
//Returns the array of histograms of pdf's of y variables with binning nbins
//If histograms have not already
//been filled, they are filled with default binning 100.

   Int_t nbins = 100;
   if (fYpdfHists.IsEmpty())
      FillYpdfHists(nbins);

   return &fYpdfHists;
}

//____________________________________________________________________
TH1D *TSPlot::GetYpdfHist(Int_t iyvar, Int_t ispecies)
{
//Returns the histogram of the pdf of variable #iyvar for species #ispecies, binning nbins
//If histograms have not already
//been filled, they are filled with default binning 100.

   Int_t nbins = 100;
   if (fYpdfHists.IsEmpty())
      FillYpdfHists(nbins);

   return (TH1D*)fYpdfHists.UncheckedAt(fNy*ispecies+iyvar);
}

//____________________________________________________________________
void TSPlot::FillSWeightsHists(Int_t nbins)
{
   //The order of histograms in the array:
   //x0_species0, x0_species1,..., x1_species0, x1_species1,..., y0_species0, y0_species1,...
   //If the histograms have already been filled with a different binning, they are refilled
   //and all histograms are deleted

   if (fSWeights.GetNoElements()==0){
      Error("GetSWeightsHists", "SWeights were not computed");
      return;
   }

   if (!fSWeightsHists.IsEmpty()){
      if (((TH1D*)fSWeightsHists.First())->GetNbinsX()!=nbins)
         fSWeightsHists.Delete();
      else
         return;
   }

   char name[30];

   //Fill histograms of x-variables weighted with sWeights
   for (Int_t ivar=0; ivar<fNx; ivar++){
      for (Int_t ispecies=0; ispecies<fNSpecies; ispecies++){
            snprintf(name,30, "x%d_species%d", ivar, ispecies);
            TH1D *h = new TH1D(name, name, nbins, fMinmax(0, ivar), fMinmax(1, ivar));
            h->Sumw2();
            for (Int_t ievent=0; ievent<fNevents; ievent++)
               h->Fill(fXvar(ievent, ivar), fSWeights(ievent, ispecies));
            fSWeightsHists.AddLast(h);
         }
   }

   //Fill histograms of y-variables (exluded from the fit), weighted with sWeights
   for (Int_t iexcl=0; iexcl<fNy; iexcl++){
      for(Int_t ispecies=0; ispecies<fNSpecies; ispecies++){
            snprintf(name,30, "y%d_species%d", iexcl, ispecies);
            TH1D *h = new TH1D(name, name, nbins, fMinmax(0, fNx+iexcl), fMinmax(1, fNx+iexcl));
            h->Sumw2();
            for (Int_t ievent=0; ievent<fNevents; ievent++)
               h->Fill(fYvar(ievent, iexcl), fSWeights(ievent, fNSpecies*(iexcl+1)+ispecies));
            fSWeightsHists.AddLast(h);
      }
   }
}

//____________________________________________________________________
TObjArray *TSPlot::GetSWeightsHists()
{
   //Returns an array of all histograms of variables, weighted with sWeights
   //If histograms have not been already filled, they are filled with default binning 50
   //The order of histograms in the array:
   //x0_species0, x0_species1,..., x1_species0, x1_species1,..., y0_species0, y0_species1,...

   Int_t nbins = 50; //default binning
   if (fSWeightsHists.IsEmpty())
      FillSWeightsHists(nbins);

   return &fSWeightsHists;
}

//____________________________________________________________________
void TSPlot::RefillHist(Int_t type, Int_t nvar, Int_t nbins, Double_t min, Double_t max, Int_t nspecies)
{
   //The Fill...Hist() methods fill the histograms with the real limits on the variables
   //This method allows to refill the specified histogram with user-set boundaries min and max
   //Parameters:
   //type = 1 - histogram of x variable #nvar
   //     = 2 - histogram of y variable #nvar
   //     = 3 - histogram of y_pdf for y #nvar and species #nspecies
   //     = 4 - histogram of x variable #nvar, species #nspecies, WITH sWeights
   //     = 5 - histogram of y variable #nvar, species #nspecies, WITH sWeights

   if (type<1 || type>5){
      Error("RefillHist", "type must lie between 1 and 5");
      return;
   }
   char name[20];
   Int_t j;
   TH1D *hremove;
   if (type==1){
      hremove = (TH1D*)fXvarHists.RemoveAt(nvar);
      delete hremove;
      snprintf(name,20,"x%d",nvar);
      TH1D *h = new TH1D(name, name, nbins, min, max);
      for (j=0; j<fNevents;j++)
         h->Fill(fXvar(j, nvar));
      fXvarHists.AddAt(h, nvar);
   }
   if (type==2){
      hremove = (TH1D*)fYvarHists.RemoveAt(nvar);
      delete hremove;
      snprintf(name,20, "y%d", nvar);
      TH1D *h = new TH1D(name, name, nbins, min, max);
      for (j=0; j<fNevents;j++)
         h->Fill(fYvar(j, nvar));
      fXvarHists.AddAt(h, nvar);
   }
   if (type==3){
      hremove = (TH1D*)fYpdfHists.RemoveAt(nspecies*fNy+nvar);
      delete hremove;
      snprintf(name,20, "pdf_species%d_y%d", nspecies, nvar);
      TH1D *h=new TH1D(name, name, nbins, min, max);
      for (j=0; j<fNevents; j++)
         h->Fill(fYpdf(j, nspecies*fNy+nvar));
      fYpdfHists.AddAt(h, nspecies*fNy+nvar);
   }
   if (type==4){
      hremove = (TH1D*)fSWeightsHists.RemoveAt(fNSpecies*nvar+nspecies);
      delete hremove;
      snprintf(name,20, "x%d_species%d", nvar, nspecies);
      TH1D *h = new TH1D(name, name, nbins, min, max);
      h->Sumw2();
      for (Int_t ievent=0; ievent<fNevents; ievent++)
         h->Fill(fXvar(ievent, nvar), fSWeights(ievent, nspecies));
      fSWeightsHists.AddAt(h, fNSpecies*nvar+nspecies);
   }
   if (type==5){
      hremove = (TH1D*)fSWeightsHists.RemoveAt(fNx*fNSpecies + fNSpecies*nvar+nspecies);
      delete hremove;
      snprintf(name,20, "y%d_species%d", nvar, nspecies);
      TH1D *h = new TH1D(name, name, nbins, min, max);
      h->Sumw2();
      for (Int_t ievent=0; ievent<fNevents; ievent++)
         h->Fill(fYvar(ievent, nvar), fSWeights(ievent, nspecies));
      fSWeightsHists.AddAt(h, fNx*fNSpecies + fNSpecies*nvar+nspecies);
   }
}
//____________________________________________________________________
TH1D *TSPlot::GetSWeightsHist(Int_t ixvar, Int_t ispecies,Int_t iyexcl)
{
   //Returns the histogram of a variable, weithed with sWeights
   //If histograms have not been already filled, they are filled with default binning 50
   //If parameter ixvar!=-1, the histogram of x-variable #ixvar is returned for species ispecies
   //If parameter ixvar==-1, the histogram of y-variable #iyexcl is returned for species ispecies
   //If the histogram has already been filled and the binning is different from the parameter nbins
   //all histograms with old binning will be deleted and refilled.


   Int_t nbins = 50; //default binning
   if (fSWeightsHists.IsEmpty())
      FillSWeightsHists(nbins);

   if (ixvar==-1)
      return (TH1D*)fSWeightsHists.UncheckedAt(fNx*fNSpecies + fNSpecies*iyexcl+ispecies);
   else
      return (TH1D*)fSWeightsHists.UncheckedAt(fNSpecies*ixvar + ispecies);

}


//____________________________________________________________________
void TSPlot::SetTree(TTree *tree)
{
   // Set the input Tree
   fTree = tree;
}

//____________________________________________________________________
void TSPlot::SetTreeSelection(const char* varexp, const char *selection, Long64_t firstentry)
{
   //Specifies the variables from the tree to be used for splot
   //
   //Variables fNx, fNy, fNSpecies and fNEvents should already be set!
   //
   //In the 1st parameter it is assumed that first fNx variables are x(control variables),
   //then fNy y variables (discriminating variables),
   //then fNy*fNSpecies ypdf variables (probability distribution functions of dicriminating
   //variables for different species). The order of pdfs should be: species0_y0, species0_y1,...
   //species1_y0, species1_y1,...species[fNSpecies-1]_y0...
   //The 2nd parameter allows to make a cut
   //TTree::Draw method description contains more details on specifying expression and selection

   TTreeFormula **var;
   std::vector<TString> cnames;
   TList *formulaList = new TList();
   TSelectorDraw *selector = (TSelectorDraw*)(((TTreePlayer*)fTree->GetPlayer())->GetSelector());

   Long64_t entry, entryNumber;
   Int_t i,nch;
   Int_t ncols;
   TObjArray *leaves = fTree->GetListOfLeaves();

   fTreename= new TString(fTree->GetName());
   if (varexp)
      fVarexp = new TString(varexp);
   if (selection)
      fSelection= new TString(selection);

   nch = varexp ? strlen(varexp) : 0;


//*-*- Compile selection expression if there is one
   TTreeFormula *select = 0;
   if (selection && strlen(selection)) {
      select = new TTreeFormula("Selection",selection,fTree);
      if (!select) return;
      if (!select->GetNdim()) { delete select; return; }
      formulaList->Add(select);
   }
//*-*- if varexp is empty, take first nx + ny + ny*nspecies columns by default

   if (nch == 0) {
      ncols = fNx + fNy + fNy*fNSpecies;
      for (i=0;i<ncols;i++) {
         cnames.push_back( leaves->At(i)->GetName() );
      }
//*-*- otherwise select only the specified columns
   } else {
      ncols = selector->SplitNames(varexp,cnames);
   }
   var = new TTreeFormula* [ncols];
   Double_t *xvars = new Double_t[ncols];

   fMinmax.ResizeTo(2, ncols);
   for (i=0; i<ncols; i++){
      fMinmax(0, i)=1e30;
      fMinmax(1, i)=-1e30;
   }

//*-*- Create the TreeFormula objects corresponding to each column
   for (i=0;i<ncols;i++) {
      var[i] = new TTreeFormula("Var1",cnames[i].Data(),fTree);
      formulaList->Add(var[i]);
   }

//*-*- Create a TreeFormulaManager to coordinate the formulas
   TTreeFormulaManager *manager=0;
   if (formulaList->LastIndex()>=0) {
      manager = new TTreeFormulaManager;
      for(i=0;i<=formulaList->LastIndex();i++) {
         manager->Add((TTreeFormula*)formulaList->At(i));
      }
      manager->Sync();
   }
//*-*- loop on all selected entries
   // fSelectedRows = 0;
   Int_t tnumber = -1;
   Long64_t selectedrows=0;
   for (entry=firstentry;entry<firstentry+fNevents;entry++) {
      entryNumber = fTree->GetEntryNumber(entry);
      if (entryNumber < 0) break;
      Long64_t localEntry = fTree->LoadTree(entryNumber);
      if (localEntry < 0) break;
      if (tnumber != fTree->GetTreeNumber()) {
         tnumber = fTree->GetTreeNumber();
         if (manager) manager->UpdateFormulaLeaves();
      }
      Int_t ndata = 1;
      if (manager && manager->GetMultiplicity()) {
         ndata = manager->GetNdata();
      }

      for(Int_t inst=0;inst<ndata;inst++) {
         Bool_t loaded = kFALSE;
         if (select) {
            if (select->EvalInstance(inst) == 0) {
               continue;
            }
         }

         if (inst==0) loaded = kTRUE;
         else if (!loaded) {
            // EvalInstance(0) always needs to be called so that
            // the proper branches are loaded.
            for (i=0;i<ncols;i++) {
               var[i]->EvalInstance(0);
            }
            loaded = kTRUE;
         }

         for (i=0;i<ncols;i++) {
            xvars[i] = var[i]->EvalInstance(inst);
         }

         // curentry = entry-firstentry;
         //printf("event#%d\n", curentry);
         //for (i=0; i<ncols; i++)
          //  printf("xvars[%d]=%f\n", i, xvars[i]);
         //selectedrows++;
         for (i=0; i<fNx; i++){
            fXvar(selectedrows, i) = xvars[i];
            if (fXvar(selectedrows, i) < fMinmax(0, i))
               fMinmax(0, i)=fXvar(selectedrows, i);
            if (fXvar(selectedrows, i) > fMinmax(1, i))
               fMinmax(1, i)=fXvar(selectedrows, i);
         }
         for (i=0; i<fNy; i++){
            fYvar(selectedrows, i) = xvars[i+fNx];
            //printf("y_in_loop(%d, %d)=%f, xvars[%d]=%f\n", selectedrows, i, fYvar(selectedrows, i), i+fNx, xvars[i+fNx]);
            if (fYvar(selectedrows, i) < fMinmax(0, i+fNx))
               fMinmax(0, i+fNx) = fYvar(selectedrows, i);
            if (fYvar(selectedrows, i) > fMinmax(1, i+fNx))
               fMinmax(1, i+fNx) = fYvar(selectedrows, i);
            for (Int_t j=0; j<fNSpecies; j++){
               fYpdf(selectedrows, j*fNy + i)=xvars[j*fNy + i+fNx+fNy];
               if (fYpdf(selectedrows, j*fNy+i) < fMinmax(0, j*fNy+i+fNx+fNy))
                  fMinmax(0, j*fNy+i+fNx+fNy) = fYpdf(selectedrows, j*fNy+i);
               if (fYpdf(selectedrows, j*fNy+i) > fMinmax(1, j*fNy+i+fNx+fNy))
                  fMinmax(1, j*fNy+i+fNx+fNy) = fYpdf(selectedrows, j*fNy+i);
            }
         }
      selectedrows++;
      }
   }
   fNevents=selectedrows;
  // for (i=0; i<fNevents; i++){
    //  printf("event#%d\n", i);
      //for (Int_t iy=0; iy<fNy; iy++)
        // printf("y[%d]=%f\n", iy, fYvar(i, iy));
      //for (Int_t ispecies=0; ispecies<fNSpecies; ispecies++){
      //   for (Int_t iy=0; iy<fNy; iy++)
        //    printf("ypdf[sp. %d, y %d]=%f\n", ispecies, iy, fYpdf(i, ispecies*fNy+iy));
     // }
   //}
   delete [] xvars;
   delete [] var;
}

//____________________________________________________________________
void Yields(Int_t &, Double_t *, Double_t &f, Double_t *x, Int_t /*iflag*/)
{
// FCN-function for Minuit

   Double_t lik;
   Int_t i, ispecies;

   TVirtualFitter *fitter = TVirtualFitter::GetFitter();
   TMatrixD *pdftot = (TMatrixD*)fitter->GetObjectFit();
   Int_t nev = pdftot->GetNrows();
   Int_t nes = pdftot->GetNcols();
   f=0;
   for (i=0; i<nev; i++){
      lik=0;
      for (ispecies=0; ispecies<nes; ispecies++)
         lik+=x[ispecies]*(*pdftot)(i, ispecies);
      if (lik<0) lik=1;
      f+=TMath::Log(lik);
   }
   //extended likelihood, equivalent to chi2
   Double_t ntot=0;
   for (i=0; i<nes; i++)
      ntot += x[i];
   f = -2*(f-ntot);
}

