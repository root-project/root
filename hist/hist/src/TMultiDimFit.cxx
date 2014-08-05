// @(#)root/hist:$Id$
// Author: Christian Holm Christensen 07/11/2000

//____________________________________________________________________
//
//
// Begin_Html
/*
 </pre>
 <H1><A NAME="SECTION00010000000000000000">
 Multidimensional Fits in ROOT</A>
 </H1>

 <H1><A NAME="SECTION00020000000000000000"></A>
 <A NAME="sec:overview"></A><BR>
 Overview
 </H1>

 <P>
 A common problem encountered in different fields of applied science is
 to find an expression for one physical quantity in terms of several
 others, which are directly measurable.

 <P>
 An example in high energy physics is the evaluation of the momentum of
 a charged particle from the observation of its trajectory in a magnetic
 field.  The problem is to relate the momentum of the particle to the
 observations, which may consists of of positional measurements at
 intervals along the particle trajectory.

 <P>
 The exact functional relationship between the measured quantities
 (e.g., the space-points) and the dependent quantity (e.g., the
 momentum) is in general not known, but one possible way of solving the
 problem, is to find an expression which reliably approximates the
 dependence of the momentum on the observations.

 <P>
 This explicit function of the observations can be obtained by a
 <I>least squares</I> fitting procedure applied to a representive
 sample of the data, for which the dependent quantity (e.g., momentum)
 and the independent observations are known. The function can then be
 used to compute the quantity of interest for new observations of the
 independent variables.

 <P>
 This class <TT>TMultiDimFit</TT> implements such a procedure in
 ROOT. It is largely based on the CERNLIB MUDIFI package
 [<A
 HREF="TMultiFimFit.html#mudifi">2</A>]. Though the basic concepts are still sound, and
 therefore kept, a few implementation details have changed, and this
 class can take advantage of MINUIT [<A
 HREF="TMultiFimFit.html#minuit">4</A>] to improve the errors
 of the fitting, thanks to the class <TT>TMinuit</TT>.

 <P>
 In [<A
 HREF="TMultiFimFit.html#wind72">5</A>] and [<A
 HREF="TMultiFimFit.html#wind81">6</A>] H. Wind demonstrates the utility
 of this procedure in the context of tracking, magnetic field
 parameterisation, and so on. The outline of the method used in this
 class is based on Winds discussion, and I refer these two excellents
 text for more information.

 <P>
 And example of usage is given in
 <A NAME="tex2html1"
 HREF="
 ./examples/multidimfit.C"><TT>$ROOTSYS/tutorials/fit/multidimfit.C</TT></A>.

 <P>

 <H1><A NAME="SECTION00030000000000000000"></A>
 <A NAME="sec:method"></A><BR>
 The Method
 </H1>

 <P>
 Let <IMG
 WIDTH="18" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img7.gif"
 ALT="$ D$"> by the dependent quantity of interest, which depends smoothly
 on the observable quantities <!-- MATH
 $x_1, \ldots, x_N$
 -->
 <IMG
 WIDTH="80" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img8.gif"
 ALT="$ x_1, \ldots, x_N$">, which we'll denote by
 <!-- MATH
 $\mathbf{x}$
 -->
 <IMG
 WIDTH="14" HEIGHT="13" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img9.gif"
 ALT="$ \mathbf{x}$">. Given a training sample of <IMG
 WIDTH="21" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img10.gif"
 ALT="$ M$"> tuples of the form,
 (<A NAME="tex2html2"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:AddRow"><TT>TMultiDimFit::AddRow</TT></A>)
 <!-- MATH
 \begin{displaymath}
 \left(\mathbf{x}_j, D_j, E_j\right)\quad,
 \end{displaymath}
 -->
 <P></P><DIV ALIGN="CENTER">
 <IMG
 WIDTH="108" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img11.gif"
 ALT="$\displaystyle \left(\mathbf{x}_j, D_j, E_j\right)\quad,
 $">
 </DIV><P></P>
 where <!-- MATH
 $\mathbf{x}_j = (x_{1,j},\ldots,x_{N,j})$
 -->
 <IMG
 WIDTH="148" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img12.gif"
 ALT="$ \mathbf{x}_j = (x_{1,j},\ldots,x_{N,j})$"> are <IMG
 WIDTH="19" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img13.gif"
 ALT="$ N$"> independent
 variables, <IMG
 WIDTH="24" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img14.gif"
 ALT="$ D_j$"> is the known, quantity dependent at <!-- MATH
 $\mathbf{x}_j$
 -->
 <IMG
 WIDTH="20" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img15.gif"
 ALT="$ \mathbf{x}_j$">,
 and <IMG
 WIDTH="23" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img16.gif"
 ALT="$ E_j$"> is the square error in <IMG
 WIDTH="24" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img14.gif"
 ALT="$ D_j$">, the class
 <A NAME="tex2html3"
 HREF="./TMultiDimFit.html"><TT>TMultiDimFit</TT></A>
 will
 try to find the parameterization
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="Dp"></A><!-- MATH
 \begin{equation}
 D_p(\mathbf{x}) = \sum_{l=1}^{L} c_l \prod_{i=1}^{N} p_{li}\left(x_i\right)
 = \sum_{l=1}^{L} c_l F_l(\mathbf{x})
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="274" HEIGHT="65" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img17.gif"
 ALT="$\displaystyle D_p(\mathbf{x}) = \sum_{l=1}^{L} c_l \prod_{i=1}^{N} p_{li}\left(x_i\right) = \sum_{l=1}^{L} c_l F_l(\mathbf{x})$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (1)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 such that
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="S"></A><!-- MATH
 \begin{equation}
 S \equiv \sum_{j=1}^{M} \left(D_j - D_p\left(\mathbf{x}_j\right)\right)^2
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="172" HEIGHT="65" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img18.gif"
 ALT="$\displaystyle S \equiv \sum_{j=1}^{M} \left(D_j - D_p\left(\mathbf{x}_j\right)\right)^2$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (2)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 is minimal. Here <!-- MATH
 $p_{li}(x_i)$
 -->
 <IMG
 WIDTH="48" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img19.gif"
 ALT="$ p_{li}(x_i)$"> are monomials, or Chebyshev or Legendre
 polynomials, labelled <!-- MATH
 $l = 1, \ldots, L$
 -->
 <IMG
 WIDTH="87" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img20.gif"
 ALT="$ l = 1, \ldots, L$">, in each variable
 <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img21.gif"
 ALT="$ x_i$">, <!-- MATH
 $i=1, \ldots, N$
 -->
 <IMG
 WIDTH="91" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img22.gif"
 ALT="$ i=1, \ldots, N$">.

 <P>
 So what <TT>TMultiDimFit</TT> does, is to determine the number of
 terms <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img23.gif"
 ALT="$ L$">, and then <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img23.gif"
 ALT="$ L$"> terms (or functions) <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$">, and the <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img23.gif"
 ALT="$ L$">
 coefficients <IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img25.gif"
 ALT="$ c_l$">, so that <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> is minimal
 (<A NAME="tex2html4"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:FindParameterization"><TT>TMultiDimFit::FindParameterization</TT></A>).

 <P>
 Of course it's more than a little unlikely that <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> will ever become
 exact zero as a result of the procedure outlined below. Therefore, the
 user is asked to provide a minimum relative error <IMG
 WIDTH="11" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img27.gif"
 ALT="$ \epsilon$">
 (<A NAME="tex2html5"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetMinRelativeError"><TT>TMultiDimFit::SetMinRelativeError</TT></A>), and <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$">
 will be considered minimized when
 <!-- MATH
 \begin{displaymath}
 R = \frac{S}{\sum_{j=1}^M D_j^2} < \epsilon
 \end{displaymath}
 -->
 <P></P><DIV ALIGN="CENTER">
 <IMG
 WIDTH="132" HEIGHT="51" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img28.gif"
 ALT="$\displaystyle R = \frac{S}{\sum_{j=1}^M D_j^2} &lt; \epsilon
 $">
 </DIV><P></P>

 <P>
 Optionally, the user may impose a functional expression by specifying
 the powers of each variable in <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img23.gif"
 ALT="$ L$"> specified functions <!-- MATH
 $F_1, \ldots,
 F_L$
 -->
 <IMG
 WIDTH="79" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img29.gif"
 ALT="$ F_1, \ldots,
 F_L$"> (<A NAME="tex2html6"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetPowers"><TT>TMultiDimFit::SetPowers</TT></A>). In that case, only the
 coefficients <IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img25.gif"
 ALT="$ c_l$"> is calculated by the class.

 <P>

 <H2><A NAME="SECTION00031000000000000000"></A>
 <A NAME="sec:selection"></A><BR>
 Limiting the Number of Terms
 </H2>

 <P>
 As always when dealing with fits, there's a real chance of
 <I>over fitting</I>. As is well-known, it's always possible to fit an
 <IMG
 WIDTH="46" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img30.gif"
 ALT="$ N-1$"> polynomial in <IMG
 WIDTH="13" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img31.gif"
 ALT="$ x$"> to <IMG
 WIDTH="19" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img13.gif"
 ALT="$ N$"> points <IMG
 WIDTH="41" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img32.gif"
 ALT="$ (x,y)$"> with <!-- MATH
 $\chi^2 = 0$
 -->
 <IMG
 WIDTH="50" HEIGHT="33" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img33.gif"
 ALT="$ \chi^2 = 0$">, but
 the polynomial is not likely to fit new data at all
 [<A
 HREF="TMultiFimFit.html#bevington">1</A>]. Therefore, the user is asked to provide an upper
 limit, <IMG
 WIDTH="41" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img34.gif"
 ALT="$ L_{max}$"> to the number of terms in <IMG
 WIDTH="25" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img35.gif"
 ALT="$ D_p$">
 (<A NAME="tex2html7"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetMaxTerms"><TT>TMultiDimFit::SetMaxTerms</TT></A>).

 <P>
 However, since there's an infinite number of <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$"> to choose from, the
 user is asked to give the maximum power. <IMG
 WIDTH="49" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img36.gif"
 ALT="$ P_{max,i}$">, of each variable
 <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img21.gif"
 ALT="$ x_i$"> to be considered in the minimization of <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$">
 (<A NAME="tex2html8"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetMaxPowers"><TT>TMultiDimFit::SetMaxPowers</TT></A>).

 <P>
 One way of obtaining values for the maximum power in variable <IMG
 WIDTH="10" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img37.gif"
 ALT="$ i$">, is
 to perform a regular fit to the dependent quantity <IMG
 WIDTH="18" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img7.gif"
 ALT="$ D$">, using a
 polynomial only in <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img21.gif"
 ALT="$ x_i$">. The maximum power is <IMG
 WIDTH="49" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img36.gif"
 ALT="$ P_{max,i}$"> is then the
 power that does not significantly improve the one-dimensional
 least-square fit over <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img21.gif"
 ALT="$ x_i$"> to <IMG
 WIDTH="18" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img7.gif"
 ALT="$ D$"> [<A
 HREF="TMultiFimFit.html#wind72">5</A>].

 <P>
 There are still a huge amount of possible choices for <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$">; in fact
 there are <!-- MATH
 $\prod_{i=1}^{N} (P_{max,i} + 1)$
 -->
 <IMG
 WIDTH="125" HEIGHT="39" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img38.gif"
 ALT="$ \prod_{i=1}^{N} (P_{max,i} + 1)$"> possible
 choices. Obviously we need to limit this. To this end, the user is
 asked to set a <I>power control limit</I>, <IMG
 WIDTH="17" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img39.gif"
 ALT="$ Q$">
 (<A NAME="tex2html9"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetPowerLimit"><TT>TMultiDimFit::SetPowerLimit</TT></A>), and a function
 <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$"> is only accepted if
 <!-- MATH
 \begin{displaymath}
 Q_l = \sum_{i=1}^{N} \frac{P_{li}}{P_{max,i}} < Q
 \end{displaymath}
 -->
 <P></P><DIV ALIGN="CENTER">
 <IMG
 WIDTH="151" HEIGHT="65" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img40.gif"
 ALT="$\displaystyle Q_l = \sum_{i=1}^{N} \frac{P_{li}}{P_{max,i}} &lt; Q
 $">
 </DIV><P></P>
 where <IMG
 WIDTH="24" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img41.gif"
 ALT="$ P_{li}$"> is the leading power of variable <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img21.gif"
 ALT="$ x_i$"> in function
 <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$">. (<A NAME="tex2html10"
 HREF="

 ./TMultiDimFit.html#TMultiDimFit:MakeCandidates"><TT>TMultiDimFit::MakeCandidates</TT></A>). So the number of
 functions increase with <IMG
 WIDTH="17" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img39.gif"
 ALT="$ Q$"> (1, 2 is fine, 5 is way out).

 <P>

 <H2><A NAME="SECTION00032000000000000000">
 Gram-Schmidt Orthogonalisation</A>
 </H2>

 <P>
 To further reduce the number of functions in the final expression,
 only those functions that significantly reduce <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> is chosen. What
 `significant' means, is chosen by the user, and will be
 discussed below (see&nbsp;<A HREF="TMultiFimFit.html#sec:selectiondetail">2.3</A>).

 <P>
 The functions <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$"> are generally not orthogonal, which means one will
 have to evaluate all possible <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$">'s over all data-points before
 finding the most significant [<A
 HREF="TMultiFimFit.html#bevington">1</A>]. We can, however, do
 better then that. By applying the <I>modified Gram-Schmidt
 orthogonalisation</I> algorithm [<A
 HREF="TMultiFimFit.html#wind72">5</A>] [<A
 HREF="TMultiFimFit.html#golub">3</A>] to the
 functions <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$">, we can evaluate the contribution to the reduction of
 <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> from each function in turn, and we may delay the actual inversion
 of the curvature-matrix
 (<A NAME="tex2html11"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:MakeGramSchmidt"><TT>TMultiDimFit::MakeGramSchmidt</TT></A>).

 <P>
 So we are let to consider an <IMG
 WIDTH="52" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img42.gif"
 ALT="$ M\times L$"> matrix <!-- MATH
 $\mathsf{F}$
 -->
 <IMG
 WIDTH="13" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img43.gif"
 ALT="$ \mathsf{F}$">, an
 element of which is given by
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:Felem"></A><!-- MATH
 \begin{equation}
 f_{jl} = F_j\left(x_{1j} , x_{2j}, \ldots, x_{Nj}\right)
 = F_l(\mathbf{x}_j)\,  \quad\mbox{with}~j=1,2,\ldots,M,
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="260" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img44.gif"
 ALT="$\displaystyle f_{jl} = F_j\left(x_{1j} , x_{2j}, \ldots, x_{Nj}\right) = F_l(\mathbf{x}_j) $">&nbsp; &nbsp;with<IMG
 WIDTH="120" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img45.gif"
 ALT="$\displaystyle &nbsp;j=1,2,\ldots,M,$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (3)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 where <IMG
 WIDTH="12" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img46.gif"
 ALT="$ j$"> labels the <IMG
 WIDTH="21" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img10.gif"
 ALT="$ M$"> rows in the training sample and <IMG
 WIDTH="9" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img47.gif"
 ALT="$ l$"> labels
 <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img23.gif"
 ALT="$ L$"> functions of <IMG
 WIDTH="19" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img13.gif"
 ALT="$ N$"> variables, and <IMG
 WIDTH="53" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img48.gif"
 ALT="$ L \leq M$">. That is, <IMG
 WIDTH="23" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img49.gif"
 ALT="$ f_{jl}$"> is
 the term (or function) numbered <IMG
 WIDTH="9" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img47.gif"
 ALT="$ l$"> evaluated at the data point
 <IMG
 WIDTH="12" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img46.gif"
 ALT="$ j$">. We have to normalise <!-- MATH
 $\mathbf{x}_j$
 -->
 <IMG
 WIDTH="20" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img15.gif"
 ALT="$ \mathbf{x}_j$"> to <IMG
 WIDTH="48" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img50.gif"
 ALT="$ [-1,1]$"> for this to
 succeed [<A
 HREF="TMultiFimFit.html#wind72">5</A>]
 (<A NAME="tex2html12"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:MakeNormalized"><TT>TMultiDimFit::MakeNormalized</TT></A>). We then define a
 matrix <!-- MATH
 $\mathsf{W}$
 -->
 <IMG
 WIDTH="19" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img51.gif"
 ALT="$ \mathsf{W}$"> of which the columns <!-- MATH
 $\mathbf{w}_j$
 -->
 <IMG
 WIDTH="24" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img52.gif"
 ALT="$ \mathbf{w}_j$"> are given by
 <BR>
 <DIV ALIGN="CENTER"><A NAME="eq:wj"></A><!-- MATH
 \begin{eqnarray}
 \mathbf{w}_1 &=& \mathbf{f}_1 = F_1\left(\mathbf x_1\right)\\
 \mathbf{w}_l &=& \mathbf{f}_l - \sum^{l-1}_{k=1} \frac{\mathbf{f}_l \bullet
 \mathbf{w}_k}{\mathbf{w}_k^2}\mathbf{w}_k\,.
 \end{eqnarray}
 -->
 <TABLE CELLPADDING="0" ALIGN="CENTER" WIDTH="100%">
 <TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT"><IMG
 WIDTH="25" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img53.gif"
 ALT="$\displaystyle \mathbf{w}_1$"></TD>
 <TD WIDTH="10" ALIGN="CENTER" NOWRAP><IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img54.gif"
 ALT="$\displaystyle =$"></TD>
 <TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="87" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img55.gif"
 ALT="$\displaystyle \mathbf{f}_1 = F_1\left(\mathbf x_1\right)$"></TD>
 <TD WIDTH=10 ALIGN="RIGHT">
 (4)</TD></TR>
 <TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT"><IMG
 WIDTH="22" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img56.gif"
 ALT="$\displaystyle \mathbf{w}_l$"></TD>
 <TD WIDTH="10" ALIGN="CENTER" NOWRAP><IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img54.gif"
 ALT="$\displaystyle =$"></TD>
 <TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="138" HEIGHT="66" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img57.gif"
 ALT="$\displaystyle \mathbf{f}_l - \sum^{l-1}_{k=1} \frac{\mathbf{f}_l \bullet
 \mathbf{w}_k}{\mathbf{w}_k^2}\mathbf{w}_k .$"></TD>
 <TD WIDTH=10 ALIGN="RIGHT">
 (5)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 and <!-- MATH
 $\mathbf{w}_{l}$
 -->
 <IMG
 WIDTH="22" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img58.gif"
 ALT="$ \mathbf{w}_{l}$"> is the component of <!-- MATH
 $\mathbf{f}_{l}$
 -->
 <IMG
 WIDTH="15" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img59.gif"
 ALT="$ \mathbf{f}_{l}$"> orthogonal
 to <!-- MATH
 $\mathbf{w}_{1}, \ldots, \mathbf{w}_{l-1}$
 -->
 <IMG
 WIDTH="97" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img60.gif"
 ALT="$ \mathbf{w}_{1}, \ldots, \mathbf{w}_{l-1}$">. Hence we obtain
 [<A
 HREF="TMultiFimFit.html#golub">3</A>],
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:worto"></A><!-- MATH
 \begin{equation}
 \mathbf{w}_k\bullet\mathbf{w}_l = 0\quad\mbox{if}~k \neq l\quad.
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="87" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img61.gif"
 ALT="$\displaystyle \mathbf{w}_k\bullet\mathbf{w}_l = 0$">&nbsp; &nbsp;if<IMG
 WIDTH="65" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img62.gif"
 ALT="$\displaystyle &nbsp;k \neq l\quad.$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (6)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>

 <P>
 We now take as a new model <!-- MATH
 $\mathsf{W}\mathbf{a}$
 -->
 <IMG
 WIDTH="28" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img63.gif"
 ALT="$ \mathsf{W}\mathbf{a}$">. We thus want to
 minimize
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:S"></A><!-- MATH
 \begin{equation}
 S\equiv \left(\mathbf{D} - \mathsf{W}\mathbf{a}\right)^2\quad,
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="136" HEIGHT="38" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img64.gif"
 ALT="$\displaystyle S\equiv \left(\mathbf{D} - \mathsf{W}\mathbf{a}\right)^2\quad,$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (7)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 where <!-- MATH
 $\mathbf{D} = \left(D_1,\ldots,D_M\right)$
 -->
 <IMG
 WIDTH="137" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img65.gif"
 ALT="$ \mathbf{D} = \left(D_1,\ldots,D_M\right)$"> is a vector of the
 dependent quantity in the sample. Differentiation with respect to
 <IMG
 WIDTH="19" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img66.gif"
 ALT="$ a_j$"> gives, using&nbsp;(<A HREF="TMultiFimFit.html#eq:worto">6</A>),
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:dS"></A><!-- MATH
 \begin{equation}
 \mathbf{D}\bullet\mathbf{w}_l - a_l\mathbf{w}_l^2 = 0
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="134" HEIGHT="35" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img67.gif"
 ALT="$\displaystyle \mathbf{D}\bullet\mathbf{w}_l - a_l\mathbf{w}_l^2 = 0$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (8)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 or
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:dS2"></A><!-- MATH
 \begin{equation}
 a_l = \frac{\mathbf{D}_l\bullet\mathbf{w}_l}{\mathbf{w}_l^2}
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="95" HEIGHT="51" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img68.gif"
 ALT="$\displaystyle a_l = \frac{\mathbf{D}_l\bullet\mathbf{w}_l}{\mathbf{w}_l^2}$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (9)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 Let <IMG
 WIDTH="21" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img69.gif"
 ALT="$ S_j$"> be the sum of squares of residuals when taking <IMG
 WIDTH="12" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img46.gif"
 ALT="$ j$"> functions
 into account. Then
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:Sj"></A><!-- MATH
 \begin{equation}
 S_l = \left[\mathbf{D} - \sum^l_{k=1} a_k\mathbf{w}_k\right]^2
 = \mathbf{D}^2 - 2\mathbf{D} \sum^l_{k=1} a_k\mathbf{w}_k
 + \sum^l_{k=1} a_k^2\mathbf{w}_k^2
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="394" HEIGHT="72" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img70.gif"
 ALT="$\displaystyle S_l = \left[\mathbf{D} - \sum^l_{k=1} a_k\mathbf{w}_k\right]^2 = ...
 ...2 - 2\mathbf{D} \sum^l_{k=1} a_k\mathbf{w}_k + \sum^l_{k=1} a_k^2\mathbf{w}_k^2$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (10)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 Using (<A HREF="TMultiFimFit.html#eq:dS2">9</A>), we see that
 <BR>
 <DIV ALIGN="CENTER"><A NAME="eq:sj2"></A><!-- MATH
 \begin{eqnarray}
 S_l &=& \mathbf{D}^2 - 2 \sum^l_{k=1} a_k^2\mathbf{w}_k^2 +
 \sum^j_{k=1} a_k^2\mathbf{w}_k^2\nonumber\\
 &=& \mathbf{D}^2 - \sum^l_{k=1} a_k^2\mathbf{w}_k^2\nonumber\\
 &=& \mathbf{D}^2 - \sum^l_{k=1} \frac{\left(\mathbf D\bullet \mathbf
 w_k\right)}{\mathbf w_k^2}
 \end{eqnarray}
 -->
 <TABLE CELLPADDING="0" ALIGN="CENTER" WIDTH="100%">
 <TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT"><IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img71.gif"
 ALT="$\displaystyle S_l$"></TD>
 <TD WIDTH="10" ALIGN="CENTER" NOWRAP><IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img54.gif"
 ALT="$\displaystyle =$"></TD>
 <TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="201" HEIGHT="67" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img72.gif"
 ALT="$\displaystyle \mathbf{D}^2 - 2 \sum^l_{k=1} a_k^2\mathbf{w}_k^2 +
 \sum^j_{k=1} a_k^2\mathbf{w}_k^2$"></TD>
 <TD WIDTH=10 ALIGN="RIGHT">
 &nbsp;</TD></TR>
 <TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT">&nbsp;</TD>
 <TD WIDTH="10" ALIGN="CENTER" NOWRAP><IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img54.gif"
 ALT="$\displaystyle =$"></TD>
 <TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="108" HEIGHT="66" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img73.gif"
 ALT="$\displaystyle \mathbf{D}^2 - \sum^l_{k=1} a_k^2\mathbf{w}_k^2$"></TD>
 <TD WIDTH=10 ALIGN="RIGHT">
 &nbsp;</TD></TR>
 <TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT">&nbsp;</TD>
 <TD WIDTH="10" ALIGN="CENTER" NOWRAP><IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img54.gif"
 ALT="$\displaystyle =$"></TD>
 <TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="137" HEIGHT="66" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img74.gif"
 ALT="$\displaystyle \mathbf{D}^2 - \sum^l_{k=1} \frac{\left(\mathbf D\bullet \mathbf
 w_k\right)}{\mathbf w_k^2}$"></TD>
 <TD WIDTH=10 ALIGN="RIGHT">
 (11)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>

 <P>
 So for each new function <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$"> included in the model, we get a
 reduction of the sum of squares of residuals of <!-- MATH
 $a_l^2\mathbf{w}_l^2$
 -->
 <IMG
 WIDTH="40" HEIGHT="33" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img75.gif"
 ALT="$ a_l^2\mathbf{w}_l^2$">,
 where <!-- MATH
 $\mathbf{w}_l$
 -->
 <IMG
 WIDTH="22" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img76.gif"
 ALT="$ \mathbf{w}_l$"> is given by (<A HREF="TMultiFimFit.html#eq:wj">4</A>) and <IMG
 WIDTH="17" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img77.gif"
 ALT="$ a_l$"> by
 (<A HREF="TMultiFimFit.html#eq:dS2">9</A>). Thus, using the Gram-Schmidt orthogonalisation, we
 can decide if we want to include this function in the final model,
 <I>before</I> the matrix inversion.

 <P>

 <H2><A NAME="SECTION00033000000000000000"></A>
 <A NAME="sec:selectiondetail"></A><BR>
 Function Selection Based on Residual
 </H2>

 <P>
 Supposing that <IMG
 WIDTH="42" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img78.gif"
 ALT="$ L-1$"> steps of the procedure have been performed, the
 problem now is to consider the <!-- MATH
 $L^{\mbox{th}}$
 -->
 <IMG
 WIDTH="31" HEIGHT="20" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img79.gif"
 ALT="$ L^{\mbox{th}}$"> function.

 <P>
 The sum of squares of residuals can be written as
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:sums"></A><!-- MATH
 \begin{equation}
 S_L = \textbf{D}^T\bullet\textbf{D} -
 \sum^L_{l=1}a^2_l\left(\textbf{w}_l^T\bullet\textbf{w}_l\right)
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="232" HEIGHT="65" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img80.gif"
 ALT="$\displaystyle S_L = \textbf{D}^T\bullet\textbf{D} - \sum^L_{l=1}a^2_l\left(\textbf{w}_l^T\bullet\textbf{w}_l\right)$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (12)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 where the relation (<A HREF="TMultiFimFit.html#eq:dS2">9</A>) have been taken into account. The
 contribution of the <!-- MATH
 $L^{\mbox{th}}$
 -->
 <IMG
 WIDTH="31" HEIGHT="20" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img79.gif"
 ALT="$ L^{\mbox{th}}$"> function to the reduction of S, is
 given by
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:dSN"></A><!-- MATH
 \begin{equation}
 \Delta S_L = a^2_L\left(\textbf{w}_L^T\bullet\textbf{w}_L\right)
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="154" HEIGHT="36" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img81.gif"
 ALT="$\displaystyle \Delta S_L = a^2_L\left(\textbf{w}_L^T\bullet\textbf{w}_L\right)$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (13)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>

 <P>
 Two test are now applied to decide whether this <!-- MATH
 $L^{\mbox{th}}$
 -->
 <IMG
 WIDTH="31" HEIGHT="20" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img79.gif"
 ALT="$ L^{\mbox{th}}$">
 function is to be included in the final expression, or not.

 <P>

 <H3><A NAME="SECTION00033100000000000000"></A>
 <A NAME="testone"></A><BR>
 Test 1
 </H3>

 <P>
 Denoting by <IMG
 WIDTH="43" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img82.gif"
 ALT="$ H_{L-1}$"> the subspace spanned by
 <!-- MATH
 $\textbf{w}_1,\ldots,\textbf{w}_{L-1}$
 -->
 <IMG
 WIDTH="102" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img83.gif"
 ALT="$ \textbf{w}_1,\ldots,\textbf{w}_{L-1}$"> the function <!-- MATH
 $\textbf{w}_L$
 -->
 <IMG
 WIDTH="27" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img5.gif"
 ALT="$ \textbf {w}_L$"> is
 by construction (see (<A HREF="TMultiFimFit.html#eq:wj">4</A>)) the projection of the function
 <IMG
 WIDTH="24" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img84.gif"
 ALT="$ F_L$"> onto the direction perpendicular to <IMG
 WIDTH="43" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img82.gif"
 ALT="$ H_{L-1}$">. Now, if the
 length of <!-- MATH
 $\textbf{w}_L$
 -->
 <IMG
 WIDTH="27" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img5.gif"
 ALT="$ \textbf {w}_L$"> (given by <!-- MATH
 $\textbf{w}_L\bullet\textbf{w}_L$
 -->
 <IMG
 WIDTH="65" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img85.gif"
 ALT="$ \textbf{w}_L\bullet\textbf{w}_L$">)
 is very small compared to the length of <!-- MATH
 $\textbf{f}_L$
 -->
 <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img3.gif"
 ALT="$ \textbf {f}_L$"> this new
 function can not contribute much to the reduction of the sum of
 squares of residuals. The test consists then in calculating the angle
 <IMG
 WIDTH="12" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img1.gif"
 ALT="$ \theta $"> between the two vectors <!-- MATH
 $\textbf{w}_L$
 -->
 <IMG
 WIDTH="27" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img5.gif"
 ALT="$ \textbf {w}_L$"> and <!-- MATH
 $\textbf{f}_L$
 -->
 <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img3.gif"
 ALT="$ \textbf {f}_L$">
 (see also figure&nbsp;<A HREF="TMultiFimFit.html#fig:thetaphi">1</A>) and requiring that it's
 <I>greater</I> then a threshold value which the user must set
 (<A NAME="tex2html14"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetMinAngle"><TT>TMultiDimFit::SetMinAngle</TT></A>).

 <P>

 <P></P>
 <DIV ALIGN="CENTER"><A NAME="fig:thetaphi"></A><A NAME="519"></A>
 <TABLE>
 <CAPTION ALIGN="BOTTOM"><STRONG>Figure 1:</STRONG>
 (a) Angle <IMG
 WIDTH="12" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img1.gif"
 ALT="$ \theta $"> between <!-- MATH
 $\textbf{w}_l$
 -->
 <IMG
 WIDTH="22" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img2.gif"
 ALT="$ \textbf {w}_l$"> and
 <!-- MATH
 $\textbf{f}_L$
 -->
 <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img3.gif"
 ALT="$ \textbf {f}_L$">, (b) angle <IMG
 WIDTH="14" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img4.gif"
 ALT="$ \phi $"> between <!-- MATH
 $\textbf{w}_L$
 -->
 <IMG
 WIDTH="27" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img5.gif"
 ALT="$ \textbf {w}_L$"> and
 <!-- MATH
 $\textbf{D}$
 -->
 <IMG
 WIDTH="18" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img6.gif"
 ALT="$ \textbf {D}$"></CAPTION>
 <TR><TD><IMG
 WIDTH="466" HEIGHT="172" BORDER="0"
 SRC="gif/multidimfit_img86.gif"
 ALT="\begin{figure}\begin{center}
 \begin{tabular}{p{.4\textwidth}p{.4\textwidth}}
 \...
 ... \put(80,100){$\mathbf{D}$}
 \end{picture} \end{tabular} \end{center}\end{figure}"></TD></TR>
 </TABLE>
 </DIV><P></P>

 <P>

 <H3><A NAME="SECTION00033200000000000000"></A> <A NAME="testtwo"></A><BR>
 Test 2
 </H3>

 <P>
 Let <!-- MATH
 $\textbf{D}$
 -->
 <IMG
 WIDTH="18" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img6.gif"
 ALT="$ \textbf {D}$"> be the data vector to be fitted. As illustrated in
 figure&nbsp;<A HREF="TMultiFimFit.html#fig:thetaphi">1</A>, the <!-- MATH
 $L^{\mbox{th}}$
 -->
 <IMG
 WIDTH="31" HEIGHT="20" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img79.gif"
 ALT="$ L^{\mbox{th}}$"> function <!-- MATH
 $\textbf{w}_L$
 -->
 <IMG
 WIDTH="27" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img5.gif"
 ALT="$ \textbf {w}_L$">
 will contribute significantly to the reduction of <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$">, if the angle
 <!-- MATH
 $\phi^\prime$
 -->
 <IMG
 WIDTH="18" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img87.gif"
 ALT="$ \phi^\prime$"> between <!-- MATH
 $\textbf{w}_L$
 -->
 <IMG
 WIDTH="27" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img5.gif"
 ALT="$ \textbf {w}_L$"> and <!-- MATH
 $\textbf{D}$
 -->
 <IMG
 WIDTH="18" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img6.gif"
 ALT="$ \textbf {D}$"> is smaller than
 an upper limit <IMG
 WIDTH="14" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img4.gif"
 ALT="$ \phi $">, defined by the user
 (<A NAME="tex2html15"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:SetMaxAngle"><TT>TMultiDimFit::SetMaxAngle</TT></A>)

 <P>
 However, the method automatically readjusts the value of this angle
 while fitting is in progress, in order to make the selection criteria
 less and less difficult to be fulfilled. The result is that the
 functions contributing most to the reduction of <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> are chosen first
 (<A NAME="tex2html16"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:TestFunction"><TT>TMultiDimFit::TestFunction</TT></A>).

 <P>
 In case <IMG
 WIDTH="14" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img4.gif"
 ALT="$ \phi $"> isn't defined, an alternative method of
 performing this second test is used: The <!-- MATH
 $L^{\mbox{th}}$
 -->
 <IMG
 WIDTH="31" HEIGHT="20" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img79.gif"
 ALT="$ L^{\mbox{th}}$"> function
 <!-- MATH
 $\textbf{f}_L$
 -->
 <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img3.gif"
 ALT="$ \textbf {f}_L$"> is accepted if (refer also to equation&nbsp;(<A HREF="TMultiFimFit.html#eq:dSN">13</A>))
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:dSN2"></A><!-- MATH
 \begin{equation}
 \Delta S_L > \frac{S_{L-1}}{L_{max}-L}
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="129" HEIGHT="51" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img88.gif"
 ALT="$\displaystyle \Delta S_L &gt; \frac{S_{L-1}}{L_{max}-L}$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (14)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 where  <IMG
 WIDTH="40" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img89.gif"
 ALT="$ S_{L-1}$"> is the sum of the <IMG
 WIDTH="42" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img78.gif"
 ALT="$ L-1$"> first residuals from the
 <IMG
 WIDTH="42" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img78.gif"
 ALT="$ L-1$"> functions previously accepted; and <IMG
 WIDTH="41" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img34.gif"
 ALT="$ L_{max}$"> is the total number
 of functions allowed in the final expression of the fit (defined by
 user).

 <P>
 >From this we see, that by restricting <IMG
 WIDTH="41" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img34.gif"
 ALT="$ L_{max}$"> -- the number of
 terms in the final model -- the fit is more difficult to perform,
 since the above selection criteria is more limiting.

 <P>
 The more coefficients we evaluate, the more the sum of squares of
 residuals <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> will be reduced. We can evaluate <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$"> before inverting
 <!-- MATH
 $\mathsf{B}$
 -->
 <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img90.gif"
 ALT="$ \mathsf{B}$"> as shown below.

 <P>

 <H2><A NAME="SECTION00034000000000000000">
 Coefficients and Coefficient Errors</A>
 </H2>

 <P>
 Having found a parameterization, that is the <IMG
 WIDTH="19" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img24.gif"
 ALT="$ F_l$">'s and <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img23.gif"
 ALT="$ L$">, that
 minimizes <IMG
 WIDTH="15" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img26.gif"
 ALT="$ S$">, we still need to determine the coefficients
 <IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img25.gif"
 ALT="$ c_l$">. However, it's a feature of how we choose the significant
 functions, that the evaluation of the <IMG
 WIDTH="16" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img25.gif"
 ALT="$ c_l$">'s becomes trivial
 [<A
 HREF="TMultiFimFit.html#wind72">5</A>]. To derive <!-- MATH
 $\mathbf{c}$
 -->
 <IMG
 WIDTH="12" HEIGHT="13" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img91.gif"
 ALT="$ \mathbf{c}$">, we first note that
 equation&nbsp;(<A HREF="TMultiFimFit.html#eq:wj">4</A>) can be written as
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:FF"></A><!-- MATH
 \begin{equation}
 \mathsf{F} = \mathsf{W}\mathsf{B}
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="60" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img92.gif"
 ALT="$\displaystyle \mathsf{F} = \mathsf{W}\mathsf{B}$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (15)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 where
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:bij"></A><!-- MATH
 \begin{equation}
 b_{ij} = \left\{\begin{array}{rcl}
 \frac{\mathbf{f}_j \bullet \mathbf{w}_i}{\mathbf{w}_i^2}
 & \mbox{if} & i < j\\
 1 & \mbox{if} & i = j\\
 0 & \mbox{if} & i > j
 \end{array}\right.
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="187" HEIGHT="79" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img93.gif"
 ALT="$\displaystyle b_{ij} = \left\{\begin{array}{rcl} \frac{\mathbf{f}_j \bullet \ma...
 ...f} &amp; i &lt; j\  1 &amp; \mbox{if} &amp; i = j\  0 &amp; \mbox{if} &amp; i &gt; j \end{array}\right.$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (16)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 Consequently, <!-- MATH
 $\mathsf{B}$
 -->
 <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img90.gif"
 ALT="$ \mathsf{B}$"> is an upper triangle matrix, which can be
 readily inverted. So we now evaluate
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:FFF"></A><!-- MATH
 \begin{equation}
 \mathsf{F}\mathsf{B}^{-1} = \mathsf{W}
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="77" HEIGHT="35" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img94.gif"
 ALT="$\displaystyle \mathsf{F}\mathsf{B}^{-1} = \mathsf{W}$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (17)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 The model <!-- MATH
 $\mathsf{W}\mathbf{a}$
 -->
 <IMG
 WIDTH="28" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img63.gif"
 ALT="$ \mathsf{W}\mathbf{a}$"> can therefore be written as
 <!-- MATH
 \begin{displaymath}
 (\mathsf{F}\mathsf{B}^{-1})\mathbf{a} =
 \mathsf{F}(\mathsf{B}^{-1}\mathbf{a})\,.
 \end{displaymath}
 -->
 <P></P><DIV ALIGN="CENTER">
 <IMG
 WIDTH="148" HEIGHT="35" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img95.gif"
 ALT="$\displaystyle (\mathsf{F}\mathsf{B}^{-1})\mathbf{a} =
 \mathsf{F}(\mathsf{B}^{-1}\mathbf{a}) .
 $">
 </DIV><P></P>
 The original model <!-- MATH
 $\mathsf{F}\mathbf{c}$
 -->
 <IMG
 WIDTH="21" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img96.gif"
 ALT="$ \mathsf{F}\mathbf{c}$"> is therefore identical with
 this if
 <P></P>
 <DIV ALIGN="CENTER"><A NAME="eq:id:cond"></A><!-- MATH
 \begin{equation}
 \mathbf{c} = \left(\mathsf{B}^{-1}\mathbf{a}\right) =
 \left[\mathbf{a}^T\left(\mathsf{B}^{-1}\right)^T\right]^T\,.
 \end{equation}
 -->
 <TABLE CELLPADDING="0" WIDTH="100%" ALIGN="CENTER">
 <TR VALIGN="MIDDLE">
 <TD NOWRAP ALIGN="CENTER"><IMG
 WIDTH="214" HEIGHT="51" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img97.gif"
 ALT="$\displaystyle \mathbf{c} = \left(\mathsf{B}^{-1}\mathbf{a}\right) = \left[\mathbf{a}^T\left(\mathsf{B}^{-1}\right)^T\right]^T .$"></TD>
 <TD NOWRAP WIDTH="10" ALIGN="RIGHT">
 (18)</TD></TR>
 </TABLE></DIV>
 <BR CLEAR="ALL"><P></P>
 The reason we use <!-- MATH
 $\left(\mathsf{B}^{-1}\right)^T$
 -->
 <IMG
 WIDTH="56" HEIGHT="42" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img98.gif"
 ALT="$ \left(\mathsf{B}^{-1}\right)^T$"> rather then
 <!-- MATH
 $\mathsf{B}^{-1}$
 -->
 <IMG
 WIDTH="32" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img99.gif"
 ALT="$ \mathsf{B}^{-1}$"> is to save storage, since
 <!-- MATH
 $\left(\mathsf{B}^{-1}\right)^T$
 -->
 <IMG
 WIDTH="56" HEIGHT="42" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img98.gif"
 ALT="$ \left(\mathsf{B}^{-1}\right)^T$"> can be stored in the same matrix as
 <!-- MATH
 $\mathsf{B}$
 -->
 <IMG
 WIDTH="15" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img90.gif"
 ALT="$ \mathsf{B}$">
 (<A NAME="tex2html17"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:MakeCoefficients"><TT>TMultiDimFit::MakeCoefficients</TT></A>). The errors in
 the coefficients is calculated by inverting the curvature matrix
 of the non-orthogonal functions <IMG
 WIDTH="23" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img100.gif"
 ALT="$ f_{lj}$"> [<A
 HREF="TMultiFimFit.html#bevington">1</A>]
 (<A NAME="tex2html18"
 HREF="

 ./TMultiDimFit.html#TMultiDimFit:MakeCoefficientErrors"><TT>TMultiDimFit::MakeCoefficientErrors</TT></A>).

 <P>

 <H2><A NAME="SECTION00035000000000000000"></A>
 <A NAME="sec:considerations"></A><BR>
 Considerations
 </H2>

 <P>
 It's important to realize that the training sample should be
 representive of the problem at hand, in particular along the borders
 of the region of interest. This is because the algorithm presented
 here, is a <I>interpolation</I>, rahter then a <I>extrapolation</I>
 [<A
 HREF="TMultiFimFit.html#wind72">5</A>].

 <P>
 Also, the independent variables <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img101.gif"
 ALT="$ x_{i}$"> need to be linear
 independent, since the procedure will perform poorly if they are not
 [<A
 HREF="TMultiFimFit.html#wind72">5</A>]. One can find an linear transformation from ones
 original variables <IMG
 WIDTH="16" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img102.gif"
 ALT="$ \xi_{i}$"> to a set of linear independent variables
 <IMG
 WIDTH="18" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img101.gif"
 ALT="$ x_{i}$">, using a <I>Principal Components Analysis</I>
 <A NAME="tex2html19"
 HREF="./TPrincipal.html">(see <TT>TPrincipal</TT>)</A>, and
 then use the transformed variable as input to this class [<A
 HREF="TMultiFimFit.html#wind72">5</A>]
 [<A
 HREF="TMultiFimFit.html#wind81">6</A>].

 <P>
 H. Wind also outlines a method for parameterising a multidimensional
 dependence over a multidimensional set of variables. An example
 of the method from [<A
 HREF="TMultiFimFit.html#wind72">5</A>], is a follows (please refer to
 [<A
 HREF="TMultiFimFit.html#wind72">5</A>] for a full discussion):

 <P>

 <OL>
 <LI>Define <!-- MATH
 $\mathbf{P} = (P_1, \ldots, P_5)$
 -->
 <IMG
 WIDTH="123" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img103.gif"
 ALT="$ \mathbf{P} = (P_1, \ldots, P_5)$"> are the 5 dependent
 quantities that define a track.
 </LI>
 <LI>Compute, for <IMG
 WIDTH="21" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img10.gif"
 ALT="$ M$"> different values of <!-- MATH
 $\mathbf{P}$
 -->
 <IMG
 WIDTH="17" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img104.gif"
 ALT="$ \mathbf{P}$">, the tracks
 through the magnetic field, and determine the corresponding
 <!-- MATH
 $\mathbf{x} = (x_1, \ldots, x_N)$
 -->
 <IMG
 WIDTH="123" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img105.gif"
 ALT="$ \mathbf{x} = (x_1, \ldots, x_N)$">.
 </LI>
 <LI>Use the simulated observations to determine, with a simple
 approximation, the values of <!-- MATH
 $\mathbf{P}_j$
 -->
 <IMG
 WIDTH="23" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img106.gif"
 ALT="$ \mathbf{P}_j$">. We call these values
 <!-- MATH
 $\mathbf{P}^\prime_j, j = 1, \ldots, M$
 -->
 <IMG
 WIDTH="122" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img107.gif"
 ALT="$ \mathbf{P}^\prime_j, j = 1, \ldots, M$">.
 </LI>
 <LI>Determine from <!-- MATH
 $\mathbf{x}$
 -->
 <IMG
 WIDTH="14" HEIGHT="13" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img9.gif"
 ALT="$ \mathbf{x}$"> a set of at least five relevant
 coordinates <!-- MATH
 $\mathbf{x}^\prime$
 -->
 <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img108.gif"
 ALT="$ \mathbf{x}^\prime$">, using contrains, <I>or
 alternative:</I>
 </LI>
 <LI>Perform a Principal Component Analysis (using
 <A NAME="tex2html20"
 HREF="./TPrincipal.html"><TT>TPrincipal</TT></A>), and use

 to get a linear transformation
 <!-- MATH
 $\mathbf{x} \rightarrow \mathbf{x}^\prime$
 -->
 <IMG
 WIDTH="53" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img109.gif"
 ALT="$ \mathbf{x} \rightarrow \mathbf{x}^\prime$">, so that
 <!-- MATH
 $\mathbf{x}^\prime$
 -->
 <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img108.gif"
 ALT="$ \mathbf{x}^\prime$"> are constrained and linear independent.
 </LI>
 <LI>Perform a Principal Component Analysis on
 <!-- MATH
 $Q_i = P_i / P^\prime_i\, i = 1, \ldots, 5$
 -->
 <IMG
 WIDTH="210" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img110.gif"
 ALT="$ Q_i = P_i / P^prime_i  i = 1, \ldots, 5$">, to get linear
 indenpendent (among themselves, but not independent of
 <!-- MATH
 $\mathbf{x}$
 -->
 <IMG
 WIDTH="14" HEIGHT="13" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img9.gif"
 ALT="$ \mathbf{x}$">) quantities <!-- MATH
 $\mathbf{Q}^\prime$
 -->
 <IMG
 WIDTH="22" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img111.gif"
 ALT="$ \mathbf{Q}^\prime$">
 </LI>
 <LI>For each component <!-- MATH
 $Q^\prime_i$
 -->
 <IMG
 WIDTH="22" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img112.gif"
 ALT="$ Q^\prime_i$"> make a mutlidimensional fit,
 using <!-- MATH
 $\mathbf{x}^\prime$
 -->
 <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img108.gif"
 ALT="$ \mathbf{x}^\prime$"> as the variables, thus determing a set of
 coefficents <!-- MATH
 $\mathbf{c}_i$
 -->
 <IMG
 WIDTH="17" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img113.gif"
 ALT="$ \mathbf{c}_i$">.
 </LI>
 </OL>

 <P>
 To process data, using this parameterisation, do

 <OL>
 <LI>Test wether the observation <!-- MATH
 $\mathbf{x}$
 -->
 <IMG
 WIDTH="14" HEIGHT="13" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img9.gif"
 ALT="$ \mathbf{x}$"> within the domain of
 the parameterization, using the result from the Principal Component
 Analysis.
 </LI>
 <LI>Determine <!-- MATH
 $\mathbf{P}^\prime$
 -->
 <IMG
 WIDTH="21" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img114.gif"
 ALT="$ \mathbf{P}^\prime$"> as before.
 </LI>
 <LI>Detetmine <!-- MATH
 $\mathbf{x}^\prime$
 -->
 <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img108.gif"
 ALT="$ \mathbf{x}^\prime$"> as before.
 </LI>
 <LI>Use the result of the fit to determind <!-- MATH
 $\mathbf{Q}^\prime$
 -->
 <IMG
 WIDTH="22" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img111.gif"
 ALT="$ \mathbf{Q}^\prime$">.
 </LI>
 <LI>Transform back to <!-- MATH
 $\mathbf{P}$
 -->
 <IMG
 WIDTH="17" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img104.gif"
 ALT="$ \mathbf{P}$"> from <!-- MATH
 $\mathbf{Q}^\prime$
 -->
 <IMG
 WIDTH="22" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img111.gif"
 ALT="$ \mathbf{Q}^\prime$">, using
 the result from the Principal Component Analysis.
 </LI>
 </OL>

 <P>

 <H2><A NAME="SECTION00036000000000000000"></A>
 <A NAME="sec:testing"></A><BR>
 Testing the parameterization
 </H2>

 <P>
 The class also provides functionality for testing the, over the
 training sample, found parameterization
 (<A NAME="tex2html21"
 HREF="
 ./TMultiDimFit.html#TMultiDimFit:Fit"><TT>TMultiDimFit::Fit</TT></A>). This is done by passing
 the class a test sample of <IMG
 WIDTH="25" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img115.gif"
 ALT="$ M_t$"> tuples of the form <!-- MATH
 $(\mathbf{x}_{t,j},
 D_{t,j}, E_{t,j})$
 -->
 <IMG
 WIDTH="111" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img116.gif"
 ALT="$ (\mathbf{x}_{t,j},
 D_{t,j}, E_{t,j})$">, where <!-- MATH
 $\mathbf{x}_{t,j}$
 -->
 <IMG
 WIDTH="29" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img117.gif"
 ALT="$ \mathbf{x}_{t,j}$"> are the independent
 variables, <IMG
 WIDTH="33" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img118.gif"
 ALT="$ D_{t,j}$"> the known, dependent quantity, and <IMG
 WIDTH="31" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img119.gif"
 ALT="$ E_{t,j}$"> is
 the square error in <IMG
 WIDTH="33" HEIGHT="29" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img118.gif"
 ALT="$ D_{t,j}$">
 (<A NAME="tex2html22"
 HREF="

 ./TMultiDimFit.html#TMultiDimFit:AddTestRow"><TT>TMultiDimFit::AddTestRow</TT></A>).

 <P>
 The parameterization is then evaluated at every <!-- MATH
 $\mathbf{x}_t$
 -->
 <IMG
 WIDTH="19" HEIGHT="28" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img120.gif"
 ALT="$ \mathbf{x}_t$"> in the
 test sample, and
 <!-- MATH
 \begin{displaymath}
 S_t \equiv \sum_{j=1}^{M_t} \left(D_{t,j} -
 D_p\left(\mathbf{x}_{t,j}\right)\right)^2
 \end{displaymath}
 -->
 <P></P><DIV ALIGN="CENTER">
 <IMG
 WIDTH="194" HEIGHT="66" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img121.gif"
 ALT="$\displaystyle S_t \equiv \sum_{j=1}^{M_t} \left(D_{t,j} -
 D_p\left(\mathbf{x}_{t,j}\right)\right)^2
 $">
 </DIV><P></P>
 is evaluated. The relative error over the test sample
 <!-- MATH
 \begin{displaymath}
 R_t = \frac{S_t}{\sum_{j=1}^{M_t} D_{t,j}^2}
 \end{displaymath}
 -->
 <P></P><DIV ALIGN="CENTER">
 <IMG
 WIDTH="118" HEIGHT="51" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img122.gif"
 ALT="$\displaystyle R_t = \frac{S_t}{\sum_{j=1}^{M_t} D_{t,j}^2}
 $">
 </DIV><P></P>
 should not be to low or high compared to <IMG
 WIDTH="16" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/multidimfit_img123.gif"
 ALT="$ R$"> from the training
 sample. Also, multiple correlation coefficient from both samples should
 be fairly close, otherwise one of the samples is not representive of
 the problem. A large difference in the reduced <IMG
 WIDTH="21" HEIGHT="33" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/multidimfit_img124.gif"
 ALT="$ \chi^2$"> over the two
 samples indicate an over fit, and the maximum number of terms in the
 parameterisation should be reduced.

 <P>
 It's possible to use <A NAME="tex2html23"
 HREF="./TMinuit.html"><I>Minuit</I></A>
 [<A
 HREF="TMultiFimFit.html#minuit">4</A>] to further improve the fit, using the test sample.

 <P>
 <DIV ALIGN="RIGHT">
 Christian Holm
 <BR>  November 2000, NBI

 </DIV>

 <P>

 <H2><A NAME="SECTION00040000000000000000">
 Bibliography</A>
 </H2><DL COMPACT><DD><P></P><DT><A NAME="bevington">1</A>
 <DD>
 Philip&nbsp;R. Bevington and D.&nbsp;Keith Robinson.
 <BR><EM>Data Reduction and Error Analysis for the Physical Sciences</EM>.
 <BR>McGraw-Hill, 2 edition, 1992.

 <P></P><DT><A NAME="mudifi">2</A>
 <DD>
 Ren&#233; Brun et&nbsp;al.
 <BR>Mudifi.
 <BR>Long writeup DD/75-23, CERN, 1980.

 <P></P><DT><A NAME="golub">3</A>
 <DD>
 Gene&nbsp;H. Golub and Charles&nbsp;F. van Loan.
 <BR><EM>Matrix Computations</EM>.
 <BR>John Hopkins Univeristy Press, Baltimore, 3 edition, 1996.

 <P></P><DT><A NAME="minuit">4</A>
 <DD>
 F.&nbsp;James.
 <BR>Minuit.
 <BR>Long writeup D506, CERN, 1998.

 <P></P><DT><A NAME="wind72">5</A>
 <DD>
 H.&nbsp;Wind.
 <BR>Function parameterization.
 <BR>In <EM>Proceedings of the 1972 CERN Computing and Data Processing
 School</EM>, volume 72-21 of <EM>Yellow report</EM>. CERN, 1972.

 <P></P><DT><A NAME="wind81">6</A>
 <DD>
 H.&nbsp;Wind.
 <BR>1. principal component analysis, 2. pattern recognition for track
 finding, 3. interpolation and functional representation.
 <BR>Yellow report EP/81-12, CERN, 1981.
 </DL>
 <pre>
 */
//End_Html
//

#include "Riostream.h"
#include "TMultiDimFit.h"
#include "TMath.h"
#include "TH1.h"
#include "TH2.h"
#include "TROOT.h"
#include "TBrowser.h"
#include "TDecompChol.h"

#define RADDEG (180. / TMath::Pi())
#define DEGRAD (TMath::Pi() / 180.)
#define HIST_XORIG     0
#define HIST_DORIG     1
#define HIST_XNORM     2
#define HIST_DSHIF     3
#define HIST_RX        4
#define HIST_RD        5
#define HIST_RTRAI     6
#define HIST_RTEST     7
#define PARAM_MAXSTUDY 1
#define PARAM_SEVERAL  2
#define PARAM_RELERR   3
#define PARAM_MAXTERMS 4


//____________________________________________________________________
static void mdfHelper(int&, double*, double&, double*, int);

//____________________________________________________________________
ClassImp(TMultiDimFit);

//____________________________________________________________________
// Static instance. Used with mdfHelper and TMinuit
TMultiDimFit* TMultiDimFit::fgInstance = 0;


//____________________________________________________________________
TMultiDimFit::TMultiDimFit()
{
   // Empty CTOR. Do not use
   fMeanQuantity           = 0;
   fMaxQuantity            = 0;
   fMinQuantity            = 0;
   fSumSqQuantity          = 0;
   fSumSqAvgQuantity       = 0;

   fNVariables             = 0;
   fSampleSize             = 0;
   fTestSampleSize         = 0;

   fMinAngle               = 1;
   fMaxAngle               = 0;
   fMaxTerms               = 0;
   fMinRelativeError       = 0;
   fMaxPowers              = 0;
   fPowerLimit             = 0;

   fMaxFunctions           = 0;
   fFunctionCodes          = 0;
   fMaxStudy               = 0;
   fMaxFuncNV              = 0;

   fMaxPowersFinal         = 0;
   fPowers                 = 0;
   fPowerIndex             = 0;

   fMaxResidual            = 0;
   fMinResidual            = 0;
   fMaxResidualRow         = 0;
   fMinResidualRow         = 0;
   fSumSqResidual          = 0;

   fNCoefficients          = 0;
   fRMS                    = 0;
   fChi2                   = 0;
   fParameterisationCode   = 0;

   fError                  = 0;
   fTestError              = 0;
   fPrecision              = 0;
   fTestPrecision          = 0;
   fCorrelationCoeff       = 0;
   fTestCorrelationCoeff   = 0;

   fHistograms             = 0;
   fHistogramMask          = 0;
   fBinVarX                = 100;
   fBinVarY                = 100;

   fFitter                 = 0;
   fPolyType               = kMonomials;
   fShowCorrelation        = kFALSE;
   fIsUserFunction         = kFALSE;
   fIsVerbose              = kFALSE;

}


//____________________________________________________________________
TMultiDimFit::TMultiDimFit(Int_t dimension,
                           EMDFPolyType type,
                           Option_t *option)
: TNamed("multidimfit","Multi-dimensional fit object"),
fQuantity(dimension),
fSqError(dimension),
fVariables(dimension*100),
fMeanVariables(dimension),
fMaxVariables(dimension),
fMinVariables(dimension)
{
   // Constructor
   // Second argument is the type of polynomials to use in
   // parameterisation, one of:
   //      TMultiDimFit::kMonomials
   //      TMultiDimFit::kChebyshev
   //      TMultiDimFit::kLegendre
   //
   // Options:
   //   K      Compute (k)correlation matrix
   //   V      Be verbose
   //
   // Default is no options.
   //

   fgInstance = this;

   fMeanQuantity           = 0;
   fMaxQuantity            = 0;
   fMinQuantity            = 0;
   fSumSqQuantity          = 0;
   fSumSqAvgQuantity       = 0;

   fNVariables             = dimension;
   fSampleSize             = 0;
   fTestSampleSize         = 0;

   fMinAngle               = 1;
   fMaxAngle               = 0;
   fMaxTerms               = 0;
   fMinRelativeError       = 0.01;
   fMaxPowers              = new Int_t[dimension];
   fPowerLimit             = 1;

   fMaxFunctions           = 0;
   fFunctionCodes          = 0;
   fMaxStudy               = 0;
   fMaxFuncNV              = 0;

   fMaxPowersFinal         = new Int_t[dimension];
   fPowers                 = 0;
   fPowerIndex             = 0;

   fMaxResidual            = 0;
   fMinResidual            = 0;
   fMaxResidualRow         = 0;
   fMinResidualRow         = 0;
   fSumSqResidual          = 0;

   fNCoefficients          = 0;
   fRMS                    = 0;
   fChi2                   = 0;
   fParameterisationCode   = 0;

   fError                  = 0;
   fTestError              = 0;
   fPrecision              = 0;
   fTestPrecision          = 0;
   fCorrelationCoeff       = 0;
   fTestCorrelationCoeff   = 0;

   fHistograms             = 0;
   fHistogramMask          = 0;
   fBinVarX                = 100;
   fBinVarY                = 100;

   fFitter                 = 0;
   fPolyType               = type;
   fShowCorrelation        = kFALSE;
   fIsUserFunction         = kFALSE;
   fIsVerbose              = kFALSE;
   TString opt             = option;
   opt.ToLower();

   if (opt.Contains("k")) fShowCorrelation = kTRUE;
   if (opt.Contains("v")) fIsVerbose       = kTRUE;
}


//____________________________________________________________________
TMultiDimFit::~TMultiDimFit()
{
   // Destructor
   delete [] fPowers;
   delete [] fMaxPowers;
   delete [] fMaxPowersFinal;
   delete [] fPowerIndex;
   delete [] fFunctionCodes;
   if (fHistograms) fHistograms->Clear("nodelete");
   delete fHistograms;
}


//____________________________________________________________________
void TMultiDimFit::AddRow(const Double_t *x, Double_t D, Double_t E)
{
   // Add a row consisting of fNVariables independent variables, the
   // known, dependent quantity, and optionally, the square error in
   // the dependent quantity, to the training sample to be used for the
   // parameterization.
   // The mean of the variables and quantity is calculated on the fly,
   // as outlined in TPrincipal::AddRow.
   // This sample should be representive of the problem at hand.
   // Please note, that if no error is given Poisson statistics is
   // assumed and the square error is set to the value of dependent
   // quantity.  See also the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   if (!x)
      return;

   if (++fSampleSize == 1) {
      fMeanQuantity  = D;
      fMaxQuantity   = D;
      fMinQuantity   = D;
      fSumSqQuantity = D * D;// G.Q. erratum on August 15th, 2008
   }
   else {
      fMeanQuantity  *= 1 - 1./Double_t(fSampleSize);
      fMeanQuantity  += D / Double_t(fSampleSize);
      fSumSqQuantity += D * D;

      if (D >= fMaxQuantity) fMaxQuantity = D;
      if (D <= fMinQuantity) fMinQuantity = D;
   }


   // If the vector isn't big enough to hold the new data, then
   // expand the vector by half it's size.
   Int_t size = fQuantity.GetNrows();
   if (fSampleSize > size) {
      fQuantity.ResizeTo(size + size/2);
      fSqError.ResizeTo(size + size/2);
   }

   // Store the value
   fQuantity(fSampleSize-1) = D;
   fSqError(fSampleSize-1) = (E == 0 ? D : E);

   // Store data point in internal vector
   // If the vector isn't big enough to hold the new data, then
   // expand the vector by half it's size
   size = fVariables.GetNrows();
   if (fSampleSize * fNVariables > size)
      fVariables.ResizeTo(size + size/2);


   // Increment the data point counter
   Int_t i,j;
   for (i = 0; i < fNVariables; i++) {
      if (fSampleSize == 1) {
         fMeanVariables(i) = x[i];
         fMaxVariables(i)  = x[i];
         fMinVariables(i)  = x[i];
      }
      else {
         fMeanVariables(i) *= 1 - 1./Double_t(fSampleSize);
         fMeanVariables(i) += x[i] / Double_t(fSampleSize);

         // Update the maximum value for this component
         if (x[i] >= fMaxVariables(i)) fMaxVariables(i)  = x[i];

         // Update the minimum value for this component
         if (x[i] <= fMinVariables(i)) fMinVariables(i)  = x[i];

      }

      // Store the data.
      j = (fSampleSize-1) * fNVariables + i;
      fVariables(j) = x[i];
   }
}


//____________________________________________________________________
void TMultiDimFit::AddTestRow(const Double_t *x, Double_t D, Double_t E)
{
   // Add a row consisting of fNVariables independent variables, the
   // known, dependent quantity, and optionally, the square error in
   // the dependent quantity, to the test sample to be used for the
   // test of the parameterization.
   // This sample needn't be representive of the problem at hand.
   // Please note, that if no error is given Poisson statistics is
   // assumed and the square error is set to the value of dependent
   // quantity.  See also the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   if (fTestSampleSize++ == 0) {
      fTestQuantity.ResizeTo(fNVariables);
      fTestSqError.ResizeTo(fNVariables);
      fTestVariables.ResizeTo(fNVariables * 100);
   }

   // If the vector isn't big enough to hold the new data, then
   // expand the vector by half it's size.
   Int_t size = fTestQuantity.GetNrows();
   if (fTestSampleSize > size) {
      fTestQuantity.ResizeTo(size + size/2);
      fTestSqError.ResizeTo(size + size/2);
   }

   // Store the value
   fTestQuantity(fTestSampleSize-1) = D;
   fTestSqError(fTestSampleSize-1) = (E == 0 ? D : E);

   // Store data point in internal vector
   // If the vector isn't big enough to hold the new data, then
   // expand the vector by half it's size
   size = fTestVariables.GetNrows();
   if (fTestSampleSize * fNVariables > size)
      fTestVariables.ResizeTo(size + size/2);


   // Increment the data point counter
   Int_t i,j;
   for (i = 0; i < fNVariables; i++) {
      j = fNVariables * (fTestSampleSize - 1) + i;
      fTestVariables(j) = x[i];

      if (x[i] > fMaxVariables(i))
         Warning("AddTestRow", "variable %d (row: %d) too large: %f > %f",
                 i, fTestSampleSize, x[i], fMaxVariables(i));
      if (x[i] < fMinVariables(i))
         Warning("AddTestRow", "variable %d (row: %d) too small: %f < %f",
                 i, fTestSampleSize, x[i], fMinVariables(i));
   }
}


//____________________________________________________________________
void TMultiDimFit::Browse(TBrowser* b)
{
   // Browse the TMultiDimFit object in the TBrowser.
   if (fHistograms) {
      TIter next(fHistograms);
      TH1* h = 0;
      while ((h = (TH1*)next()))
         b->Add(h,h->GetName());
   }
   if (fVariables.IsValid())
      b->Add(&fVariables, "Variables (Training)");
   if (fQuantity.IsValid())
      b->Add(&fQuantity, "Quantity (Training)");
   if (fSqError.IsValid())
      b->Add(&fSqError, "Error (Training)");
   if (fMeanVariables.IsValid())
      b->Add(&fMeanVariables, "Mean of Variables (Training)");
   if (fMaxVariables.IsValid())
      b->Add(&fMaxVariables, "Mean of Variables (Training)");
   if (fMinVariables.IsValid())
      b->Add(&fMinVariables, "Min of Variables (Training)");
   if (fTestVariables.IsValid())
      b->Add(&fTestVariables, "Variables (Test)");
   if (fTestQuantity.IsValid())
      b->Add(&fTestQuantity, "Quantity (Test)");
   if (fTestSqError.IsValid())
      b->Add(&fTestSqError, "Error (Test)");
   if (fFunctions.IsValid())
      b->Add(&fFunctions, "Functions");
   if(fCoefficients.IsValid())
      b->Add(&fCoefficients,"Coefficients");
   if(fCoefficientsRMS.IsValid())
      b->Add(&fCoefficientsRMS,"Coefficients Errors");
   if (fOrthFunctions.IsValid())
      b->Add(&fOrthFunctions, "Orthogonal Functions");
   if (fOrthFunctionNorms.IsValid())
      b->Add(&fOrthFunctionNorms, "Orthogonal Functions Norms");
   if (fResiduals.IsValid())
      b->Add(&fResiduals, "Residuals");
   if(fOrthCoefficients.IsValid())
      b->Add(&fOrthCoefficients,"Orthogonal Coefficients");
   if (fOrthCurvatureMatrix.IsValid())
      b->Add(&fOrthCurvatureMatrix,"Orthogonal curvature matrix");
   if(fCorrelationMatrix.IsValid())
      b->Add(&fCorrelationMatrix,"Correlation Matrix");
   if (fFitter)
      b->Add(fFitter, fFitter->GetName());
}


//____________________________________________________________________
void TMultiDimFit::Clear(Option_t *option)
{
   // Clear internal structures and variables
   Int_t i, j, n = fNVariables, m = fMaxFunctions;

   // Training sample, dependent quantity
   fQuantity.Zero();
   fSqError.Zero();
   fMeanQuantity                 = 0;
   fMaxQuantity                  = 0;
   fMinQuantity                  = 0;
   fSumSqQuantity                = 0;
   fSumSqAvgQuantity             = 0;

   // Training sample, independent variables
   fVariables.Zero();
   fNVariables                   = 0;
   fSampleSize                   = 0;
   fMeanVariables.Zero();
   fMaxVariables.Zero();
   fMinVariables.Zero();

   // Test sample
   fTestQuantity.Zero();
   fTestSqError.Zero();
   fTestVariables.Zero();
   fTestSampleSize               = 0;

   // Functions
   fFunctions.Zero();
   //for (i = 0; i < fMaxTerms; i++)  fPowerIndex[i]    = 0;
   //for (i = 0; i < fMaxTerms; i++)  fFunctionCodes[i] = 0;
   fMaxFunctions                 = 0;
   fMaxStudy                     = 0;
   fOrthFunctions.Zero();
   fOrthFunctionNorms.Zero();

   // Control parameters
   fMinRelativeError             = 0;
   fMinAngle                     = 0;
   fMaxAngle                     = 0;
   fMaxTerms                     = 0;

   // Powers
   for (i = 0; i < n; i++) {
      fMaxPowers[i]               = 0;
      fMaxPowersFinal[i]          = 0;
      for (j = 0; j < m; j++)
         fPowers[i * n + j]        = 0;
   }
   fPowerLimit                   = 0;

   // Residuals
   fMaxResidual                  = 0;
   fMinResidual                  = 0;
   fMaxResidualRow               = 0;
   fMinResidualRow               = 0;
   fSumSqResidual                = 0;

   // Fit
   fNCoefficients                = 0;
   fOrthCoefficients             = 0;
   fOrthCurvatureMatrix          = 0;
   fRMS                          = 0;
   fCorrelationMatrix.Zero();
   fError                        = 0;
   fTestError                    = 0;
   fPrecision                    = 0;
   fTestPrecision                = 0;

   // Coefficients
   fCoefficients.Zero();
   fCoefficientsRMS.Zero();
   fResiduals.Zero();
   fHistograms->Clear(option);

   // Options
   fPolyType                     = kMonomials;
   fShowCorrelation              = kFALSE;
   fIsUserFunction               = kFALSE;
}


//____________________________________________________________________
Double_t TMultiDimFit::Eval(const Double_t *x, const Double_t* coeff) const
{
   // Evaluate parameterization at point x. Optional argument coeff is
   // a vector of coefficients for the parameterisation, fNCoefficients
   // elements long.
   Double_t returnValue = fMeanQuantity;
   Double_t term        = 0;
   Int_t    i, j;

   for (i = 0; i < fNCoefficients; i++) {
      // Evaluate the ith term in the expansion
      term = (coeff ? coeff[i] : fCoefficients(i));
      for (j = 0; j < fNVariables; j++) {
         // Evaluate the factor (polynomial) in the j-th variable.
         Int_t    p  =  fPowers[fPowerIndex[i] * fNVariables + j];
         Double_t y  =  1 + 2. / (fMaxVariables(j) - fMinVariables(j))
         * (x[j] - fMaxVariables(j));
         term        *= EvalFactor(p,y);
      }
      // Add this term to the final result
      returnValue += term;
   }
   return returnValue;
}


//____________________________________________________________________
Double_t TMultiDimFit::EvalError(const Double_t *x, const Double_t* coeff) const
{
   // Evaluate parameterization error at point x. Optional argument coeff is
   // a vector of coefficients for the parameterisation, fNCoefficients
   // elements long.
   Double_t returnValue = 0;
   Double_t term        = 0;
   Int_t    i, j;

   for (i = 0; i < fNCoefficients; i++) {
      //     std::cout << "Error coef " << i << " -> " << fCoefficientsRMS(i) << std::endl;
   }
   for (i = 0; i < fNCoefficients; i++) {
      // Evaluate the ith term in the expansion
      term = (coeff ? coeff[i] : fCoefficientsRMS(i));
      for (j = 0; j < fNVariables; j++) {
         // Evaluate the factor (polynomial) in the j-th variable.
         Int_t    p  =  fPowers[fPowerIndex[i] * fNVariables + j];
         Double_t y  =  1 + 2. / (fMaxVariables(j) - fMinVariables(j))
         * (x[j] - fMaxVariables(j));
         term        *= EvalFactor(p,y);
         //   std::cout << "i,j " << i << ", " << j << "  "  << p << "  " << y << "  " << EvalFactor(p,y) << "  " << term << std::endl;
      }
      // Add this term to the final result
      returnValue += term*term;
      //      std::cout << " i = " << i << " value = " << returnValue << std::endl;
   }
   returnValue = sqrt(returnValue);
   return returnValue;
}


//____________________________________________________________________
Double_t TMultiDimFit::EvalControl(const Int_t *iv) const
{
   // PRIVATE METHOD:
   // Calculate the control parameter from the passed powers
   Double_t s = 0;
   Double_t epsilon = 1e-6; // a small number
   for (Int_t i = 0; i < fNVariables; i++) {
      if (fMaxPowers[i] != 1)
         s += (epsilon + iv[i] - 1) / (epsilon + fMaxPowers[i] - 1);
   }
   return s;
}

//____________________________________________________________________
Double_t TMultiDimFit::EvalFactor(Int_t p, Double_t x) const
{
   // PRIVATE METHOD:
   // Evaluate function with power p at variable value x
   Int_t    i   = 0;
   Double_t p1  = 1;
   Double_t p2  = 0;
   Double_t p3  = 0;
   Double_t r   = 0;

   switch(p) {
      case 1:
         r = 1;
         break;
      case 2:
         r =  x;
         break;
      default:
         p2 = x;
         for (i = 3; i <= p; i++) {
            p3 = p2 * x;
            if (fPolyType == kLegendre)
               p3 = ((2 * i - 3) * p2 * x - (i - 2) * p1) / (i - 1);
            else if (fPolyType == kChebyshev)
               p3 = 2 * x * p2 - p1;
            p1 = p2;
            p2 = p3;
         }
         r = p3;
   }

   return r;
}


//____________________________________________________________________
void TMultiDimFit::FindParameterization(Option_t *)
{
   // Find the parameterization
   //
   // Options:
   //     None so far
   //
   // For detailed description of what this entails, please refer to the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   MakeNormalized();
   MakeCandidates();
   MakeParameterization();
   MakeCoefficients();
   MakeCoefficientErrors();
   MakeCorrelation();
}

//____________________________________________________________________
void TMultiDimFit::Fit(Option_t *option)
{
   // Try to fit the found parameterisation to the test sample.
   //
   // Options
   //     M     use Minuit to improve coefficients
   //
   // Also, refer to
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   Int_t i, j;
   Double_t*      x    = new Double_t[fNVariables];
   Double_t  sumSqD    = 0;
   Double_t    sumD    = 0;
   Double_t  sumSqR    = 0;
   Double_t    sumR    = 0;

   // Calculate the residuals over the test sample
   for (i = 0; i < fTestSampleSize; i++) {
      for (j = 0; j < fNVariables; j++)
         x[j] = fTestVariables(i * fNVariables + j);
      Double_t res =  fTestQuantity(i) - Eval(x);
      sumD         += fTestQuantity(i);
      sumSqD       += fTestQuantity(i) * fTestQuantity(i);
      sumR         += res;
      sumSqR       += res * res;
      if (TESTBIT(fHistogramMask,HIST_RTEST))
         ((TH1D*)fHistograms->FindObject("res_test"))->Fill(res);
   }
   Double_t dAvg         = sumSqD - (sumD * sumD) / fTestSampleSize;
   Double_t rAvg         = sumSqR - (sumR * sumR) / fTestSampleSize;
   fTestCorrelationCoeff = (dAvg - rAvg) / dAvg;
   fTestError            = sumSqR;
   fTestPrecision        = sumSqR / sumSqD;

   TString opt(option);
   opt.ToLower();

   if (!opt.Contains("m"))
      MakeChi2();

   if (fNCoefficients * 50 > fTestSampleSize)
      Warning("Fit", "test sample is very small");

   if (!opt.Contains("m")) {
      Error("Fit", "invalid option");
      delete [] x;
      return;
   }

   fFitter = TVirtualFitter::Fitter(0,fNCoefficients);
   if (!fFitter) {
      Error("Fit", "Vannot create Fitter");
      delete [] x;
      return;
   }
   fFitter->SetFCN(mdfHelper);

   const Int_t  maxArgs = 16;
   Int_t           args = 1;
   Double_t*   arglist  = new Double_t[maxArgs];
   arglist[0]           = -1;
   fFitter->ExecuteCommand("SET PRINT",arglist,args);

   for (i = 0; i < fNCoefficients; i++) {
      Double_t startVal = fCoefficients(i);
      Double_t startErr = fCoefficientsRMS(i);
      fFitter->SetParameter(i, Form("coeff%02d",i),
                            startVal, startErr, 0, 0);
   }

   // arglist[0]           = 0;
   args                 = 1;
   // fFitter->ExecuteCommand("SET PRINT",arglist,args);
   fFitter->ExecuteCommand("MIGRAD",arglist,args);

   for (i = 0; i < fNCoefficients; i++) {
      Double_t val = 0, err = 0, low = 0, high = 0;
      fFitter->GetParameter(i, Form("coeff%02d",i),
                            val, err, low, high);
      fCoefficients(i)    = val;
      fCoefficientsRMS(i) = err;
   }
   delete [] x;
}

//____________________________________________________________________
TMultiDimFit* TMultiDimFit::Instance()
{
   // Return the static instance.
   return fgInstance;
}

//____________________________________________________________________
void TMultiDimFit::MakeCandidates()
{
   // PRIVATE METHOD:
   // Create list of candidate functions for the parameterisation. See
   // also
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   Int_t i = 0;
   Int_t j = 0;
   Int_t k = 0;

   // The temporary array to store the powers in. We don't need to
   // initialize this array however.
   fMaxFuncNV = fNVariables * fMaxFunctions;
   Int_t *powers = new Int_t[fMaxFuncNV];

   // store of `control variables'
   Double_t* control  = new Double_t[fMaxFunctions];

   // We've better initialize the variables
   Int_t *iv = new Int_t[fNVariables];
   for (i = 0; i < fNVariables; i++)
      iv[i] = 1;

   if (!fIsUserFunction) {

      // Number of funcs selected
      Int_t     numberFunctions = 0;

      // Absolute max number of functions
      Int_t maxNumberFunctions = 1;
      for (i = 0; i < fNVariables; i++)
         maxNumberFunctions *= fMaxPowers[i];

      while (kTRUE) {
         // Get the control value for this function
         Double_t s = EvalControl(iv);

         if (s <= fPowerLimit) {

            // Call over-loadable method Select, as to allow the user to
            // interfere with the selection of functions.
            if (Select(iv)) {
               numberFunctions++;

               // If we've reached the user defined limit of how many
               // functions we can consider, break out of the loop
               if (numberFunctions > fMaxFunctions)
                  break;

               // Store the control value, so we can sort array of powers
               // later on
               control[numberFunctions-1] = Int_t(1.0e+6*s);

               // Store the powers in powers array.
               for (i = 0; i < fNVariables; i++) {
                  j = (numberFunctions - 1) * fNVariables + i;
                  powers[j] = iv[i];
               }
            } // if (Select())
         } // if (s <= fPowerLimit)

         for (i = 0; i < fNVariables; i++)
            if (iv[i] < fMaxPowers[i])
               break;

         // If all variables have reached their maximum power, then we
         // break out of the loop
         if (i == fNVariables) {
            fMaxFunctions = numberFunctions;
            break;
         }

         // Next power in variable i
         if (i < fNVariables) iv[i]++;

         for (j = 0; j < i; j++)
            iv[j] = 1;
      } // while (kTRUE)
   }
   else {
      // In case the user gave an explicit function
      for (i = 0; i < fMaxFunctions; i++) {
         // Copy the powers to working arrays
         for (j = 0; j < fNVariables; j++) {
            powers[i * fNVariables + j] = fPowers[i * fNVariables + j];
            iv[j]                 = fPowers[i * fNVariables + j];
         }

         control[i] = Int_t(1.0e+6*EvalControl(iv));
      }
   }

   // Now we need to sort the powers according to least `control
   // variable'
   Int_t *order = new Int_t[fMaxFunctions];
   for (i = 0; i < fMaxFunctions; i++)
      order[i] = i;
   fMaxFuncNV = fMaxFunctions * fNVariables;
   fPowers = new Int_t[fMaxFuncNV];

   for (i = 0; i < fMaxFunctions; i++) {
      Double_t x = control[i];
      Int_t    l = order[i];
      k = i;

      for (j = i; j < fMaxFunctions; j++) {
         if (control[j] <= x) {
            x = control[j];
            l = order[j];
            k = j;
         }
      }

      if (k != i) {
         control[k] = control[i];
         control[i] = x;
         order[k]   = order[i];
         order[i]   = l;
      }
   }

   for (i = 0; i < fMaxFunctions; i++)
      for (j = 0; j < fNVariables; j++)
         fPowers[i * fNVariables + j] = powers[order[i] * fNVariables + j];

   delete [] control;
   delete [] powers;
   delete [] order;
   delete [] iv;
}


//____________________________________________________________________
Double_t TMultiDimFit::MakeChi2(const Double_t* coeff)
{
   // Calculate Chi square over either the test sample. The optional
   // argument coeff is a vector of coefficients to use in the
   // evaluation of the parameterisation. If coeff == 0, then the found
   // coefficients is used.
   // Used my MINUIT for fit (see TMultDimFit::Fit)
   fChi2 = 0;
   Int_t i, j;
   Double_t* x = new Double_t[fNVariables];
   for (i = 0; i < fTestSampleSize; i++) {
      // Get the stored point
      for (j = 0; j < fNVariables; j++)
         x[j] = fTestVariables(i * fNVariables + j);

      // Evaluate function. Scale to shifted values
      Double_t f = Eval(x,coeff);

      // Calculate contribution to Chic square
      fChi2 += 1. / TMath::Max(fTestSqError(i),1e-20)
      * (fTestQuantity(i) - f) * (fTestQuantity(i) - f);
   }

   // Clean up
   delete [] x;

   return fChi2;
}


//____________________________________________________________________
void TMultiDimFit::MakeCode(const char* filename, Option_t *option)
{
   // Generate the file <filename> with .C appended if argument doesn't
   // end in .cxx or .C. The contains the implementation of the
   // function:
   //
   //   Double_t <funcname>(Double_t *x)
   //
   // which does the same as TMultiDimFit::Eval. Please refer to this
   // method.
   //
   // Further, the static variables:
   //
   //     Int_t    gNVariables
   //     Int_t    gNCoefficients
   //     Double_t gDMean
   //     Double_t gXMean[]
   //     Double_t gXMin[]
   //     Double_t gXMax[]
   //     Double_t gCoefficient[]
   //     Int_t    gPower[]
   //
   // are initialized. The only ROOT header file needed is Rtypes.h
   //
   // See TMultiDimFit::MakeRealCode for a list of options


   TString outName(filename);
   if (!outName.EndsWith(".C") && !outName.EndsWith(".cxx"))
      outName += ".C";

   MakeRealCode(outName.Data(),"",option);
}



//____________________________________________________________________
void TMultiDimFit::MakeCoefficientErrors()
{
   // PRIVATE METHOD:
   // Compute the errors on the coefficients. For this to be done, the
   // curvature matrix of the non-orthogonal functions, is computed.
   Int_t    i = 0;
   Int_t    j = 0;
   Int_t    k = 0;
   TVectorD iF(fSampleSize);
   TVectorD jF(fSampleSize);
   fCoefficientsRMS.ResizeTo(fNCoefficients);

   TMatrixDSym curvatureMatrix(fNCoefficients);

   // Build the curvature matrix
   for (i = 0; i < fNCoefficients; i++) {
      iF = TMatrixDRow(fFunctions,i);
      for (j = 0; j <= i; j++) {
         jF = TMatrixDRow(fFunctions,j);
         for (k = 0; k < fSampleSize; k++)
            curvatureMatrix(i,j) +=
            1 / TMath::Max(fSqError(k), 1e-20) * iF(k) * jF(k);
         curvatureMatrix(j,i) = curvatureMatrix(i,j);
      }
   }

   // Calculate Chi Square
   fChi2 = 0;
   for (i = 0; i < fSampleSize; i++) {
      Double_t f = 0;
      for (j = 0; j < fNCoefficients; j++)
         f += fCoefficients(j) * fFunctions(j,i);
      fChi2 += 1. / TMath::Max(fSqError(i),1e-20) * (fQuantity(i) - f)
      * (fQuantity(i) - f);
   }

   // Invert the curvature matrix
   const TVectorD diag = TMatrixDDiag_const(curvatureMatrix);
   curvatureMatrix.NormByDiag(diag);

   TDecompChol chol(curvatureMatrix);
   if (!chol.Decompose())
      Error("MakeCoefficientErrors", "curvature matrix is singular");
   chol.Invert(curvatureMatrix);

   curvatureMatrix.NormByDiag(diag);

   for (i = 0; i < fNCoefficients; i++)
      fCoefficientsRMS(i) = TMath::Sqrt(curvatureMatrix(i,i));
}


//____________________________________________________________________
void TMultiDimFit::MakeCoefficients()
{
   // PRIVATE METHOD:
   // Invert the model matrix B, and compute final coefficients. For a
   // more thorough discussion of what this means, please refer to the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   //
   // First we invert the lower triangle matrix fOrthCurvatureMatrix
   // and store the inverted matrix in the upper triangle.

   Int_t i = 0, j = 0;
   Int_t col = 0, row = 0;

   // Invert the B matrix
   for (col = 1; col < fNCoefficients; col++) {
      for (row = col - 1; row > -1; row--) {
         fOrthCurvatureMatrix(row,col) = 0;
         for (i = row; i <= col ; i++)
            fOrthCurvatureMatrix(row,col) -=
            fOrthCurvatureMatrix(i,row)
            * fOrthCurvatureMatrix(i,col);
      }
   }

   // Compute the final coefficients
   fCoefficients.ResizeTo(fNCoefficients);

   for (i = 0; i < fNCoefficients; i++) {
      Double_t sum = 0;
      for (j = i; j < fNCoefficients; j++)
         sum += fOrthCurvatureMatrix(i,j) * fOrthCoefficients(j);
      fCoefficients(i) = sum;
   }

   // Compute the final residuals
   fResiduals.ResizeTo(fSampleSize);
   for (i = 0; i < fSampleSize; i++)
      fResiduals(i) = fQuantity(i);

   for (i = 0; i < fNCoefficients; i++)
      for (j = 0; j < fSampleSize; j++)
         fResiduals(j) -= fCoefficients(i) * fFunctions(i,j);

   // Compute the max and minimum, and squared sum of the evaluated
   // residuals
   fMinResidual = 10e10;
   fMaxResidual = -10e10;
   Double_t sqRes  = 0;
   for (i = 0; i < fSampleSize; i++){
      sqRes += fResiduals(i) * fResiduals(i);
      if (fResiduals(i) <= fMinResidual) {
         fMinResidual     = fResiduals(i);
         fMinResidualRow  = i;
      }
      if (fResiduals(i) >= fMaxResidual) {
         fMaxResidual     = fResiduals(i);
         fMaxResidualRow  = i;
      }
   }

   fCorrelationCoeff = fSumSqResidual / fSumSqAvgQuantity;
   fPrecision        = TMath::Sqrt(sqRes / fSumSqQuantity);

   // If we use histograms, fill some more
   if (TESTBIT(fHistogramMask,HIST_RD) ||
       TESTBIT(fHistogramMask,HIST_RTRAI) ||
       TESTBIT(fHistogramMask,HIST_RX)) {
      for (i = 0; i < fSampleSize; i++) {
         if (TESTBIT(fHistogramMask,HIST_RD))
            ((TH2D*)fHistograms->FindObject("res_d"))->Fill(fQuantity(i),
                                                            fResiduals(i));
         if (TESTBIT(fHistogramMask,HIST_RTRAI))
            ((TH1D*)fHistograms->FindObject("res_train"))->Fill(fResiduals(i));

         if (TESTBIT(fHistogramMask,HIST_RX))
            for (j = 0; j < fNVariables; j++)
               ((TH2D*)fHistograms->FindObject(Form("res_x_%d",j)))
               ->Fill(fVariables(i * fNVariables + j),fResiduals(i));
      }
   } // If histograms

}


//____________________________________________________________________
void TMultiDimFit::MakeCorrelation()
{
   // PRIVATE METHOD:
   // Compute the correlation matrix
   if (!fShowCorrelation)
      return;

   fCorrelationMatrix.ResizeTo(fNVariables,fNVariables+1);

   Double_t d2      = 0;
   Double_t ddotXi  = 0; // G.Q. needs to be reinitialized in the loop over i fNVariables
   Double_t xiNorm  = 0; // G.Q. needs to be reinitialized in the loop over i fNVariables
   Double_t xidotXj = 0; // G.Q. needs to be reinitialized in the loop over j fNVariables
   Double_t xjNorm  = 0; // G.Q. needs to be reinitialized in the loop over j fNVariables

   Int_t i, j, k, l, m;  // G.Q. added m variable
   for (i = 0; i < fSampleSize; i++)
      d2 += fQuantity(i) * fQuantity(i);

   for (i = 0; i < fNVariables; i++) {
      ddotXi = 0.; // G.Q. reinitialisation
      xiNorm = 0.; // G.Q. reinitialisation
      for (j = 0; j< fSampleSize; j++) {
         // Index of sample j of variable i
         k =  j * fNVariables + i;
         ddotXi += fQuantity(j) * (fVariables(k) - fMeanVariables(i));
         xiNorm += (fVariables(k) - fMeanVariables(i))
         * (fVariables(k) - fMeanVariables(i));
      }
      fCorrelationMatrix(i,0) = ddotXi / TMath::Sqrt(d2 * xiNorm);

      for (j = 0; j < i; j++) {
         xidotXj = 0.; // G.Q. reinitialisation
         xjNorm = 0.; // G.Q. reinitialisation
         for (k = 0; k < fSampleSize; k++) {
            // Index of sample j of variable i
            // l =  j * fNVariables + k;  // G.Q.
            l =  k * fNVariables + j; // G.Q.
            m =  k * fNVariables + i; // G.Q.
                                      // G.Q.        xidotXj += (fVariables(i) - fMeanVariables(i))
                                      // G.Q.          * (fVariables(l) - fMeanVariables(j));
            xidotXj += (fVariables(m) - fMeanVariables(i))
            * (fVariables(l) - fMeanVariables(j));  // G.Q. modified index for Xi
            xjNorm  += (fVariables(l) - fMeanVariables(j))
            * (fVariables(l) - fMeanVariables(j));
         }
         //fCorrelationMatrix(i+1,j) = xidotXj / TMath::Sqrt(xiNorm * xjNorm);
         fCorrelationMatrix(i,j+1) = xidotXj / TMath::Sqrt(xiNorm * xjNorm);
      }
   }
}



//____________________________________________________________________
Double_t TMultiDimFit::MakeGramSchmidt(Int_t function)
{
   // PRIVATE METHOD:
   // Make Gram-Schmidt orthogonalisation. The class description gives
   // a thorough account of this algorithm, as well as
   // references. Please refer to the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html


   // calculate w_i, that is, evaluate the current function at data
   // point i
   Double_t f2                        = 0;
   fOrthCoefficients(fNCoefficients)      = 0;
   fOrthFunctionNorms(fNCoefficients)  = 0;
   Int_t j        = 0;
   Int_t k        = 0;

   for (j = 0; j < fSampleSize; j++) {
      fFunctions(fNCoefficients, j) = 1;
      fOrthFunctions(fNCoefficients, j) = 0;
      // First, however, we need to calculate f_fNCoefficients
      for (k = 0; k < fNVariables; k++) {
         Int_t    p   =  fPowers[function * fNVariables + k];
         Double_t x   =  fVariables(j * fNVariables + k);
         fFunctions(fNCoefficients, j) *= EvalFactor(p,x);
      }

      // Calculate f dot f in f2
      f2 += fFunctions(fNCoefficients,j) *  fFunctions(fNCoefficients,j);
      // Assign to w_fNCoefficients f_fNCoefficients
      fOrthFunctions(fNCoefficients, j) = fFunctions(fNCoefficients, j);
   }

   // the first column of w is equal to f
   for (j = 0; j < fNCoefficients; j++) {
      Double_t fdw = 0;
      // Calculate (f_fNCoefficients dot w_j) / w_j^2
      for (k = 0; k < fSampleSize; k++) {
         fdw += fFunctions(fNCoefficients, k) * fOrthFunctions(j,k)
         / fOrthFunctionNorms(j);
      }

      fOrthCurvatureMatrix(fNCoefficients,j) = fdw;
      // and subtract it from the current value of w_ij
      for (k = 0; k < fSampleSize; k++)
         fOrthFunctions(fNCoefficients,k) -= fdw * fOrthFunctions(j,k);
   }

   for (j = 0; j < fSampleSize; j++) {
      // calculate squared length of w_fNCoefficients
      fOrthFunctionNorms(fNCoefficients) +=
      fOrthFunctions(fNCoefficients,j)
      * fOrthFunctions(fNCoefficients,j);

      // calculate D dot w_fNCoefficients in A
      fOrthCoefficients(fNCoefficients) += fQuantity(j)
      * fOrthFunctions(fNCoefficients, j);
   }

   // First test, but only if didn't user specify
   if (!fIsUserFunction)
      if (TMath::Sqrt(fOrthFunctionNorms(fNCoefficients) / (f2 + 1e-10))
          < TMath::Sin(fMinAngle*DEGRAD))
         return 0;

   // The result found by this code for the first residual is always
   // much less then the one found be MUDIFI. That's because it's
   // supposed to be. The cause is the improved precision of Double_t
   // over DOUBLE PRECISION!
   fOrthCurvatureMatrix(fNCoefficients,fNCoefficients) = 1;
   Double_t b = fOrthCoefficients(fNCoefficients);
   fOrthCoefficients(fNCoefficients) /= fOrthFunctionNorms(fNCoefficients);

   // Calculate the residual from including this fNCoefficients.
   Double_t dResidur = fOrthCoefficients(fNCoefficients) * b;

   return dResidur;
}


//____________________________________________________________________
void TMultiDimFit::MakeHistograms(Option_t *option)
{
   // Make histograms of the result of the analysis. This message
   // should be sent after having read all data points, but before
   // finding the parameterization
   //
   // Options:
   //     A         All the below
   //     X         Original independent variables
   //     D         Original dependent variables
   //     N         Normalised independent variables
   //     S         Shifted dependent variables
   //     R1        Residuals versus normalised independent variables
   //     R2        Residuals versus dependent variable
   //     R3        Residuals computed on training sample
   //     R4        Residuals computed on test sample
   //
   // For a description of these quantities, refer to
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   TString opt(option);
   opt.ToLower();

   if (opt.Length() < 1)
      return;

   if (!fHistograms)
      fHistograms = new TList;

   // Counter variable
   Int_t i = 0;

   // Histogram of original variables
   if (opt.Contains("x") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_XORIG);
      for (i = 0; i < fNVariables; i++)
         if (!fHistograms->FindObject(Form("x_%d_orig",i)))
            fHistograms->Add(new TH1D(Form("x_%d_orig",i),
                                      Form("Original variable # %d",i),
                                      fBinVarX, fMinVariables(i),
                                      fMaxVariables(i)));
   }

   // Histogram of original dependent variable
   if (opt.Contains("d") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_DORIG);
      if (!fHistograms->FindObject("d_orig"))
         fHistograms->Add(new TH1D("d_orig", "Original Quantity",
                                   fBinVarX, fMinQuantity, fMaxQuantity));
   }

   // Histograms of normalized variables
   if (opt.Contains("n") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_XNORM);
      for (i = 0; i < fNVariables; i++)
         if (!fHistograms->FindObject(Form("x_%d_norm",i)))
            fHistograms->Add(new TH1D(Form("x_%d_norm",i),
                                      Form("Normalized variable # %d",i),
                                      fBinVarX, -1,1));
   }

   // Histogram of shifted dependent variable
   if (opt.Contains("s") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_DSHIF);
      if (!fHistograms->FindObject("d_shifted"))
         fHistograms->Add(new TH1D("d_shifted", "Shifted Quantity",
                                   fBinVarX, fMinQuantity - fMeanQuantity,
                                   fMaxQuantity - fMeanQuantity));
   }

   // Residual from training sample versus independent variables
   if (opt.Contains("r1") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_RX);
      for (i = 0; i < fNVariables; i++)
         if (!fHistograms->FindObject(Form("res_x_%d",i)))
            fHistograms->Add(new TH2D(Form("res_x_%d",i),
                                      Form("Computed residual versus x_%d", i),
                                      fBinVarX, -1,    1,
                                      fBinVarY,
                                      fMinQuantity - fMeanQuantity,
                                      fMaxQuantity - fMeanQuantity));
   }

   // Residual from training sample versus. dependent variable
   if (opt.Contains("r2") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_RD);
      if (!fHistograms->FindObject("res_d"))
         fHistograms->Add(new TH2D("res_d",
                                   "Computed residuals vs Quantity",
                                   fBinVarX,
                                   fMinQuantity - fMeanQuantity,
                                   fMaxQuantity - fMeanQuantity,
                                   fBinVarY,
                                   fMinQuantity - fMeanQuantity,
                                   fMaxQuantity - fMeanQuantity));
   }

   // Residual from training sample
   if (opt.Contains("r3") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_RTRAI);
      if (!fHistograms->FindObject("res_train"))
         fHistograms->Add(new TH1D("res_train",
                                   "Computed residuals over training sample",
                                   fBinVarX, fMinQuantity - fMeanQuantity,
                                   fMaxQuantity - fMeanQuantity));

   }
   if (opt.Contains("r4") || opt.Contains("a")) {
      SETBIT(fHistogramMask,HIST_RTEST);
      if (!fHistograms->FindObject("res_test"))
         fHistograms->Add(new TH1D("res_test",
                                   "Distribution of residuals from test",
                                   fBinVarX,fMinQuantity - fMeanQuantity,
                                   fMaxQuantity - fMeanQuantity));
   }
}


//____________________________________________________________________
void TMultiDimFit::MakeMethod(const Char_t* classname, Option_t* option)
{
   // Generate the file <classname>MDF.cxx which contains the
   // implementation of the method:
   //
   //   Double_t <classname>::MDF(Double_t *x)
   //
   // which does the same as  TMultiDimFit::Eval. Please refer to this
   // method.
   //
   // Further, the public static members:
   //
   //   Int_t    <classname>::fgNVariables
   //   Int_t    <classname>::fgNCoefficients
   //   Double_t <classname>::fgDMean
   //   Double_t <classname>::fgXMean[]       //[fgNVariables]
   //   Double_t <classname>::fgXMin[]        //[fgNVariables]
   //   Double_t <classname>::fgXMax[]        //[fgNVariables]
   //   Double_t <classname>::fgCoefficient[] //[fgNCoeffficents]
   //   Int_t    <classname>::fgPower[]       //[fgNCoeffficents*fgNVariables]
   //
   // are initialized, and assumed to exist. The class declaration is
   // assumed to be in <classname>.h and assumed to be provided by the
   // user.
   //
   // See TMultiDimFit::MakeRealCode for a list of options
   //
   // The minimal class definition is:
   //
   //   class <classname> {
   //   public:
   //     Int_t    <classname>::fgNVariables;     // Number of variables
   //     Int_t    <classname>::fgNCoefficients;  // Number of terms
   //     Double_t <classname>::fgDMean;          // Mean from training sample
   //     Double_t <classname>::fgXMean[];        // Mean from training sample
   //     Double_t <classname>::fgXMin[];         // Min from training sample
   //     Double_t <classname>::fgXMax[];         // Max from training sample
   //     Double_t <classname>::fgCoefficient[];  // Coefficients
   //     Int_t    <classname>::fgPower[];        // Function powers
   //
   //     Double_t Eval(Double_t *x);
   //   };
   //
   // Whether the method <classname>::Eval should be static or not, is
   // up to the user.

   MakeRealCode(Form("%sMDF.cxx", classname), classname, option);
}



//____________________________________________________________________
void TMultiDimFit::MakeNormalized()
{
   // PRIVATE METHOD:
   // Normalize data to the interval [-1;1]. This is needed for the
   // classes method to work.

   Int_t i = 0;
   Int_t j = 0;
   Int_t k = 0;

   for (i = 0; i < fSampleSize; i++) {
      if (TESTBIT(fHistogramMask,HIST_DORIG))
         ((TH1D*)fHistograms->FindObject("d_orig"))->Fill(fQuantity(i));

      fQuantity(i) -= fMeanQuantity;
      fSumSqAvgQuantity  += fQuantity(i) * fQuantity(i);

      if (TESTBIT(fHistogramMask,HIST_DSHIF))
         ((TH1D*)fHistograms->FindObject("d_shifted"))->Fill(fQuantity(i));

      for (j = 0; j < fNVariables; j++) {
         Double_t range = 1. / (fMaxVariables(j) - fMinVariables(j));
         k              = i * fNVariables + j;

         // Fill histograms of original independent variables
         if (TESTBIT(fHistogramMask,HIST_XORIG))
            ((TH1D*)fHistograms->FindObject(Form("x_%d_orig",j)))
            ->Fill(fVariables(k));

         // Normalise independent variables
         fVariables(k) = 1 + 2 * range * (fVariables(k) - fMaxVariables(j));

         // Fill histograms of normalised independent variables
         if (TESTBIT(fHistogramMask,HIST_XNORM))
            ((TH1D*)fHistograms->FindObject(Form("x_%d_norm",j)))
            ->Fill(fVariables(k));

      }
   }
   // Shift min and max of dependent variable
   fMaxQuantity -= fMeanQuantity;
   fMinQuantity -= fMeanQuantity;

   // Shift mean of independent variables
   for (i = 0; i < fNVariables; i++) {
      Double_t range = 1. / (fMaxVariables(i) - fMinVariables(i));
      fMeanVariables(i) = 1 + 2 * range * (fMeanVariables(i)
                                           - fMaxVariables(i));
   }
}


//____________________________________________________________________
void TMultiDimFit::MakeParameterization()
{
   // PRIVATE METHOD:
   // Find the parameterization over the training sample. A full account
   // of the algorithm is given in the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html

   Int_t     i              = -1;
   Int_t     j              = 0;
   Int_t     k              = 0;
   Int_t     maxPass        = 3;
   Int_t     studied        = 0;
   Double_t  squareResidual = fSumSqAvgQuantity;
   fNCoefficients            = 0;
   fSumSqResidual           = fSumSqAvgQuantity;
   fFunctions.ResizeTo(fMaxTerms,fSampleSize);
   fOrthFunctions.ResizeTo(fMaxTerms,fSampleSize);
   fOrthFunctionNorms.ResizeTo(fMaxTerms);
   fOrthCoefficients.ResizeTo(fMaxTerms);
   fOrthCurvatureMatrix.ResizeTo(fMaxTerms,fMaxTerms);
   fFunctions = 1;

   fFunctionCodes = new Int_t[fMaxFunctions];
   fPowerIndex    = new Int_t[fMaxTerms];
   Int_t l;
   for (l=0;l<fMaxFunctions;l++) fFunctionCodes[l] = 0;
   for (l=0;l<fMaxTerms;l++)     fPowerIndex[l]    = 0;

   if (fMaxAngle != 0)  maxPass = 100;
   if (fIsUserFunction) maxPass = 1;

   // Loop over the number of functions we want to study.
   // increment inspection counter
   while(kTRUE) {

      // Reach user defined limit of studies
      if (studied++ >= fMaxStudy) {
         fParameterisationCode = PARAM_MAXSTUDY;
         break;
      }

      // Considered all functions several times
      if (k >= maxPass) {
         fParameterisationCode = PARAM_SEVERAL;
         break;
      }

      // increment function counter
      i++;

      // If we've reached the end of the functions, restart pass
      if (i == fMaxFunctions) {
         if (fMaxAngle != 0)
            fMaxAngle += (90 - fMaxAngle) / 2;
         i = 0;
         studied--;
         k++;
         continue;
      }
      if (studied == 1)
         fFunctionCodes[i] = 0;
      else if (fFunctionCodes[i] >= 2)
         continue;

      // Print a happy message
      if (fIsVerbose && studied == 1)
         std::cout << "Coeff   SumSqRes    Contrib   Angle      QM   Func"
         << "     Value        W^2  Powers" << std::endl;

      // Make the Gram-Schmidt
      Double_t dResidur = MakeGramSchmidt(i);

      if (dResidur == 0) {
         // This function is no good!
         // First test is in MakeGramSchmidt
         fFunctionCodes[i] = 1;
         continue;
      }

      // If user specified function, assume they know what they are doing
      if (!fIsUserFunction) {
         // Flag this function as considered
         fFunctionCodes[i] = 2;

         // Test if this function contributes to the fit
         if (!TestFunction(squareResidual, dResidur)) {
            fFunctionCodes[i] = 1;
            continue;
         }
      }

      // If we get to here, the function currently considered is
      // fNCoefficients, so we increment the counter
      // Flag this function as OK, and store and the number in the
      // index.
      fFunctionCodes[i]          = 3;
      fPowerIndex[fNCoefficients] = i;
      fNCoefficients++;

      // We add the current contribution to the sum of square of
      // residuals;
      squareResidual -= dResidur;


      // Calculate control parameter from this function
      for (j = 0; j < fNVariables; j++) {
         if (fNCoefficients == 1
             || fMaxPowersFinal[j] <= fPowers[i * fNVariables + j] - 1)
            fMaxPowersFinal[j] = fPowers[i * fNVariables + j] - 1;
      }
      Double_t s = EvalControl(&fPowers[i * fNVariables]);

      // Print the statistics about this function
      if (fIsVerbose) {
         std::cout << std::setw(5)  << fNCoefficients << " "
         << std::setw(10) << std::setprecision(4) << squareResidual << " "
         << std::setw(10) << std::setprecision(4) << dResidur << " "
         << std::setw(7)  << std::setprecision(3) << fMaxAngle << " "
         << std::setw(7)  << std::setprecision(3) << s << " "
         << std::setw(5)  << i << " "
         << std::setw(10) << std::setprecision(4)
         << fOrthCoefficients(fNCoefficients-1) << " "
         << std::setw(10) << std::setprecision(4)
         << fOrthFunctionNorms(fNCoefficients-1) << " "
         << std::flush;
         for (j = 0; j < fNVariables; j++)
            std::cout << " " << fPowers[i * fNVariables + j] - 1 << std::flush;
         std::cout << std::endl;
      }

      if (fNCoefficients >= fMaxTerms /* && fIsVerbose */) {
         fParameterisationCode = PARAM_MAXTERMS;
         break;
      }

      Double_t err  = TMath::Sqrt(TMath::Max(1e-20,squareResidual) /
                                  fSumSqAvgQuantity);
      if (err < fMinRelativeError) {
         fParameterisationCode = PARAM_RELERR;
         break;
      }

   }

   fError          = TMath::Max(1e-20,squareResidual);
   fSumSqResidual -= fError;
   fRMS = TMath::Sqrt(fError / fSampleSize);
}


//____________________________________________________________________
void TMultiDimFit::MakeRealCode(const char *filename,
                                const char *classname,
                                Option_t *)
{
   // PRIVATE METHOD:
   // This is the method that actually generates the code for the
   // evaluation the parameterization on some point.
   // It's called by TMultiDimFit::MakeCode and TMultiDimFit::MakeMethod.
   //
   // The options are: NONE so far
   Int_t i, j;

   Bool_t  isMethod     = (classname[0] == '\0' ? kFALSE : kTRUE);
   const char *prefix   = (isMethod ? Form("%s::", classname) : "");
   const char *cv_qual  = (isMethod ? "" : "static ");

   std::ofstream outFile(filename,std::ios::out|std::ios::trunc);
   if (!outFile) {
      Error("MakeRealCode","couldn't open output file '%s'",filename);
      return;
   }

   if (fIsVerbose)
      std::cout << "Writing on file \"" << filename << "\" ... " << std::flush;
   //
   // Write header of file
   //
   // Emacs mode line ;-)
   outFile << "// -*- mode: c++ -*-" << std::endl;
   // Info about creator
   outFile << "// " << std::endl
   << "// File " << filename
   << " generated by TMultiDimFit::MakeRealCode" << std::endl;
   // Time stamp
   TDatime date;
   outFile << "// on " << date.AsString() << std::endl;
   // ROOT version info
   outFile << "// ROOT version " << gROOT->GetVersion()
   << std::endl << "//" << std::endl;
   // General information on the code
   outFile << "// This file contains the function " << std::endl
   << "//" << std::endl
   << "//    double  " << prefix << "MDF(double *x); " << std::endl
   << "//" << std::endl
   << "// For evaluating the parameterization obtained" << std::endl
   << "// from TMultiDimFit and the point x" << std::endl
   << "// " << std::endl
   << "// See TMultiDimFit class documentation for more "
   << "information " << std::endl << "// " << std::endl;
   // Header files
   if (isMethod)
      // If these are methods, we need the class header
      outFile << "#include \"" << classname << ".h\"" << std::endl;

   //
   // Now for the data
   //
   outFile << "//" << std::endl
   << "// Static data variables"  << std::endl
   << "//" << std::endl;
   outFile << cv_qual << "int    " << prefix << "gNVariables    = "
   << fNVariables << ";" << std::endl;
   outFile << cv_qual << "int    " << prefix << "gNCoefficients = "
   << fNCoefficients << ";" << std::endl;
   outFile << cv_qual << "double " << prefix << "gDMean         = "
   << fMeanQuantity << ";" << std::endl;

   // Assignment to mean vector.
   outFile << "// Assignment to mean vector." << std::endl;
   outFile << cv_qual << "double " << prefix
   << "gXMean[] = {" << std::endl;
   for (i = 0; i < fNVariables; i++)
      outFile << (i != 0 ? ", " : "  ") << fMeanVariables(i) << std::flush;
   outFile << " };" << std::endl << std::endl;

   // Assignment to minimum vector.
   outFile << "// Assignment to minimum vector." << std::endl;
   outFile << cv_qual << "double " << prefix
   << "gXMin[] = {" << std::endl;
   for (i = 0; i < fNVariables; i++)
      outFile << (i != 0 ? ", " : "  ") << fMinVariables(i) << std::flush;
   outFile << " };" << std::endl << std::endl;

   // Assignment to maximum vector.
   outFile << "// Assignment to maximum vector." << std::endl;
   outFile << cv_qual << "double " << prefix
   << "gXMax[] = {" << std::endl;
   for (i = 0; i < fNVariables; i++)
      outFile << (i != 0 ? ", " : "  ") << fMaxVariables(i) << std::flush;
   outFile << " };" << std::endl << std::endl;

   // Assignment to coefficients vector.
   outFile << "// Assignment to coefficients vector." << std::endl;
   outFile << cv_qual << "double " << prefix
   << "gCoefficient[] = {" << std::flush;
   for (i = 0; i < fNCoefficients; i++)
      outFile << (i != 0 ? "," : "") << std::endl
      << "  " << fCoefficients(i) << std::flush;
   outFile << std::endl << " };" << std::endl << std::endl;

   // Assignment to error coefficients vector.
   outFile << "// Assignment to error coefficients vector." << std::endl;
   outFile << cv_qual << "double " << prefix
   << "gCoefficientRMS[] = {" << std::flush;
   for (i = 0; i < fNCoefficients; i++)
      outFile << (i != 0 ? "," : "") << std::endl
      << "  " << fCoefficientsRMS(i) << std::flush;
   outFile << std::endl << " };" << std::endl << std::endl;

   // Assignment to powers vector.
   outFile << "// Assignment to powers vector." << std::endl
   << "// The powers are stored row-wise, that is" << std::endl
   << "//  p_ij = " << prefix
   << "gPower[i * NVariables + j];" << std::endl;
   outFile << cv_qual << "int    " << prefix
   << "gPower[] = {" << std::flush;
   for (i = 0; i < fNCoefficients; i++) {
      for (j = 0; j < fNVariables; j++) {
         if (j != 0) outFile << std::flush << "  ";
         else        outFile << std::endl << "  ";
         outFile << fPowers[fPowerIndex[i] * fNVariables + j]
         << (i == fNCoefficients - 1 && j == fNVariables - 1 ? "" : ",")
         << std::flush;
      }
   }
   outFile << std::endl << "};" << std::endl << std::endl;


   //
   // Finally we reach the function itself
   //
   outFile << "// " << std::endl
   << "// The "
   << (isMethod ? "method " : "function ")
   << "  double " << prefix
   << "MDF(double *x)"
   << std::endl << "// " << std::endl;
   outFile << "double " << prefix
   << "MDF(double *x) {" << std::endl
   << "  double returnValue = " << prefix << "gDMean;" << std::endl
   << "  int    i = 0, j = 0, k = 0;" << std::endl
   << "  for (i = 0; i < " << prefix << "gNCoefficients ; i++) {"
   << std::endl
   << "    // Evaluate the ith term in the expansion" << std::endl
   << "    double term = " << prefix << "gCoefficient[i];"
   << std::endl
   << "    for (j = 0; j < " << prefix << "gNVariables; j++) {"
   << std::endl
   << "      // Evaluate the polynomial in the jth variable." << std::endl
   << "      int power = "<< prefix << "gPower["
   << prefix << "gNVariables * i + j]; " << std::endl
   << "      double p1 = 1, p2 = 0, p3 = 0, r = 0;" << std::endl
   << "      double v =  1 + 2. / ("
   << prefix << "gXMax[j] - " << prefix
   << "gXMin[j]) * (x[j] - " << prefix << "gXMax[j]);" << std::endl
   << "      // what is the power to use!" << std::endl
   << "      switch(power) {" << std::endl
   << "      case 1: r = 1; break; " << std::endl
   << "      case 2: r = v; break; " << std::endl
   << "      default: " << std::endl
   << "        p2 = v; " << std::endl
   << "        for (k = 3; k <= power; k++) { " << std::endl
   << "          p3 = p2 * v;" << std::endl;
   if (fPolyType == kLegendre)
      outFile << "          p3 = ((2 * i - 3) * p2 * v - (i - 2) * p1)"
      << " / (i - 1);" << std::endl;
   if (fPolyType == kChebyshev)
      outFile << "          p3 = 2 * v * p2 - p1; " << std::endl;
   outFile << "          p1 = p2; p2 = p3; " << std::endl << "        }" << std::endl
   << "        r = p3;" << std::endl << "      }" << std::endl
   << "      // multiply this term by the poly in the jth var" << std::endl
   << "      term *= r; " << std::endl << "    }" << std::endl
   << "    // Add this term to the final result" << std::endl
   << "    returnValue += term;" << std::endl << "  }" << std::endl
   << "  return returnValue;" << std::endl << "}" << std::endl << std::endl;

   // EOF
   outFile << "// EOF for " << filename << std::endl;

   // Close the file
   outFile.close();

   if (fIsVerbose)
      std::cout << "done" << std::endl;
}


//____________________________________________________________________
void TMultiDimFit::Print(Option_t *option) const
{
   // Print statistics etc.
   // Options are
   //   P        Parameters
   //   S        Statistics
   //   C        Coefficients
   //   R        Result of parameterisation
   //   F        Result of fit
   //   K        Correlation Matrix
   //   M        Pretty print formula
   //
   Int_t i = 0;
   Int_t j = 0;

   TString opt(option);
   opt.ToLower();

   if (opt.Contains("p")) {
      // Print basic parameters for this object
      std::cout << "User parameters:" << std::endl
      << "----------------" << std::endl
      << " Variables:                    " << fNVariables << std::endl
      << " Data points:                  " << fSampleSize << std::endl
      << " Max Terms:                    " << fMaxTerms << std::endl
      << " Power Limit Parameter:        " << fPowerLimit << std::endl
      << " Max functions:                " << fMaxFunctions << std::endl
      << " Max functions to study:       " << fMaxStudy << std::endl
      << " Max angle (optional):         " << fMaxAngle << std::endl
      << " Min angle:                    " << fMinAngle << std::endl
      << " Relative Error accepted:      " << fMinRelativeError << std::endl
      << " Maximum Powers:               " << std::flush;
      for (i = 0; i < fNVariables; i++)
         std::cout << " " << fMaxPowers[i] - 1 << std::flush;
      std::cout << std::endl << std::endl
      << " Parameterisation will be done using " << std::flush;
      if (fPolyType == kChebyshev)
         std::cout << "Chebyshev polynomials" << std::endl;
      else if (fPolyType == kLegendre)
         std::cout << "Legendre polynomials" << std::endl;
      else
         std::cout << "Monomials" << std::endl;
      std::cout << std::endl;
   }

   if (opt.Contains("s")) {
      // Print statistics for read data
      std::cout << "Sample statistics:" << std::endl
      << "------------------" << std::endl
      << "                 D"  << std::flush;
      for (i = 0; i < fNVariables; i++)
         std::cout << " " << std::setw(10) << i+1 << std::flush;
      std::cout << std::endl << " Max:   " << std::setw(10) << std::setprecision(7)
      << fMaxQuantity << std::flush;
      for (i = 0; i < fNVariables; i++)
         std::cout << " " << std::setw(10) << std::setprecision(4)
         << fMaxVariables(i) << std::flush;
      std::cout << std::endl << " Min:   " << std::setw(10) << std::setprecision(7)
      << fMinQuantity << std::flush;
      for (i = 0; i < fNVariables; i++)
         std::cout << " " << std::setw(10) << std::setprecision(4)
         << fMinVariables(i) << std::flush;
      std::cout << std::endl << " Mean:  " << std::setw(10) << std::setprecision(7)
      << fMeanQuantity << std::flush;
      for (i = 0; i < fNVariables; i++)
         std::cout << " " << std::setw(10) << std::setprecision(4)
         << fMeanVariables(i) << std::flush;
      std::cout << std::endl << " Function Sum Squares:         " << fSumSqQuantity
      << std::endl << std::endl;
   }

   if (opt.Contains("r")) {
      std::cout << "Results of Parameterisation:" << std::endl
      << "----------------------------" << std::endl
      << " Total reduction of square residuals    "
      << fSumSqResidual << std::endl
      << " Relative precision obtained:           "
      << fPrecision   << std::endl
      << " Error obtained:                        "
      << fError << std::endl
      << " Multiple correlation coefficient:      "
      << fCorrelationCoeff   << std::endl
      << " Reduced Chi square over sample:        "
      << fChi2 / (fSampleSize - fNCoefficients) << std::endl
      << " Maximum residual value:                "
      << fMaxResidual << std::endl
      << " Minimum residual value:                "
      << fMinResidual << std::endl
      << " Estimated root mean square:            "
      << fRMS << std::endl
      << " Maximum powers used:                   " << std::flush;
      for (j = 0; j < fNVariables; j++)
         std::cout << fMaxPowersFinal[j] << " " << std::flush;
      std::cout << std::endl
      << " Function codes of candidate functions." << std::endl
      << "  1: considered,"
      << "  2: too little contribution,"
      << "  3: accepted." << std::flush;
      for (i = 0; i < fMaxFunctions; i++) {
         if (i % 60 == 0)
            std::cout << std::endl << " " << std::flush;
         else if (i % 10 == 0)
            std::cout << " " << std::flush;
         std::cout << fFunctionCodes[i];
      }
      std::cout << std::endl << " Loop over candidates stopped because " << std::flush;
      switch(fParameterisationCode){
         case PARAM_MAXSTUDY:
            std::cout << "max allowed studies reached" << std::endl; break;
         case PARAM_SEVERAL:
            std::cout << "all candidates considered several times" << std::endl; break;
         case PARAM_RELERR:
            std::cout << "wanted relative error obtained" << std::endl; break;
         case PARAM_MAXTERMS:
            std::cout << "max number of terms reached" << std::endl; break;
         default:
            std::cout << "some unknown reason" << std::endl;
            break;
      }
      std::cout << std::endl;
   }

   if (opt.Contains("f")) {
      std::cout << "Results of Fit:" << std::endl
      << "---------------" << std::endl
      << " Test sample size:                      "
      << fTestSampleSize << std::endl
      << " Multiple correlation coefficient:      "
      << fTestCorrelationCoeff << std::endl
      << " Relative precision obtained:           "
      << fTestPrecision   << std::endl
      << " Error obtained:                        "
      << fTestError << std::endl
      << " Reduced Chi square over sample:        "
      << fChi2 / (fSampleSize - fNCoefficients) << std::endl
      << std::endl;
      if (fFitter) {
         fFitter->PrintResults(1,1);
         std::cout << std::endl;
      }
   }

   if (opt.Contains("c")){
      std::cout << "Coefficients:" << std::endl
      << "-------------" << std::endl
      << "   #         Value        Error   Powers" << std::endl
      << " ---------------------------------------" << std::endl;
      for (i = 0; i < fNCoefficients; i++) {
         std::cout << " " << std::setw(3) << i << "  "
         << std::setw(12) << fCoefficients(i) << "  "
         << std::setw(12) << fCoefficientsRMS(i) << "  " << std::flush;
         for (j = 0; j < fNVariables; j++)
            std::cout << " " << std::setw(3)
            << fPowers[fPowerIndex[i] * fNVariables + j] - 1 << std::flush;
         std::cout << std::endl;
      }
      std::cout << std::endl;
   }
   if (opt.Contains("k") && fCorrelationMatrix.IsValid()) {
      std::cout << "Correlation Matrix:" << std::endl
      << "-------------------";
      fCorrelationMatrix.Print();
   }

   if (opt.Contains("m")) {
      std::cout << "Parameterization:" << std::endl
      << "-----------------" << std::endl
      << "  Normalised variables: " << std::endl;
      for (i = 0; i < fNVariables; i++)
         std::cout << "\ty_" << i << "\t= 1 + 2 * (x_" << i << " - "
         << fMaxVariables(i) << ") / ("
         << fMaxVariables(i) << " - " << fMinVariables(i) << ")"
         << std::endl;
      std::cout << std::endl
      << "  f(";
      for (i = 0; i < fNVariables; i++) {
         std::cout << "y_" << i;
         if (i != fNVariables-1) std::cout << ", ";
      }
      std::cout << ") = ";
      for (i = 0; i < fNCoefficients; i++) {
         if (i != 0)
            std::cout << std::endl << "\t" << (fCoefficients(i) < 0 ? "- " : "+ ")
            << TMath::Abs(fCoefficients(i));
         else
            std::cout << fCoefficients(i);
         for (j = 0; j < fNVariables; j++) {
            Int_t p = fPowers[fPowerIndex[i] * fNVariables + j];
            switch (p) {
               case 1: break;
               case 2: std::cout << " * y_" << j; break;
               default:
                  switch(fPolyType) {
                     case kLegendre:  std::cout << " * L_" << p-1 << "(y_" << j << ")"; break;
                     case kChebyshev: std::cout << " * C_" << p-1 << "(y_" << j << ")"; break;
                     default:         std::cout << " * y_" << j << "^" << p-1; break;
                  }
            }

         }
      }
      std::cout << std::endl;
   }
}


//____________________________________________________________________
Bool_t TMultiDimFit::Select(const Int_t *)
{
   // Selection method. User can override this method for specialized
   // selection of acceptable functions in fit. Default is to select
   // all. This message is sent during the build-up of the function
   // candidates table once for each set of powers in
   // variables. Notice, that the argument array contains the powers
   // PLUS ONE. For example, to De select the function
   //     f = x1^2 * x2^4 * x3^5,
   // this method should return kFALSE if given the argument
   //     { 3, 4, 6 }
   return kTRUE;
}

//____________________________________________________________________
void TMultiDimFit::SetMaxAngle(Double_t ang)
{
   // Set the max angle (in degrees) between the initial data vector to
   // be fitted, and the new candidate function to be included in the
   // fit.  By default it is 0, which automatically chooses another
   // selection criteria. See also
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   if (ang >= 90 || ang < 0) {
      Warning("SetMaxAngle", "angle must be in [0,90)");
      return;
   }

   fMaxAngle = ang;
}

//____________________________________________________________________
void TMultiDimFit::SetMinAngle(Double_t ang)
{
   // Set the min angle (in degrees) between a new candidate function
   // and the subspace spanned by the previously accepted
   // functions. See also
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   if (ang > 90 || ang <= 0) {
      Warning("SetMinAngle", "angle must be in [0,90)");
      return;
   }

   fMinAngle = ang;

}


//____________________________________________________________________
void TMultiDimFit::SetPowers(const Int_t* powers, Int_t terms)
{
   // Define a user function. The input array must be of the form
   // (p11, ..., p1N, ... ,pL1, ..., pLN)
   // Where N is the dimension of the data sample, L is the number of
   // terms (given in terms) and the first number, labels the term, the
   // second the variable.  More information is given in the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   fIsUserFunction = kTRUE;
   fMaxFunctions   = terms;
   fMaxTerms       = terms;
   fMaxStudy       = terms;
   fMaxFuncNV      = fMaxFunctions * fNVariables;
   fPowers         = new Int_t[fMaxFuncNV];
   Int_t i, j;
   for (i = 0; i < fMaxFunctions; i++)
      for(j = 0; j < fNVariables; j++)
         fPowers[i * fNVariables + j] = powers[i * fNVariables + j]  + 1;
}

//____________________________________________________________________
void TMultiDimFit::SetPowerLimit(Double_t limit)
{
   // Set the user parameter for the function selection. The bigger the
   // limit, the more functions are used. The meaning of this variable
   // is defined in the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   fPowerLimit = limit;
}

//____________________________________________________________________
void TMultiDimFit::SetMaxPowers(const Int_t* powers)
{
   // Set the maximum power to be considered in the fit for each
   // variable. See also
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   if (!powers)
      return;

   for (Int_t i = 0; i < fNVariables; i++)
      fMaxPowers[i] = powers[i]+1;
}

//____________________________________________________________________
void TMultiDimFit::SetMinRelativeError(Double_t error)
{
   // Set the acceptable relative error for when sum of square
   // residuals is considered minimized. For a full account, refer to
   // the
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html
   fMinRelativeError = error;
}


//____________________________________________________________________
Bool_t TMultiDimFit::TestFunction(Double_t squareResidual,
                                  Double_t dResidur)
{
   // PRIVATE METHOD:
   // Test whether the currently considered function contributes to the
   // fit. See also
   // Begin_Html<a href="#TMultiDimFit:description">class description</a>End_Html

   if (fNCoefficients != 0) {
      // Now for the second test:
      if (fMaxAngle == 0) {
         // If the user hasn't supplied a max angle do the test as,
         if (dResidur <
             squareResidual / (fMaxTerms - fNCoefficients + 1 + 1E-10)) {
            return kFALSE;
         }
      }
      else {
         // If the user has provided a max angle, test if the calculated
         // angle is less then the max angle.
         if (TMath::Sqrt(dResidur/fSumSqAvgQuantity) <
             TMath::Cos(fMaxAngle*DEGRAD)) {
            return kFALSE;
         }
      }
   }
   // If we get here, the function is OK
   return kTRUE;
}


//____________________________________________________________________
void mdfHelper(int& /*npar*/, double* /*divs*/, double& chi2,
               double* coeffs, int /*flag*/)
{
   // Helper function for doing the minimisation of Chi2 using Minuit

   // Get pointer  to current TMultiDimFit object.
   TMultiDimFit* mdf = TMultiDimFit::Instance();
   chi2     = mdf->MakeChi2(coeffs);
}
