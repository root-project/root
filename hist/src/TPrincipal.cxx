//____________________________________________________________________
//Begin_Html  
/* 
   </pre>
<H1><A NAME="SECTION00010000000000000000"></A>
<A NAME="sec:lintra"></A>
<BR>
Principal Components Analysis (PCA)
</H1>

<P>
The current implementation is based on the LINTRA package from CERNLIB
by R. Brun, H. Hansroul, and J. Kubler.
The class has been implemented by Christian Holm Cristensen in August 2000.

<P>

<H2><A NAME="SECTION00011000000000000000"></A>
<A NAME="sec:intro1"></A>
<BR>
Introduction
</H2>

<P>
In many applications of various fields of research, the treatment of
large amounts of data requires powerful techniques capable of rapid
data reduction and analysis. Usually, the quantities most
conveniently measured by the experimentalist, are not necessarily the
most significant for classification and analysis of the data. It is
then useful to have a way of selecting an optimal set of variables
necessary for the recognition process and reducing the dimensionality
of the problem, resulting in an easier classification procedure. 

<P>
This paper describes the implementation of one such method of
feature selection, namely the principal components analysis. This  
multidimensional technique is well known in the field of pattern
recognition and and its use in Particle Physics has been documented
elsewhere (cf. H. Wind, <I>Function Parameterization</I>, CERN
72-21).  

<P>

<H2><A NAME="SECTION00012000000000000000"></A>
<A NAME="sec:overview"></A>
<BR>
Overview
</H2>

<P>
Suppose we have prototypes which are trajectories of particles,
passing through a spectrometer. If one measures the passage of the
particle at say 8 fixed planes, the trajectory is described by an
8-component vector:
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
\mathbf{x} = \left(x_1, x_2, \ldots, x_8\right)
\end{displaymath}
 -->


<IMG
 WIDTH="145" HEIGHT="31" BORDER="0"
 SRC="gif/principal_img1.gif"
 ALT="\begin{displaymath}
\mathbf{x} = \left(x_1, x_2, \ldots, x_8\right)
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>
in 8-dimensional pattern space. 

<P>
One proceeds by generating a a representative tracks sample and
building up the covariance matrix <IMG
 WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img2.gif"
 ALT="$\mathsf{C}$">. Its eigenvectors and
eigenvalues are computed by standard methods, and thus a new basis is
obtained for the original 8-dimensional space the expansion of the
prototypes, 
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
\mathbf{x}_m = \sum^8_{i=1} a_{m_i} \mathbf{e}_i
\quad
\mbox{where}
\quad
a_{m_i} = \mathbf{x}^T\bullet\mathbf{e}_i
\end{displaymath}
 -->


<IMG
 WIDTH="295" HEIGHT="58" BORDER="0"
 SRC="gif/principal_img3.gif"
 ALT="\begin{displaymath}
\mathbf{x}_m = \sum^8_{i=1} a_{m_i} \mathbf{e}_i
\quad
\mbox{where}
\quad
a_{m_i} = \mathbf{x}^T\bullet\mathbf{e}_i
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>

<P>
allows the study of the behavior of the coefficients <IMG
 WIDTH="31" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img4.gif"
 ALT="$a_{m_i}$"> for all
the tracks of the sample. The eigenvectors which are insignificant for
the trajectory description in the expansion will have their
corresponding coefficients <IMG
 WIDTH="31" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img4.gif"
 ALT="$a_{m_i}$"> close to zero for all the
prototypes. 

<P>
On one hand, a reduction of the dimensionality is then obtained by
omitting these least significant vectors in the subsequent analysis. 

<P>
On the other hand, in the analysis of real data, these least
significant variables(?) can be used for the pattern
recognition problem of extracting the valid combinations of
coordinates describing a true trajectory from the set of all possible
wrong combinations. 

<P>
The program described here performs this principal components analysis
on a sample of data provided by the user. It computes the covariance
matrix, its eigenvalues ands corresponding eigenvectors and exhibits
the behavior of the principal components (<IMG
 WIDTH="31" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img4.gif"
 ALT="$a_{m_i}$">), thus providing
to the user all the means of understanding his data. 

<P>
A short outline of the method of Principal Components is given in
subsection&nbsp;<A HREF="prin_node1.html#sec:method">1.3</A>.

<P>

<H2><A NAME="SECTION00013000000000000000"></A>
<A NAME="sec:method"></A>
<BR>
Principal Components Method
</H2>

<P>
Let's consider a sample of <IMG
 WIDTH="23" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img5.gif"
 ALT="$M$"> prototypes each being characterized by
<IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img6.gif"
 ALT="$P$"> variables 
<!-- MATH
 $x_1, x_2, \ldots, x_P$
 -->
<IMG
 WIDTH="107" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img7.gif"
 ALT="$x_1, x_2, \ldots, x_P$">. Each prototype is a point, or a
column vector, in a <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img6.gif"
 ALT="$P$">-dimensional <I>pattern space</I>. 
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\mathbf{x} = \left[\begin{array}{c}
    x_1\\x_2\\\vdots\\x_P\end{array}\right]\,,
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><IMG
 WIDTH="102" HEIGHT="102" BORDER="0"
 SRC="gif/principal_img8.gif"
 ALT="\begin{displaymath}
\mathbf{x} = \left[\begin{array}{c}
x_1\\ x_2\\ \vdots\\ x_P\end{array}\right]\,,
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(1)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
where each <IMG
 WIDTH="23" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img9.gif"
 ALT="$x_n$"> represents the particular value associated with the
<IMG
 WIDTH="15" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img10.gif"
 ALT="$n$">-dimension. 

<P>
Those <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img6.gif"
 ALT="$P$"> variables are the quantities accessible to the
experimentalist, but are not necessarily the most significant for the
classification purpose. 

<P>
The <I>Principal Components Method</I> consists of applying a
<I>linear</I> transformation to the original variables. This
transformation is described by an orthogonal matrix and is equivalent
to a rotation of the original pattern space into a new set of
coordinate vectors, which hopefully provide easier feature
identification and dimensionality reduction. 

<P>
Let's define the covariance matrix:
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\mathsf{C} = \left\langle\mathbf{y}\mathbf{y}^T\right\rangle
  \quad\mbox{where}\quad
  \mathbf{y} = \mathbf{x} - \left\langle\mathbf{x}\right\rangle\,,
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:C"></A><IMG
 WIDTH="267" HEIGHT="37" BORDER="0"
 SRC="gif/principal_img11.gif"
 ALT="\begin{displaymath}
\mathsf{C} = \left\langle\mathbf{y}\mathbf{y}^T\right\rangl...
...athbf{y} = \mathbf{x} - \left\langle\mathbf{x}\right\rangle\,,
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(2)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
and the brackets indicate mean value over the sample of <IMG
 WIDTH="23" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img5.gif"
 ALT="$M$">
prototypes. 

<P>
This matrix <IMG
 WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img2.gif"
 ALT="$\mathsf{C}$"> is real, positive definite, symmetric, and will
have all its eigenvalues greater then zero. It will now be show that
among the family of all the complete orthonormal bases of the pattern
space, the base formed by the eigenvectors of the covariance matrix
and belonging to the largest eigenvalues, corresponds to the most 
significant features of the description of the original prototypes. 

<P>
let the prototypes be expanded on into a set of <IMG
 WIDTH="20" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img12.gif"
 ALT="$N$"> basis vectors

<!-- MATH
 $\mathbf{e}_n, n=1,\ldots,N,N+1, \ldots, P$
 -->
<IMG
 WIDTH="233" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img13.gif"
 ALT="$\mathbf{e}_n, n=1,\ldots,N,N+1, \ldots, P$">, 
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\mathbf{y}_i = \sum^N_{i=1} a_{i_n} \mathbf{e}_n,
  \quad 
  i = 1, \ldots, M, 
  \quad
  N < P
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:yi"></A><IMG
 WIDTH="303" HEIGHT="58" BORDER="0"
 SRC="gif/principal_img14.gif"
 ALT="\begin{displaymath}
\mathbf{y}_i = \sum^N_{i=1} a_{i_n} \mathbf{e}_n,
\quad
i = 1, \ldots, M,
\quad
N &lt; P
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(3)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>

<P>
The `best' feature coordinates <IMG
 WIDTH="23" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img15.gif"
 ALT="$\mathbf{e}_n$">, spanning a <I>feature
  space</I>,  will be obtained by minimizing the error due to this
truncated expansion, i.e.,  
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\min\left(E_N\right) = 
  \min\left[\left\langle\left(\mathbf{y}_i - \sum^N_{i=1} a_{i_n} \mathbf{e}_n\right)^2\right\rangle\right]
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:mini"></A><IMG
 WIDTH="306" HEIGHT="65" BORDER="0"
 SRC="gif/principal_img16.gif"
 ALT="\begin{displaymath}
\min\left(E_N\right) =
\min\left[\left\langle\left(\mathb...
...\sum^N_{i=1} a_{i_n} \mathbf{e}_n\right)^2\right\rangle\right]
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(4)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
with the conditions:
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\mathbf{e}_k\bullet\mathbf{e}_j = \delta_{jk} = 
  \left\{\begin{array}{rcl}
    1 & \mbox{for} & k = j\\
    0 & \mbox{for} & k \neq j
  \end{array}\right.
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:ortocond"></A><IMG
 WIDTH="240" HEIGHT="54" BORDER="0"
 SRC="gif/principal_img17.gif"
 ALT="\begin{displaymath}
\mathbf{e}_k\bullet\mathbf{e}_j = \delta_{jk} =
\left\{\b...
...for} &amp; k = j\\
0 &amp; \mbox{for} &amp; k \neq j
\end{array}\right.
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(5)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>

<P>
Multiplying (<A HREF="prin_node1.html#eq:yi">3</A>) by 
<!-- MATH
 $\mathbf{e}^T_n$
 -->
<IMG
 WIDTH="24" HEIGHT="38" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img18.gif"
 ALT="$\mathbf{e}^T_n$"> using (<A HREF="prin_node1.html#eq:ortocond">5</A>),
we get  
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
a_{i_n} = \mathbf{y}_i^T\bullet\mathbf{e}_n\,,
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:ai"></A><IMG
 WIDTH="108" HEIGHT="31" BORDER="0"
 SRC="gif/principal_img19.gif"
 ALT="\begin{displaymath}
a_{i_n} = \mathbf{y}_i^T\bullet\mathbf{e}_n\,,
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(6)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
so the error becomes
<BR>
<DIV ALIGN="CENTER"><A NAME="eq:error"></A>

<!-- MATH
 \begin{eqnarray}
E_N &=& 
  \left\langle\left[\sum_{n=N+1}^P  a_{i_n}\mathbf{e}_n\right]^2\right\rangle\nonumber\\
  &=& 
  \left\langle\left[\sum_{n=N+1}^P  \mathbf{y}_i^T\bullet\mathbf{e}_n\mathbf{e}_n\right]^2\right\rangle\nonumber\\
  &=&
  \left\langle\sum_{n=N+1}^P  \mathbf{e}_n^T\mathbf{y}_i\mathbf{y}_i^T\mathbf{e}_n\right\rangle\nonumber\\
  &=&
  \sum_{n=N+1}^P  \mathbf{e}_n^T\mathsf{C}\mathbf{e}_n
\end{eqnarray}
 -->

<TABLE ALIGN="CENTER" CELLPADDING="0" WIDTH="100%">
<TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT"><IMG
 WIDTH="30" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img20.gif"
 ALT="$\displaystyle E_N$"></TD>
<TD ALIGN="CENTER" NOWRAP><IMG
 WIDTH="18" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img21.gif"
 ALT="$\textstyle =$"></TD>
<TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="151" HEIGHT="80" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img22.gif"
 ALT="$\displaystyle \left\langle\left[\sum_{n=N+1}^P a_{i_n}\mathbf{e}_n\right]^2\right\rangle$"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
&nbsp;</TD></TR>
<TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT">&nbsp;</TD>
<TD ALIGN="CENTER" NOWRAP><IMG
 WIDTH="18" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img21.gif"
 ALT="$\textstyle =$"></TD>
<TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="184" HEIGHT="80" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img23.gif"
 ALT="$\displaystyle \left\langle\left[\sum_{n=N+1}^P \mathbf{y}_i^T\bullet\mathbf{e}_n\mathbf{e}_n\right]^2\right\rangle$"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
&nbsp;</TD></TR>
<TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT">&nbsp;</TD>
<TD ALIGN="CENTER" NOWRAP><IMG
 WIDTH="18" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img21.gif"
 ALT="$\textstyle =$"></TD>
<TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="156" HEIGHT="69" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img24.gif"
 ALT="$\displaystyle \left\langle\sum_{n=N+1}^P \mathbf{e}_n^T\mathbf{y}_i\mathbf{y}_i^T\mathbf{e}_n\right\rangle$"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
&nbsp;</TD></TR>
<TR VALIGN="MIDDLE"><TD NOWRAP ALIGN="RIGHT">&nbsp;</TD>
<TD ALIGN="CENTER" NOWRAP><IMG
 WIDTH="18" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img21.gif"
 ALT="$\textstyle =$"></TD>
<TD ALIGN="LEFT" NOWRAP><IMG
 WIDTH="104" HEIGHT="69" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img25.gif"
 ALT="$\displaystyle \sum_{n=N+1}^P \mathbf{e}_n^T\mathsf{C}\mathbf{e}_n$"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(7)</TD></TR>
</TABLE></DIV>
<BR CLEAR="ALL"><P></P>

<P>
The minimization of the sum in (<A HREF="prin_node1.html#eq:error">7</A>) is obtained when each
term 
<!-- MATH
 $\mathbf{e}_n^\mathsf{C}\mathbf{e}_n$
 -->
<IMG
 WIDTH="41" HEIGHT="38" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img26.gif"
 ALT="$\mathbf{e}_n^\mathsf{C}\mathbf{e}_n$"> is minimum, since <IMG
 WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img2.gif"
 ALT="$\mathsf{C}$"> is
positive definite. By the method of Lagrange multipliers, and the
condition&nbsp;(<A HREF="prin_node1.html#eq:ortocond">5</A>), we get

<P>
 
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
E_N = \sum^P_{n=N+1} \left(\mathbf{e}_n^T\mathsf{C}\mathbf{e}_n - 
    l_n\mathbf{e}_n^T\bullet\mathbf{e}_n + l_n\right)
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:minerror"></A><IMG
 WIDTH="291" HEIGHT="60" BORDER="0"
 SRC="gif/principal_img27.gif"
 ALT="\begin{displaymath}
E_N = \sum^P_{n=N+1} \left(\mathbf{e}_n^T\mathsf{C}\mathbf{e}_n -
l_n\mathbf{e}_n^T\bullet\mathbf{e}_n + l_n\right)
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(8)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
The minimum condition 
<!-- MATH
 $\frac{dE_N}{d\mathbf{e}^T_n} = 0$
 -->
<IMG
 WIDTH="68" HEIGHT="40" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img28.gif"
 ALT="$\frac{dE_N}{d\mathbf{e}^T_n} = 0$"> leads to the
equation 
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\mathsf{C}\mathbf{e}_n = l_n\mathbf{e}_n\,,
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:Ce"></A><IMG
 WIDTH="91" HEIGHT="30" BORDER="0"
 SRC="gif/principal_img29.gif"
 ALT="\begin{displaymath}
\mathsf{C}\mathbf{e}_n = l_n\mathbf{e}_n\,,
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(9)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
which shows that <IMG
 WIDTH="23" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img15.gif"
 ALT="$\mathbf{e}_n$"> is an eigenvector of the covariance
matrix <IMG
 WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img2.gif"
 ALT="$\mathsf{C}$"> with eigenvalue <IMG
 WIDTH="19" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img30.gif"
 ALT="$l_n$">. The estimated minimum error is
then given by
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
E_N \sim \sum^P_{n=N+1} \mathbf{e}_n^T\bullet l_n\mathbf{e}_n 
      = \sum^P_{n=N+1}  l_n\,,
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:esterror"></A><IMG
 WIDTH="264" HEIGHT="60" BORDER="0"
 SRC="gif/principal_img31.gif"
 ALT="\begin{displaymath}
E_N \sim \sum^P_{n=N+1} \mathbf{e}_n^T\bullet l_n\mathbf{e}_n
= \sum^P_{n=N+1} l_n\,,
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(10)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
where 
<!-- MATH
 $l_n,\,n=N+1,\ldots,P$
 -->
<IMG
 WIDTH="161" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img32.gif"
 ALT="$l_n,\,n=N+1,\ldots,P$"> are the eigenvalues associated with the
omitted eigenvectors in the expansion&nbsp;(<A HREF="prin_node1.html#eq:yi">3</A>). Thus, by choosing
the <IMG
 WIDTH="20" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img12.gif"
 ALT="$N$"> largest eigenvalues, and their associated eigenvectors, the
error <IMG
 WIDTH="30" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img33.gif"
 ALT="$E_N$"> is minimized. 

<P>
The transformation matrix to go from the pattern space to the feature
space consists of the ordered eigenvectors

<!-- MATH
 $\mathbf{e}_1,\ldots,\mathbf{e}_P$
 -->
<IMG
 WIDTH="80" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
 SRC="gif/principal_img34.gif"
 ALT="$\mathbf{e}_1,\ldots,\mathbf{e}_P$"> for its columns
<BR>
<DIV ALIGN="RIGHT">


<!-- MATH
 \begin{equation}
\mathsf{T} = \left[
    \begin{array}{cccc}
      \mathbf{e}_1 &
      \mathbf{e}_2 &
      \vdots &
      \mathbf{e}_P
    \end{array}\right]
  = \left[
    \begin{array}{cccc}
      \mathbf{e}_{1_1} &  \mathbf{e}_{2_1} & \cdots &  \mathbf{e}_{P_1}\\
      \mathbf{e}_{1_2} &  \mathbf{e}_{2_2} & \cdots &  \mathbf{e}_{P_2}\\
      \vdots        &  \vdots        & \ddots &  \vdots \\
      \mathbf{e}_{1_P} &  \mathbf{e}_{2_P} & \cdots &  \mathbf{e}_{P_P}\\
    \end{array}\right]
\end{equation}
 -->

<TABLE WIDTH="100%" ALIGN="CENTER">
<TR VALIGN="MIDDLE"><TD ALIGN="CENTER" NOWRAP><A NAME="eq:trans"></A><IMG
 WIDTH="378" HEIGHT="102" BORDER="0"
 SRC="gif/principal_img35.gif"
 ALT="\begin{displaymath}
\mathsf{T} = \left[
\begin{array}{cccc}
\mathbf{e}_1 &amp;
\...
...bf{e}_{2_P} &amp; \cdots &amp; \mathbf{e}_{P_P}\\
\end{array}\right]
\end{displaymath}"></TD>
<TD WIDTH=10 ALIGN="RIGHT">
(11)</TD></TR>
</TABLE>
<BR CLEAR="ALL"></DIV><P></P>
This is an orthogonal transformation, or rotation, of the pattern
space and feature selection results in ignoring certain coordinates
in the transformed space.
   <p>
   <DIV ALIGN="RIGHT">
   Christian Holm<br>
   August 2000, CERN
   </DIV>
   <pre> 
 */
//End_Html

// $Id: TPrincipal.cxx,v 1.5 2000/08/15 08:44:15 brun Exp $
// $Date: 2000/08/15 08:44:15 $
// $Author: brun $

#include "TPrincipal.h"

#include "TVectorD.h"
#include "TMatrixD.h"
#include "TList.h"
#include "TH2.h"
#include "TDatime.h"
#include "TBrowser.h"
#include "TROOT.h"

#include <fstream.h>
#include <iostream.h>
#include <iomanip.h>


ClassImp(TPrincipal); 

//____________________________________________________________________
TPrincipal::TPrincipal(void) 
{
  // Empty CTOR, Do not use. 

      fHistograms = 0;
}

//____________________________________________________________________
TPrincipal::~TPrincipal(void) 
{
  // Empty DTOR

   if (fHistograms) {
      fHistograms->Delete();
      delete fHistograms;
   }
}

//____________________________________________________________________
TPrincipal::TPrincipal(Int_t nVariables, Option_t* opt) 
  : fMeanValues(1,nVariables),
    fSigmas(1,nVariables),
    fCovarianceMatrix(1,nVariables, 1,nVariables),
    fEigenVectors(1,nVariables,1,nVariables),
    fEigenValues(1,nVariables),
    fOffDiagonal(1,nVariables),
    fUserData(nVariables*1000)
{
  // Ctor. Argument is number of variables in the sample of data 
  // Options are:
  //   N       Normalize the covariance matrix (default)
  //   <empty> Do not Normalize the covariance matrix
  if (nVariables <= 1) {
    Error("TPrincipal", "You can't be serious - nVariables == 1!!!");
    return;
  }

  fHistograms = 0;
  fIsNormalised       = kFALSE;
  fNumberOfDataPoints = 0;
  fNumberOfVariables  = nVariables; 
  while (strlen(opt) > 0) {
    switch(*opt++) {
    case 'N':
      fIsNormalised = kTRUE;
      break;
    default:
      Warning("TPrincipal", "Option '%s' for unknown - ignored",
	      *opt);
      break;
    }
  }

  if (!fMeanValues.IsValid())
    Error("TPrincipal","Couldn't create vector mean values");
  if (!fSigmas.IsValid())
    Error("TPrincipal","Couldn't create vector sigmas");
  if (!fCovarianceMatrix.IsValid())
    Error("TPrincipal","Couldn't create covariance matrix");
  if (!fEigenVectors.IsValid())
    Error("TPrincipal","Couldn't create eigenvector matrix");
  if (!fEigenValues.IsValid())
    Error("TPrincipal","Couldn't create eigenvalue vector");
  if (!fOffDiagonal.IsValid())
    Error("TPrincipal","Couldn't create offdiagonal vector");
  if (!fUserData.IsValid())
    Error("TPrincipal","Couldn't create user data vector");
}

//____________________________________________________________________
void
TPrincipal::AddRow(Double_t* p) 
{
  // Begin_Html
  /* 
     </PRE>
Add a data point and update the covariance matrix. The input
array must be <TT>fNumberOfVariables</TT> long.

<P>
The Covariance matrix and mean values of the input data is caculated
on the fly by the following equations:  
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
\left<x_i\right>^{(1)}  = x_{i1}
\end{displaymath}
 -->


<IMG
 WIDTH="90" HEIGHT="31" BORDER="0"
 SRC="gif/principal_img36.gif"
 ALT="\begin{displaymath}
\left&lt;x_i\right&gt;^{(1)} = x_{i1}
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
\left<x_i\right>^{(n)} = \left<x_i\right>^{(n-1)} 
+ \frac1n \left(x_{in} - \left<x_i\right>^{(n-1)}\right)
\end{displaymath}
 -->


<IMG
 WIDTH="302" HEIGHT="42" BORDER="0"
 SRC="gif/principal_img37.gif"
 ALT="\begin{displaymath}
\left&lt;x_i\right&gt;^{(n)} = \left&lt;x_i\right&gt;^{(n-1)}
+ \frac1n \left(x_{in} - \left&lt;x_i\right&gt;^{(n-1)}\right)
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
C_{ij}^{(1)} = 0
\end{displaymath}
 -->


<IMG
 WIDTH="62" HEIGHT="34" BORDER="0"
 SRC="gif/principal_img38.gif"
 ALT="\begin{displaymath}
C_{ij}^{(1)} = 0
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
C_{ij}^{(n)} = C_{ij}^{(n-1)} 
+ \frac1{n-1}\left[\left(x_{in} - \left<x_i\right>^{(n)}\right)
  \left(x_{jn} - \left<x_j\right>^{(n)}\right)\right] 
- \frac1n C_{ij}^{(n-1)}
\end{displaymath}
 -->


<IMG
 WIDTH="504" HEIGHT="43" BORDER="0"
 SRC="gif/principal_img39.gif"
 ALT="\begin{displaymath}
C_{ij}^{(n)} = C_{ij}^{(n-1)}
+ \frac1{n-1}\left[\left(x_{i...
...\left&lt;x_j\right&gt;^{(n)}\right)\right]
- \frac1n C_{ij}^{(n-1)}
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>
since this is a really fast method, with no rounding errors (please
refer to CERN 72-21 pp. 54-106).  

<P>
The data is stored internally in a <TT>TVectorD</TT>, in the following
way: 
<BR><P></P>
<DIV ALIGN="CENTER">

<!-- MATH
 \begin{displaymath}
\mathbf{x} = \left[\left(x_{11},\ldots,x_{P1}\right),\ldots, 
    \left(x_{1i},\ldots,x_{Pi}\right), \ldots\right]
\end{displaymath}
 -->


<IMG
 WIDTH="319" HEIGHT="31" BORDER="0"
 SRC="gif/principal_img40.gif"
 ALT="\begin{displaymath}
\mathbf{x} = \left[\left(x_{11},\ldots,x_{P1}\right),\ldots,
\left(x_{1i},\ldots,x_{Pi}\right), \ldots\right]
\end{displaymath}">
</DIV>
<BR CLEAR="ALL">
<P></P>
With <IMG
 WIDTH="18" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="gif/principal_img6.gif"
 ALT="$P$"> as defined in the class description.
     <PRE>
  */
  // End_Html 
  if (!p) 
    return; 

  // Increment the data point counter 
  Int_t i,j;
  if (++fNumberOfDataPoints == 1) {
    for (Int_t i = 0; i < fNumberOfVariables; i++)
      fMeanValues(i+1) = p[i];
  }
  else {
    
    Double_t cor = 1 - 1./Double_t(fNumberOfDataPoints - 1);
    for (i = 0; i < fNumberOfVariables; i++) {
      
      fMeanValues(i + 1) *= cor;
      fMeanValues(i + 1) += p[i] / Double_t(fNumberOfDataPoints - 1); 
      Double_t t1 = (p[i] - fMeanValues(i+1)) / (fNumberOfDataPoints - 1);
      
      // Setting Matrix (lower triangle) elements
      for (j = 0; j < i + 1; j++) {
	fCovarianceMatrix(i+1,j+1) *= cor;
	fCovarianceMatrix(i+1,j+1) += t1 * (p[j] - fMeanValues(j+1));  
      }
    }
  }
  
  // Store data point in internal vector
  // If the vector isn't big enough to hold the new data, then 
  // expand the vector by half it's size. 
  Int_t size = fUserData.GetNrows();
  if (fNumberOfDataPoints * fNumberOfVariables > size)
    fUserData.ResizeTo(size + size/2);
 
  for (i = 0; i < fNumberOfVariables; i++) {
     j = (fNumberOfDataPoints-1) * fNumberOfVariables + i;
    fUserData(j) = p[i];
  }
  
}

//____________________________________________________________________
void
TPrincipal::Browse(TBrowser* b)
{
  // Browse the TPrincipal object in the TBrowser. 
  if (fHistograms) {
    TIter next(fHistograms);
    TH1* h = 0;
    while ((h = (TH1*)next())) 
      b->Add(h,h->GetName());
  }
  
  b->Add(&fUserData,"User Data");
  b->Add(&fCovarianceMatrix,"Covariance Matrix");
  b->Add(&fMeanValues,"Mean value vector");
  b->Add(&fSigmas,"Sigma value vector");
  b->Add(&fEigenValues,"Eigenvalue vector");
  b->Add(&fEigenVectors,"Eigenvector Matrix");

}

//____________________________________________________________________
void
TPrincipal::Clear(Option_t* opt) 
{
  // Clear the data in Object. Notice, that's not possible to change
  // the dimension of the original data. 
  if (fHistograms) {
    fHistograms->Delete(opt);
  }
  
  fNumberOfDataPoints = 0;
  fTrace              = 0;
  fCovarianceMatrix.Zero();
  fEigenVectors.Zero();
  fEigenValues.Zero();
  fMeanValues.Zero();
  fSigmas.Zero();
  fOffDiagonal.Zero();
  
  fUserData.ResizeTo(fNumberOfVariables * 1000);
  fUserData.Zero();
}

//____________________________________________________________________
const Double_t*
TPrincipal::GetRow(Int_t row) 
{
  // Return a row of the user supplied data. 
  // If row is out of bounds, 0 is returned. 
  // It's up to the user to delete the returned array. 
  // Row 0 is the first row;
  if (row >= fNumberOfDataPoints) 
    return 0;
  
  Int_t index   = row  * fNumberOfVariables;
  return &fUserData(index);
}


//____________________________________________________________________
void
TPrincipal::MakeCode(const Char_t* filename, Option_t *opt)
{
  // Generates the file <filename>, with .C appended if it does
  // argument doesn't end in .cxx or .C.
  //
  // The file contains the implementation of two functions
  //
  //    void X2P(Double_t* x, Double* p)
  //    void P2X(Double_t* p, Double* x, Int_t nTest)
  // 
  // which does the same as  TPrincipal::X2P and TPrincipal::P2X
  // respectively. Please refer to these methods.
  // 
  // Further, the static variables:
  //
  //    Int_t    fgNVariables
  //    Double_t fgEigenValues[]
  //    Double_t fgEigenVectors[]
  //    Double_t fgMeanValues[]
  //    Double_t fgSigmaValues[]
  // 
  // are initialized. The only ROOT header file needed is Rtypes.h
  //
  // See TPrincipal::MakeRealCode for a list of options

  TString outName(filename);
  if (!outName.EndsWith(".C") && !outName.EndsWith(".cxx"))
    outName += ".C";

  MakeRealCode(outName.Data(),"",opt);
}

//____________________________________________________________________
void 
TPrincipal::MakeEigenVectors(void) 
{
  // Begin_Html
  /*
    </PRE>
    PRIVATE METHOD:
    <BR>
    Find eigenvalues and vectors of tridiagonalised covariance matrix
    according to the <I>QL with implicit shift</I> algorithm from  
    <A NAME="tex2html1"
    HREF="http://www.nr.com">Numerical Recipes in C</A>
    section&nbsp;11.3. 
    <P>
    The basic idea is to find matrices <IMG
    WIDTH="17" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img41.gif"
    ALT="$\mathsf{Q}$"> and <IMG
    WIDTH="14" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img42.gif"
    ALT="$\mathsf{L}$"> so that  
    <!-- MATH
    $\mathsf{C} = \mathsf{Q} \cdot \mathsf{L}$
    -->
    <IMG
    WIDTH="74" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img43.gif"
    ALT="$\mathsf{C} = \mathsf{Q} \cdot \mathsf{L}$">, where  <IMG
    WIDTH="17" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img41.gif"
    ALT="$\mathsf{Q}$"> is orthogonal and
    <IMG
    WIDTH="14" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img42.gif"
    ALT="$\mathsf{L}$"> is lower triangular. The <I>QL</I> algorithm
    consist of a 
    sequence of orthogonal transformations
    <BR><P></P>
    <DIV ALIGN="CENTER">

    <!-- MATH
    \begin{displaymath}
    \mathsf{C}_s = \mathsf{Q}_s \cdot \mathsf{L}_s
    \end{displaymath}
    -->
    

    <IMG
    WIDTH="89" HEIGHT="29" BORDER="0"
    SRC="gif/principal_img44.gif"
    ALT="\begin{displaymath}
    \mathsf{C}_s = \mathsf{Q}_s \cdot \mathsf{L}_s
    \end{displaymath}">
    </DIV>
    <BR CLEAR="ALL">
    <P></P>
    <BR><P></P>
    <DIV ALIGN="CENTER">
    
    <!-- MATH
    \begin{displaymath}
    \mathsf{C}_{s+1} = \mathsf{L}_s   \cdot \mathsf{Q}_s 
    = \mathsf{Q}_s^T \cdot \mathsf{C}_s \cdot \mathsf{Q}_s
    \end{displaymath}
    -->
    
	      
    <IMG
    WIDTH="215" HEIGHT="31" BORDER="0"
    SRC="gif/principal_img45.gif"
    ALT="\begin{displaymath}
    \mathsf{C}_{s+1} = \mathsf{L}_s \cdot \mathsf{Q}_s
    = \mathsf{Q}_s^T \cdot \mathsf{C}_s \cdot \mathsf{Q}_s
    \end{displaymath}">
    </DIV>
    <BR CLEAR="ALL">
    <P></P>
    (1) If <IMG
    WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img2.gif"
    ALT="$\mathsf{C}$"> have eigenvalues with different absolute value 
    <IMG
    WIDTH="25" HEIGHT="34" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img46.gif"
    ALT="$\vert l_i\vert$">,  then 
    <!-- MATH
    $\mathsf{C}_s \rightarrow$
    -->
    <IMG
    WIDTH="45" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img47.gif"
    ALT="$\mathsf{C}_s \rightarrow$">&nbsp;[lower triangular form] as
    
    <!-- MATH
    $s\rightarrow\infty$
    -->
    <IMG
    WIDTH="57" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img48.gif"
    ALT="$s\rightarrow\infty$">. The eigenvalues appear on the diagonal in
    increasing order of absolute magnitude. (2) If If <IMG
    WIDTH="16" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img2.gif"
    ALT="$\mathsf{C}$"> has an
    eigenvalue <IMG
    WIDTH="25" HEIGHT="34" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img46.gif"
    ALT="$\vert l_i\vert$"> of multiplicty of order <IMG
    WIDTH="13" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img49.gif"
    ALT="$p$">, 
    
    <!-- MATH
    $\mathsf{C}_s \rightarrow$
    -->
    <IMG
    WIDTH="45" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img47.gif"
    ALT="$\mathsf{C}_s \rightarrow$">&nbsp;[lower triangular form] as
    
    <!-- MATH
    $s\rightarrow\infty$
    -->
    <IMG
    WIDTH="57" HEIGHT="16" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img48.gif"
    ALT="$s\rightarrow\infty$">, except for a diagona block matrix of order <IMG
    WIDTH="13" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img49.gif"
    ALT="$p$">,
    whose eigenvalues 
    <!-- MATH
    $\rightarrow l_i$
    -->
    <IMG
    WIDTH="37" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img50.gif"
    ALT="$\rightarrow l_i$">.  
    <PRE>
  */ 
  // End_Html

  // See comments above on TPrincipal::MakeTridiagonal for more on
  // local changes. 

  TVectorD& d = fEigenValues;
  TVectorD& e = fOffDiagonal;
  TMatrixD& z = fEigenVectors;
  Int_t&    n = fNumberOfVariables;

  // It's convenient to renumber the e vector elements 
  for (Int_t i = 2; i <= n; i++) 
    e(i-1) = e(i);
  e(n) = 0;

  for (Int_t l = 1; l <= n; l++) {
    Int_t iter = 0;
    Int_t m    = 0;

    do {
      for (m = l; m <= n-1; m++) {
	// Look for a single small sub-diagonal element  to split the
	// matrix 
	Double_t dd = TMath::Abs(d(m)) + TMath::Abs(d(m+1));
	if ((Double_t)(TMath::Abs(e(m)) + dd) == dd)
	  break;
      }

      if (m != l) {
	if (iter++ == 30) {
	  Error("MakeEigenVectors","too many iterations\n");
	  return;
	}

	// Form shift
	Double_t g = (d(l+1) - d(l)) / (2 * e(l));
	Double_t r = TMath::Sqrt(g * g + 1);
	// This is d_m - k_s 
	g          = d(m) - d(l) + e(l) / (g + TMath::Sign(r,g));
	Double_t s = 1;
	Double_t c = 1;
	Double_t p = 0;
	Int_t i    = 0;
	for (i = m-1; i >= l; i--) {
	  // A plane rotation as in the original QL, followed by
	  // Givens rotations to restore tridiagonal form
          Double_t f = s * e(i); 
	  Double_t b = c * e(i);
	  r          = TMath::Sqrt(f*f + g*g);
	  e(i+1)     = r;

	  if (r == 0) {
	    // Recover from underflow
	    d(i + 1) -= p;
	    e(m)     =  0;
	    break;
	  }
	  s      = f / r;
	  c      = g / r;
	  g      = d(i+1) - p;
	  r      = (d(i) - g) * s + 2 * c * b;
	  p      = s * r;
	  d(i+1) = g + p;
	  g      = c * r - b;

	  for (Int_t k = 1; k <= n; k++) {
	    // Form Eigenvectors
	    f        = z(k,i+1);
	    z(k,i+1) = s * z(k,i) + c * f;
	    z(k,i)   = c * z(k,i) - s * f;
	  }
	}  // for (i = m)

	if (r == 0 && i >= l) 
	  continue;

	d(l) -= p;
	e(l)  = g;
	e(m)  = 0;
	
      } // if (m != l)
    } while (m != l);
  } // for (l = 0)
}

//____________________________________________________________________
void
TPrincipal::MakeHistograms(const char* name, Option_t *opt)
{
  // Make histograms of the result of the analysis. 
  // The option string say which histograms to create
  //      X         Histogram original data
  //      P         Histogram principal components corresponding to
  //                original data 
  //      D         Histogram the difference between the original data
  //                and the projection of principal unto a lower
  //                dimensional subspace (2D histograms)
  //      E         Histogram the eigenvalues
  //      S         Histogram the square of the residues 
  //                (see TPrincipal::SumOfSquareResidues)
  // The histograms will be named <name>_<type><number>, where <name>
  // is the first argument, <type> is one of X,P,D,E,S, and <number>
  // is the variable. 
  Bool_t makeX  = kFALSE;
  Bool_t makeD  = kFALSE;
  Bool_t makeP  = kFALSE;
  Bool_t makeE  = kFALSE;
  Bool_t makeS  = kFALSE;

  Int_t len     = strlen(opt);
  Int_t i,j,k;
  for (i = 0; i < len; i++) {
    switch (opt[i]) {
    case 'X':
    case 'x':
      makeX = kTRUE;
      break;
    case 'd':
    case 'D':
      makeD = kTRUE;
      break;
    case 'P':
    case 'p':
      makeP = kTRUE;
      break;
    case 'E':
    case 'e':
      makeE = kTRUE;
      break;
    case 's':
    case 'S':
      makeS = kTRUE;
      break;
    default:
      Warning("MakeHistograms","Unknown option: %c",opt[i]);
    }
  }

  // If no option was given, then exit gracefully
  if (!makeX && !makeD && !makeP && !makeE && !makeS)
    return;

  // If the list of histograms doesn't exist, create it. 
  if (!fHistograms) 
    fHistograms = new TList;

  // Don't create the histograms if they are already in the TList. 
  if (makeX && fHistograms->FindObject(Form("%s_x0",name))) 
    makeX = kFALSE;
  if (makeD && fHistograms->FindObject(Form("%s_d0",name))) 
    makeD = kFALSE;
  if (makeP && fHistograms->FindObject(Form("%s_p0",name))) 
    makeP = kFALSE;
  if (makeE && fHistograms->FindObject(Form("%s_e",name))) 
    makeE = kFALSE;
  if (makeS && fHistograms->FindObject(Form("%s_s",name))) 
    makeS = kFALSE;

  TH1F **hX  = 0;
  TH2F **hD  = 0;
  TH1F **hP  = 0;
  TH1F *hE   = 0;
  TH1F *hS   = 0;

  // Initialize the arrays of histograms needed 
  if (makeX)
    hX = new TH1F * [fNumberOfVariables]; 

  if (makeD)
    hD = new TH2F * [fNumberOfVariables]; 

  if (makeP)
    hP = new TH1F * [fNumberOfVariables]; 

  if (makeE){  
    hE = new TH1F(Form("%s_e",name), "Eigenvalues of Covariance matrix",
		  fNumberOfVariables,0,fNumberOfVariables);
    hE->SetXTitle("Eigenvalue");
    fHistograms->Add(hE);
  }
  
  if (makeS) {
    hS = new TH1F(Form("%s_s",name),"E_{N}",
		  fNumberOfVariables-1,1,fNumberOfVariables);
    hS->SetXTitle("N");
    hS->SetYTitle("#sum_{i=1}^{M} (x_{i} - x'_{N,i})^{2}");
    fHistograms->Add(hS);
  }
  
  // Initialize sub elements of the histogram arrays 
  for (i = 0; i < fNumberOfVariables; i++) {
    if (makeX) {
      // We allow 4 sigma spread in the original data in our
      // histogram. 
      Double_t xlowb  = fMeanValues(i+1) - 4 * fSigmas(i+1);
      Double_t xhighb = fMeanValues(i+1) + 4 * fSigmas(i+1);
      Int_t    xbins  = fNumberOfDataPoints/100;
      hX[i]           = new TH1F(Form("%s_x%d", name, i),
				 Form("Pattern space, variable %d", i),
				 xbins,xlowb,xhighb);
      hX[i]->SetXTitle(Form("x_{%d}",i));
      fHistograms->Add(hX[i]); 
    }
    
    if(makeD) {
      // The upper limit below is arbitrary!!!
      Double_t dlowb  = 0;
      Double_t dhighb = 20;
      Int_t    dbins  = fNumberOfDataPoints/100;
      hD[i]           = new TH2F(Form("%s_d%d", name, i),
				 Form("Distance from pattern to "
				      "feature space, variable %d", i),
				 dbins,dlowb,dhighb,
				 fNumberOfVariables-1,
				 1,
				 fNumberOfVariables);
      hD[i]->SetXTitle(Form("|x_{%d} - x'_{%d,N}|/#sigma_{%d}",i,i,i));
      hD[i]->SetYTitle("N");
      fHistograms->Add(hD[i]);
    }
    
    if(makeP) {
      // For some reason, the trace of the none-scaled matrix 
      // (see TPrincipal::MakeNormalised) should enter here. Taken
      // from LINTRA code. 
      Double_t plowb   = -10 * TMath::Sqrt(fEigenValues(i+1) * fTrace); 
      Double_t phighb  = -plowb;
      Int_t    pbins   = 100;
      hP[i]            = new TH1F(Form("%s_p%d", name, i),
				  Form("Feature space, variable %d", i),
				  pbins,plowb,phighb);
      hX[i]->SetXTitle(Form("p_{%d}",i));
      fHistograms->Add(hP[i]);
    }
    
    if (makeE)
      // The Eigenvector histogram is easy
      hE->Fill(i,fEigenValues(i+1));

  }
  for (i = 0; i < fNumberOfDataPoints; i++) {
    Double_t* x = 0;
    Double_t* p = new Double_t[fNumberOfVariables];
    Double_t* d = new Double_t[fNumberOfVariables];

    if (makeX||makeP||makeD||makeS)
      // update the original data histogram 
      x  = (Double_t*)(GetRow(i));

    if (makeP||makeD||makeS)
      // calculate the corresponding principal component
      X2P(x,p);

    if (makeD || makeS) {
      // Calculate the difference between the original data, and the
      // same project onto principal components, and then to a lower
      // dimensional sub-space
      for (j = fNumberOfVariables; j > 0; j--) {
	P2X(p,d,j);
      
	for (k = 0; k < fNumberOfVariables; k++) {
	  // We use the absolute value of the difference!
	  d[k] = x[k] - d[k];

	  if (makeS) 
	    hS->Fill(j,d[k]*d[k]);

	  if (makeD) {
	    d[k] = TMath::Abs(d[k]) / fSigmas(k+1);
	    (hD[k])->Fill(d[k],j);
	  }
	}
      }
    }
    
    if (makeX||makeP) {
      // If we are asked to make any of these histograms, we have to
      // go here
      for (j = 0; j < fNumberOfVariables; j++) {
	if (makeX)
	  (hX[j])->Fill(x[j]);      

	if (makeP)
	  (hP[j])->Fill(p[j]);
      }
    }
    
    // Clean up
    if (d)
      delete [] d;
    if (p)
      delete [] p;
  }

  // Normalize the residues
  if (makeS)
    hS->Scale(Double_t(1.)/fNumberOfDataPoints);
}

//____________________________________________________________________
void 
TPrincipal::MakeNormalised(void) 
{
  // PRIVATE METHOD: Normalize the covariance matrix
  
  Int_t i,j;
  for (i = 1; i <= fNumberOfVariables; i++) {
    fSigmas(i) = TMath::Sqrt(fCovarianceMatrix(i,i));
    if (fIsNormalised) 
      for (j = 1; j <= i; j++) 
	fCovarianceMatrix(i,j) /= (fSigmas(i) * fSigmas(j));
    
    fTrace += fCovarianceMatrix(i,i);
  }

  // Fill remaining parts of matrix, and scale. 
  for (i = 1; i <= fNumberOfVariables; i++)     
    for (j = 1; j <= i; j++) {
      fCovarianceMatrix(i,j) /= fTrace;
      fCovarianceMatrix(j,i) = fCovarianceMatrix(i,j); 
    }

}

//____________________________________________________________________
void
TPrincipal::MakeMethods(const Char_t* classname, Option_t *opt) 
{
  // Generate the file <classname>PCA.cxx which contains the
  // implementation of two methods:
  // 
  //    void <classname>::X2P(Double_t* x, Double* p)
  //    void <classname>::P2X(Double_t* p, Double* x, Int_t nTest)
  // 
  // which does the same as  TPrincipal::X2P and TPrincipal::P2X
  // respectivly. Please refer to these methods.
  // 
  // Further, the public static members:
  //
  //    Int_t    <classname>::fgNVariables
  //    Double_t <classname>::fgEigenValues[]
  //    Double_t <classname>::fgEigenVectors[]
  //    Double_t <classname>::fgMeanValues[]
  //    Double_t <classname>::fgSigmaValues[]
  // 
  // are initialized, and assumed to exist. The class declaration is
  // assumed to be in <classname>.h and assumed to be provided by the
  // user. 
  //
  // See TPrincipal::MakeRealCode for a list of options
  // 
  // The minimal class definition is:
  //
  //   class <classname> {
  //   public:
  //     static Int_t    fgNVariables;
  //     static Double_t fgEigenVectors[];
  //     static Double_t fgEigenValues[];
  //     static Double_t fgMeanValues[];
  //     static Double_t fgSigmaValues[];
  //     
  //     void X2P(Double_t* x, Double_t* p);
  //     void P2X(Double_t* p, Double_t* x, Int_t nTest);
  //   };
  // 
  // Whether the methods <classname>::X2P and <classname>::P2X should
  // be static or not, is up to the user. 

  
  MakeRealCode(Form("%sPCA.cxx", classname), classname, opt);
}


//____________________________________________________________________
void
TPrincipal::MakePrincipals(void) 
{
  // Perform the principal components analysis. 
  // This is done is several stages:
  // * Transform the covariance matrix into a tridiagonal matrix, using
  //   method Principal::MakeTridiagonal();
  // * Find the eigenvalues and vectors of the tridiagonal matrix,
  //   using the method Principal::MakeEigenVectors();
  
  // Normalize matrix
  MakeNormalised();

  // Tridiagonalize matrix
  MakeTridiagonal();

  // Make eigenvectors and -values
  MakeEigenVectors();

  // Order eigenvalues and -vectors
  MakeOrdered();
}

//____________________________________________________________________
void
TPrincipal::MakeOrdered(void)
{
  // Begin_Html
  /*
    </PRE>
    PRIVATE METHOD:
    <BR>
    Order the eigenvalues and vectors by ascending eigenvalue. The
    algorithm is a straight insertion. It's taken from 
    <A NAME="tex2html2"
    HREF="http://www.nr.com">Numerical Recipes in C</A>
    section 11.1.
    <PRE>
  */
  // End_Html
    
    Int_t i,j,k;
    for (i = 1; i <= fNumberOfVariables; i++) {
       k = i;
    Double_t p = fEigenValues(i);

    for (j = i + 1; j <= fNumberOfVariables; j++)
      if (fEigenValues(j) >= p) {
	k = j;
	p = fEigenValues(j);
      }
    
    if (k != i) {
      fEigenValues(k) = fEigenValues(i);
      fEigenValues(i) = p;
      
      for (j = 1; j <= fNumberOfVariables; j++) {
	p                  = fEigenVectors(j,i);
	fEigenVectors(j,i) = fEigenVectors(j,k);
	fEigenVectors(j,k) = p;
      }
    }
  }   
}

//____________________________________________________________________
void
TPrincipal::MakeRealCode(const Char_t* filename, const Char_t* classname,
			 Option_t *opt) 
{
  // PRIVATE METHOD:
  // This is the method that actually generates the code for the
  // transformations to and from feature space and pattern space 
  // It's called by TPrincipal::MakeCode and TPrincipal::MakeMethods. 
  //  
  // The options are: NONE so far

  Bool_t  isMethod = (classname[0] == '\0' ? kFALSE : kTRUE);
  Char_t* prefix   = (isMethod ? Form("%s::", classname) : "");
  Char_t* cv_qual  = (isMethod ? "" : "static ");

  ofstream outFile(filename,ios::out|ios::trunc); 
  if (!outFile) {
    Error("MakeRealCode","couldn't open output file '%s'",filename); 
    return;
  }
  
  cout << "Writing on file \"" << filename << "\" ... " << flush;
  //
  // Write header of file 
  //
  // Emacs mode line ;-)
  outFile << "// -*- mode: c++ -*-" << endl;
  // Info about creator
  outFile << "// " << endl
	  << "// File " << filename
	  << " generated by TPrincipal::MakeCode" << endl;
  // Time stamp
  TDatime date;
  outFile << "// on " << date.AsString() << endl;
  // ROOT version info
  outFile << "// ROOT version " << gROOT->GetVersion() 
	  << endl << "//" << endl;
  // General information on the code
  outFile << "// This file contains the functions " << endl
	  << "//" << endl
	  << "//    void  " << prefix 
	  << "X2P(Double_t* x, Double_t* p); " << endl 
	  << "//    void  " << prefix 
	  << "P2X(Double_t* p, Double_t* x, Int_t nTest);" 
	  << endl << "//" << endl
	  << "// The first for transforming original data x in " << endl
	  << "// pattern space, to principal components p in " << endl
	  << "// feature space. The second function is for the" << endl
	  << "// inverse transformation, but using only nTest" << endl
	  << "// of the principal components in the expansion" << endl
	  << "// " << endl
	  << "// See TPrincipal class documentation for more "
	  << "information " << endl << "// " << endl;
  // Header files
  outFile << "#ifndef __CINT__" << endl;
  if (isMethod) 
    // If these are methods, we need the class header 
    outFile << "#include \"" << classname << ".h\"" << endl;
  else 
    // otherwise, we need the typedefs of Int_t and Double_t
    outFile << "#include <Rtypes.h> // needed for Double_t etc" << endl;
  // Finish the preprocessor block
  outFile << "#endif" << endl << endl;

  // 
  // Now for the data
  //
  // We make the Eigenvector matrix, Eigenvalue vector, Sigma vector,
  // and Mean value vector static, since all are needed in both
  // functions. Also ,the number of variables are stored in a static
  // variable. 
  outFile << "//" << endl 
	  << "// Static data variables"  << endl
	  << "//" << endl;
  outFile << cv_qual << "Int_t    " << prefix << "fgNVariables = " 
	  << fNumberOfVariables << ";" << endl;

  // Assign the values to the Eigenvector matrix. The elements are
  // stored row-wise, that is element 
  //    M[i][j] = e[i * nVariables + j]
  // where i and j are zero-based.
  outFile << endl << "// Assignment of eigenvector matrix." << endl
	  << "// Elements are stored row-wise, that is" << endl
	  << "//    M[i][j] = e[i * nVariables + j] " << endl
	  << "// where i and j are zero-based" << endl;
  outFile << cv_qual << "Double_t " << prefix 
	  << "fgEigenVectors[] = {" << flush;
  Int_t i,j;
  for (i = 0; i < fNumberOfVariables; i++) {
    for (j = 0; j < fNumberOfVariables; j++) {
      Int_t index = i * fNumberOfVariables + j;
      outFile << (index != 0 ? "," : "" ) << endl
	      << "  "  << fEigenVectors(i+1,j+1) << flush; 
    }
  }
  outFile << "};" << endl << endl;

  // Assignment to eigenvalue vector. Zero-based.
  outFile << "// Assignment to eigen value vector. Zero-based." << endl;
  outFile << cv_qual << "Double_t " << prefix 
	  << "fgEigenValues[] = {" << flush;
  for (i = 0; i < fNumberOfVariables; i++) 
    outFile << (i != 0 ? "," : "") << endl
	    << "  " << fEigenValues(i+1) << flush;
  outFile << endl << "};" << endl << endl;

  // Assignment to mean Values vector. Zero-based.
  outFile << "// Assignment to mean value vector. Zero-based." << endl;
  outFile << cv_qual << "Double_t " << prefix 
	  << "fgMeanValues[] = {" << flush;
  for (i = 0; i < fNumberOfVariables; i++) 
    outFile << (i != 0 ? "," : "") << endl
	    << "  " << fMeanValues(i+1) << flush;
  outFile << endl << "};" << endl << endl;

  // Assignment to mean Values vector. Zero-based.
  outFile << "// Assignment to sigma value vector. Zero-based." << endl;
  outFile << cv_qual << "Double_t " << prefix 
	  << "fgSigmaValues[] = {" << flush;
  for (i = 0; i < fNumberOfVariables; i++) 
    outFile << (i != 0 ? "," : "") << endl
	    << "  " << fSigmas(i+1) << flush;
  outFile << endl << "};" << endl << endl;

  // 
  // Finally we reach the functions themselves
  // 
  // First: void x2p(Double_t* x, Double_t* p);
  //
  outFile << "// " << endl
	  << "// The " 
    	  << (isMethod ? "method " : "function ")
	  << "  void " << prefix 
	  << "X2P(Double_t* x, Double_t* p)" 
	  << endl << "// " << endl;
  outFile << "void " << prefix 
	  << "X2P(Double_t* x, Double_t* p) {" << endl
	  << "  for (Int_t i = 0; i < fgNVariables; i++) {" << endl
	  << "    p[i] = 0;" << endl
	  << "    for (Int_t j = 0; j < fgNVariables; j++)" << endl
	  << "      p[i] += (x[j] - fgMeanValues[j]) " << endl
	  << "        * fgEigenVectors[j *  fgNVariables + i] " 
	  << "/ fgSigmaValues[j];" << endl << endl << "  }" 
	  << endl << "}" << endl << endl;
  // 
  // Now: void p2x(Double_t* p, Double_t* x, Int_t nTest);
  // 
  outFile << "// " << endl << "// The " 
	  << (isMethod ? "method " : "function ")
	  << "  void " << prefix 
	  << "P2X(Double_t* p, Double_t* x, Int_t nTest)" 
	  << endl << "// " << endl;
  outFile << "void " << prefix 
	  << "P2X(Double_t* p, Double_t* x, Int_t nTest) {" << endl
	  << "  for (Int_t i = 0; i < fgNVariables; i++) {" << endl
	  << "    x[i] = fgMeanValues[i];" << endl
	  << "    for (Int_t j = 0; j < nTest; j++)" << endl
	  << "      x[i] = p[j] * fgSigmaValues[i] " << endl
	  << "        * fgEigenValues[j *  fgNVariables + i];" << endl
	  << "  }" << endl << "}" << endl << endl;
  
  // EOF
  outFile << "// EOF for " << filename << endl;

  // Close the file 
  outFile.close();

  cout << "done" << endl;
}

//____________________________________________________________________
void 
TPrincipal::MakeTridiagonal(void)
{
  // Begin_Html
  /*
    </PRE>
    PRIVATE METHOD:
    <BR>
    Tridiagonalise the covariance matrix according to the Householder
    method as described in 
    <A NAME="tex2html3"
    HREF="http://www.nr.com">Numerical Recipes in C</A>
    section&nbsp;11.2.
    
    <P>
    The basic idea is to perform <IMG
    WIDTH="48" HEIGHT="32" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img51.gif"
    ALT="$P-2$"> orthogonal transformation, where
    each transformation eat away the off-diagonal elements, except the
    inner most. 
    <PRE>
   */
  // End_Html

  // The comments in this algorithm are modified version of those in
  // "Numerical ...". Please refer to that book (web-page) for more on
  // the algorithm. 
  // Notice that we store the diagonal elements of the tridiagonal
  // matrix directly in fEigenVectors, hence no vector d is declared,
  // only a reference. Also, we may use the fEigenValues vector as
  // temporary storage space for the off-diagonal elements. 

  // Copy covariance matrix to temporary matrix, so that we leave
  // the covariance matrix intact;
  fEigenVectors = fCovarianceMatrix;
  TMatrixD& a   = fEigenVectors;
  TVectorD& d   = fEigenValues;
  TVectorD& e   = fOffDiagonal;
  Int_t&    n   = fNumberOfVariables;

  Double_t hh, g, f;
  Int_t i,j,k;
  
  for (i = n; i >= 2; i--) {
    Int_t    l     = i - 1;
    Double_t h     = 0;
    Double_t scale = 0;
    
    if (l > 1) {
      for (k = 1; k <= l; k++) 
	scale += TMath::Abs(a(i,k));
	
      if (scale == 0)
	// Skip transformation
	e(i) = a(i,l);

      else {
	for (k = 1; k <= l; k++) {
	  // Use scaled elements of a for transformation
	  a(i,k) /= scale;
	  // Calculate sigma in h
	  h += a(i,k) * a(i,k);
	}

	f      =  a(i,l);
	g      =  (f >= 0. ? -TMath::Sqrt(h) : TMath::Sqrt(h));
	e(i)   =  scale * g;
	h      -= f * g; // Now h is eq. (11.2.4) in "Numerical ..."
	a(i,l) =  f - g;
	f      = 0;

	for (j = 1; j <= l; j++) {
	  // Store the u/H in ith column of a;
	  a(j,i) = a(i,j) / h;
	  // Form element A dot u in g;
	  g = 0;
	  
	  for (k = 1; k <= j; k++) 
	    g += a(j,k) * a(i, k);
	 
	  for (k = j + 1; k <= l; k++) 
	    g += a(k,j) * a(i, k);

	  // Form element of vector p in temporarily unused element of
	  // e 
	  e(j) =  g / h;
	  f    += e(j) * a(i,j);
	}

	// Form K eq (11.2.11)
	hh = f / (h + h);

	// Form vector q and store in e overwriting p
	for (j = 1; j <= l; j++) {
	  f    = a(i,j);
	  e(j) = g = e(j) - hh * f;

	  for (k = 1; k <= j; k++) 
	    // Reduce a, eq (11.2.13)
	    a(j,k) -= (f * e(k) + g * a(i,k));
	}
      }
    }
    else
      e(i) = a(i,l);

    d(i) = h;
  }
  
  d(1) = 0;
  e(1) = 0;

  for (i = 1; i <= n; i++) {
    // Begin accumulation of transformation matrix
    Int_t l = i - 1;

    if (d(i)) {
      // This block is skipped if i = 1;
      for (j = 1; j <= l; j++) {
	g = 0;
	
	for (k = 1; k <= l; k++) 
	  // Use vector u/H stored in a to form P dot Q
	  g += a(i,k) * a(k,j);

	for (k = 1; k <= l; k++)
	  a(k,j) -= g * a(k,i);
      }
    }

    d(i)   = a(i,i);
    a(i,i) = 1;
    
    for (j = 1; j <= l; j++) {
      a(j,i) = a(i,j) = 0;
    }
  }

}

//____________________________________________________________________
void
TPrincipal::P2X(Double_t* p, Double_t* x, Int_t nTest)
{
  // Calculate x as a function of nTest of the most significant
  // principal components p, and return it in x.  
  // It's the users responsibility to make sure that both x and p are
  // of the right size (i.e., memory must be allocated for x). 

  for (Int_t i = 1; i <= fNumberOfVariables; i++){
    x[i-1] = fMeanValues(i);
    for (Int_t j = 1; j <= nTest; j++)
      x[i-1] += p[j-1] * fSigmas(i) * fEigenVectors(j,i);
  }

}

//____________________________________________________________________
void
TPrincipal::Print(Option_t* opt)
{
  // Print the statistics
  // Options are 
  //      M            Print mean values of original data 
  //      S            Print sigma values of original data 
  //      E            Print eigenvalues of covarinace matrix
  //      V            Print eigenvectors of covarinace matrix
  // Default is MSE

  Bool_t printV = kFALSE;
  Bool_t printM = kFALSE;
  Bool_t printS = kFALSE;
  Bool_t printE = kFALSE;
  
  Int_t len     = strlen(opt);
  for (Int_t i = 0; i < len; i++) {
    switch (opt[i]) {
    case 'V':
    case 'v':
      printV = kTRUE;
      break;
    case 'M':
    case 'm':
      printM = kTRUE;
      break;
    case 'S':
    case 's':
      printS = kTRUE;
      break;
    case 'E':
    case 'e':
      printE = kTRUE;
      break;
    default:
      Warning("Print", "Unknown option '%c'",opt[i]);
      break;
    }
  }
  
  if (printM||printS||printE) {
    cout << " Variable #  " << flush; 
    if (printM)
      cout << "| Mean Value " << flush;
    if (printS)
      cout << "|   Sigma    " << flush;
    if (printE)
      cout << "| Eigenvalue" << flush;
    cout << endl;

    cout << "-------------" << flush;
    if (printM)
      cout << "+------------" << flush;
    if (printS)
      cout << "+------------" << flush;
    if (printE)
      cout << "+------------" << flush;
    cout << endl;

    for (Int_t i = 1; i <= fNumberOfVariables; i++) {      
      cout << setw(12) << i << " " << flush;
      if (printM) 
	cout << "| " << setw(10) << setprecision(4) 
	     << fMeanValues(i) << " " << flush; 
      if (printS) 
	cout << "| " << setw(10) << setprecision(4) 
	     << fSigmas(i) << " " << flush; 
      if (printE) 
	cout << "| " << setw(10) << setprecision(4) 
	     << fEigenValues(i) << " " << flush; 
      cout << endl;
    }
    cout << endl;
  }
  
  if(printV) {
    for (Int_t i = 1; i <= fNumberOfVariables; i++) {
      cout << "Eigenvector # " << i << flush;
      TVectorD v(1,fNumberOfVariables);
      v = TMatrixDColumn(fEigenVectors,i);
      v.Print();
    }
  }
}

//____________________________________________________________________
void
TPrincipal::SumOfSquareResiduals(Double_t* x, Double_t* s)
{
  // PRIVATE METHOD: 
  // Begin_html
  /*
    </PRE>
    Calculates the sum of the square residuals, that is 
    <BR><P></P>
    <DIV ALIGN="CENTER">
    
    <!-- MATH
    \begin{displaymath}
    E_N = \sum_{i=1}^P \left(x_i - x^\prime_i\right)^2
    \end{displaymath}
    -->
    
    
    <IMG
    WIDTH="147" HEIGHT="58" BORDER="0"
    SRC="gif/principal_img52.gif"
    ALT="\begin{displaymath}
    E_N = \sum_{i=1}^P \left(x_i - x^\prime_i\right)^2
    \end{displaymath}">
    </DIV>
    <BR CLEAR="ALL">
    <P></P>
    where 
    <!-- MATH
    $x^\prime_i = \sum_{j=i}^N p_i e_{n_j}$
    -->
    <IMG
    WIDTH="122" HEIGHT="40" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img53.gif"
    ALT="$x^\prime_i = \sum_{j=i}^N p_i e_{n_j}$">, <IMG
    WIDTH="19" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img54.gif"
    ALT="$p_i$"> is the
    <IMG
    WIDTH="28" HEIGHT="23" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img55.gif"
    ALT="$i^{\mbox{th}}$"> component of the principal vector, corresponding to
    <IMG
    WIDTH="20" HEIGHT="30" ALIGN="MIDDLE" BORDER="0"
    SRC="gif/principal_img56.gif"
    ALT="$x_i$">, the original data; I.e., the square distance to the space
    spanned by <IMG
    WIDTH="20" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
    SRC="gif/principal_img12.gif"
    ALT="$N$"> eigenvectors. 
    <BR>
    <PRE>
  */
  // End_Html
  if (!x) 
    return;

  Double_t p[100];
  Double_t xp[100];

  X2P(x,p);
  for (Int_t i = fNumberOfVariables; i > 0; i--) {
    P2X(p,xp,i);
    for (Int_t j = 0; j < fNumberOfVariables; j++) {
      s[i] += (x[j] - xp[j])*(x[j] - xp[j]);
    }
  }
}

//____________________________________________________________________
void
TPrincipal::Test(Option_t* opt) 
{
  // Test the PCA, bye calculating the sum square of residuals 
  // (see method SumOfSquareResiduals), and display the histogram 
  MakeHistograms("pca","S");

  TH1 *pca_s = 0;
  if (fHistograms) pca_s = (TH1*)fHistograms->FindObject("pca_s");
  if (!pca_s) {
    Warning("Test", "Couldn't get histogram of square residuals"); 
    return;
  }
  
  pca_s->Draw();
}
      
//____________________________________________________________________
void
TPrincipal::X2P(Double_t *x, Double_t *p)
{
  // Calculate the principal components from the original data vector
  // x, and return it in p. 
  // It's the users responsibility to make sure that both x and p are
  // of the right size (i.e., memory must be allocated for p). 
  for (Int_t i = 1; i <= fNumberOfVariables; i++){
    p[i-1] = 0;
    for (Int_t j = 1; j <= fNumberOfVariables; j++)
      p[i-1] += (x[j-1] - fMeanValues(j)) * fEigenVectors(j,i) /
	fSigmas(j);
  }
  
}
