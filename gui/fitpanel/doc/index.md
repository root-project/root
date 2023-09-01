\defgroup fitpanel ROOT Fit Panel
\ingroup gui
\brief Classes forming the user interface of the Fit Panel in ROOT.


## The Fit Panel

\image html fitpanel.png

To display the Fit Panel right click on a histogram to pop up the
context menu, and then select the menu entry Fit Panel.

By design, this user interface is planned to contain two tabs:
"General" and "Minimization". Currently, the "General" tab provides
user interface elements for setting the fit function, fit method and
different fit, draw, print options.
The "Minimization tab" provides the option to set the Minimizer to use in the fit and
its specific options.

The fit panel is a modeless dialog, i.e. when opened, it does not
prevent users from interacting with other windows. Its first prototype
is a singleton application. When the Fit Panel is activated, users can
select an object for fitting in the usual way, i.e. by left-mouse
click on it. If the selected object is suitable for fitting, the fit
panel is connected with this object and users can perform fits by
setting different parameters and options.

### Function Choice and Settings


*‘Predefined' combo box* - contains a list of predefined functions in
ROOT. You have a choice of several polynomials, a Gaussian, a Landau,
and an Exponential function. The default one is Gaussian.

*‘Operation' radio button group* defines the selected operational mode
between functions:

*Nop* - no operation (default);

*Add* - addition;

*Conv* - convolution (will be implemented in the future).

Users can enter the function expression into the text entry field
below the ‘Predefined' combo box. The entered string is checked after
the Enter key was pressed and an error message shows up, if the
function string is not accepted.

‘*Set Parameters*' button opens a dialog for parameters settings,
which will be explained later.

### Fitter Settings


*‘Method' combo box* currently provides only two fit model choices:
Chi-square and Binned Likelihood. The default one is Chi-square. The
Binned Likelihood is recommended for bins with low statistics.

*‘Linear Fit' check button* sets the use of Linear fitter when is
selected. Otherwise the minimization is done by Minuit, i.e. fit
option "`F`" is applied. The Linear fitter can be selected only for
functions linear in parameters (for example - `polN)`.

*‘Robust' number entry* sets the robust value when fitting graphs.

*‘No Chi-square' check button* switch On/Off the fit option "`C`" -
do not calculate Chi-square (for Linear fitter).

*‘Integral' check button* switch On/Off the option "`I`" - use
integral of function instead of value in bin center.

*‘Best Errors'* sets On/Off the option "`E`" - better errors
estimation by using Minos technique.

*‘All weights = 1'* sets On/Off the option "`W`"- all weights set to 1
excluding empty bins; error bars ignored.

*‘Empty bins, weights=1'* sets On/Off the option "`WW`" - all weights
equal to 1 including empty bins; error bars ignored.

*‘Use range'* sets On/Off the option "`R`" - fit only data within the
specified function range. Sliders settings are used if this option is
set to On. Users can change the function range values by pressing the
left mouse button near to the left/right slider edges. It is possible
to change both values simultaneously by pressing the left mouse button
near to the slider center and moving it to a new position.

*‘Improve fit results'* sets On/Off the option "`M`"- after minimum is
found, search for a new one.

*‘Add to list'* sets On/Off the option "`+`"- add function to the list
without deleting the previous one. When fitting a histogram, the
function is attached to the histogram's list of functions. By default,
the previously fitted function is deleted and replaced with the most
recent one, so the list only contains one function. Setting this
option to On will add the newly fitted function to the existing list
of functions for the histogram. Note that the fitted functions are
saved with the histogram when it is written to a ROOT file. By
default, the function is drawn on the pad displaying the histogram.

### Draw Options


*‘SAME'* sets On/Off function drawing on the same pad. When a fit is
executed, the image of the function is drawn on the current pad.

*‘No drawing'* sets On/Off the option "`0`"- do not draw the fit
results.

*‘Do not store/draw'* sets On/Off option "`N`"- do not store the
function and do not draw it.

### Advances Options

The advance option button is enabled only after having performed the fit and provides
additional drawing options that can be used after having done the fit. These new drawing tools,
which can be selected by the "Advanced Drawing Tool"  panel that pops up when clicking the "Advanced" button, are:

* *Contour*: to plot the confidence contour of two chosen parameters. One can select the number of points to draw the contour
(more points might require more time to compute it), the parameters and the desired confidence level .

* *Scan* : to plot a scan of the minimization function (likelihood or chi-squared) around the minimum as function of the chosen parameter.

* *Conf Interval* : to plot the confidence interval of the fitted function as a filled coloured band around its central value.
   One can select the desired confidence level for the band to be plotted.

### Print Options


This set of options specifies the amount of feedback printed on the
root command line after performed fits.

*‘Verbose'* - prints fit results after each iteration.

*‘Quiet'* - no fit information is printed.

*‘Default'* - between Verbose and Quiet.

### Command Buttons


*Fit button* - performs a fit taking different option settings via the
Fit Panel interface.

*Reset* - sets the GUI elements and related fit settings to the
default ones.

*Close* - closes the Fit panel window.

### Minimization Options

With this tab one can select specific options for minimization. These include

*  The minimizer library ( *Minuit*, *Minuit2*, *Fumili*, *GSL*, *Genetics* )
*  The method (algorithm) for minimization. For example for Minuit one can choose between (*Migrad*, *Simplex* or *Scan*)
*  Error definition
*  Minimization tolerance
*  Number of iterations/function calls
*  Print Level: (*Default*, *Verbose* or *Quiet*).

