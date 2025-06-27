# Histogram Terminology

This document collects, defines, and explains terms that are used in ROOT's histogram package.
The goal is to start from a common understanding, which should avoid ambiguities and ease discussions.
It also helps (future) developers to navigate the code because classes and methods are named accordingly.
The list is ordered alphabetically, though dependent terms are kept together with their parent.
It is supposed to be exhaustive; any missing term should be added when needed.

An *axis* is a bin configuration in one dimension.
A *regular axis* has equidistant bins in the interval $[a, b)$.
A *variable bin axis* is configured with explicit bin edges $[e_{n}, e_{n+1})$.
A *categorical axis* has a unique label per bin.
*Axes* is the plural of axis and usually means the bin configurations for all dimensions of a histogram.

A *bin content* is the value of a single bin.
The *bin content type* can be an integer type, a floating-point type, the special `RDoubleBinWithError`, or a user-defined type.

A *bin error* is the Poisson error of a bin content.
With the special `RDoubleBinWithError`, it is the square root of the sum of weights squared: $\sqrt{\sum w_i^2}$
Otherwise it is the square root of the bin content, which is only correct with unweighted filling.

A *bin index* (plural *indices*) refers to a single bin of a dimension, an array of indices refers to a bin in a histogram.
A *normal bin* is inside an axis and its index starts from 0.
*Underflow* and *overflow* bins, also called *flow bins*, are outside the axis and their index has a special value.
The *invalid bin index* is another special value.

A *bin index range* is a range from `begin` (inclusive) to `end` (exclusive).
For its purpose, the underflow bin is ordered before all normal bins while the overflow bin is placed after.
As the `end` is exclusive, the invalid bin index is ordered last to make it possible to include the overflow bin.

*Filling* a histogram means to add an entry to a histogram.
*Concurrent filling* allows to modify the same histogram without (external) synchronization.

A *histogram* is the combination of an axes configuration and storage of bin contents.
For most use cases, it also includes (global) *histogram statistics*.
On the one hand, these are the number of entries, the sum of weights, and the sum of weights squared.
The number of *effective entries* can be computed as the ratio $$\frac{(\sum w_i)^2}{\sum w_i^2}$$.
Furthermore, for each dimension the histogram statistics include the sum of weights times value and the sum of weights times value squared.
This allows to compute the arithmetic mean and the standard deviation of the values before binning.

A *linearized index* starts from 0 up to the total number of bins, potentially including flow bins.
For a single axis, it places the flow bins after the normal bins.
The *global index* is a combination of the linearized indices from all axes.

A *profile* is a histogram that computes the arithmetic mean and standard deviation per bin.
During filling, it accepts an additional `double` value and accumulates its sum and sum of squares.

*Slicing* means to extract a subset of the normal bins in each dimension.
Bin contents of excluded normal bins are added to the flow bins.

A *snapshot* is a consistent clone of the histogram during concurrent filling.

A *weight* is an optional floating-point value passed during filling.
It defaults to $1$ if not specified, which is also called unweighted filling.
