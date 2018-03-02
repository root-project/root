#!/bin/bash

HTMLPATH=$DOXYGEN_OUTPUT_DIRECTORY/html

sed -i'.back' -e 's/Collaboration diagram for /Libraries for /g'  $HTMLPATH/class$1.html
rm $HTMLPATH/class$1.html.back

# Picture name containing the "coll graph"
PICNAME=$HTMLPATH/"class"$1"__coll__graph.svg"

# Find the libraries for the class $1
root -l -b -q "libs.C(\"$1\")"

if [[ ! -f libslist.dot ]] ; then
   exit
fi

sed -i'.back' -e "s/\.so.*$/\";/" libslist.dot
rm libslist.dot.back

# Generate the dot file describing the graph for libraries
echo "digraph G {" > libraries.dot
echo "   rankdir=LR;" >>  libraries.dot
echo "   node [shape=box, fontname=Arial];" >>  libraries.dot
cat mainlib.dot >> libraries.dot;
cat libslist.dot | grep -v "\.dylib" \
                 | grep lib[A-Z] | sed -e "s/\(.*\)\(lib.*;\)$/   mainlib->\"\2/" \
                 >> libraries.dot
echo "   mainlib [shape=box, fillcolor=\"#ABACBA\", style=filled];" >>  libraries.dot
echo "}" >> libraries.dot

# Generate the SVG image of the graph
dot -Tsvg libraries.dot -o $PICNAME

# Make sure the picture size in the html file the same as the svg
PICSIZE=`grep "svg width" $PICNAME | sed -e "s/<svg //"`
sed -i'.back' -e "s/\(^.*src\)\(.*__coll__graph.svg\"\)\( width.*\">\)\(.*$\)/<div class=\"center\"><img src\2 $PICSIZE><\/div>/"  $HTMLPATH/class$1.html
rm $HTMLPATH/class$1.html.back
