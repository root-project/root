#!/bin/sh
#
# Generate the ROOT User's Guide in PDF format.

oxygendir="/Applications/oxygen"

docbookdirs="/usr/share/sgml/docbook/xsl-stylesheets-1.75.2 \
             /usr/share/xml/docbook/stylesheet/docbook-xsl \
             /sw/share/xml/xsl/docbook-xsl"

fopjars="/usr/share/java/fop.jar \
         /sw/share/java/fop/fop.jar"

if [ -d "$oxygendir" ]; then
   docbookdirs="$oxygendir/frameworks/docbook/xsl"
   fopjars="$oxygendir/lib/fop.jar"
fi

for d in $docbookdirs; do
   if [ -d "$d" ]; then
      docbook=$d
   fi
done

if [ -z "$docbook" ]; then
   echo "$0: no docbook installation found"
   exit 1
fi

if `which fop > /dev/null 2>& 1`; then
   fop=`which fop`
else
   for f in $fopjars; do
      if [ -f "$f" ]; then
         fopjar=$f
      fi
   done

   if [ -z "$fopjar" ]; then
      echo "$0: no fop.jar file found"
      exit 1
   fi
fi

# for more printed output options see:
# http://xml.web.cern.ch/XML/www.sagehill.net/xml/docbookxsl/PrintOutput.html

xsltproc --xinclude --output ROOTUsersGuide.fo \
   --stringparam paper.type A4 \
   $docbook/fo/docbook.xsl \
   ROOTUsersGuide.xml

if [ -x "$fop" ]; then
   $fop ROOTUsersGuide.fo ROOTUsersGuide.pdf
else
   java -Xmx1024m -jar $fopjar ROOTUsersGuide.fo ROOTUsersGuide.pdf
fi

rm ROOTUsersGuide.fo

exit 0
