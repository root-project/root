#!/bin/sh
#
# Generate the ROOT User's Guide in ePub format.

oxygendir="/Applications/oxygen"

docbookdirs="/usr/share/sgml/docbook/xsl-stylesheets-1.75.2 \
             /usr/share/xml/docbook/stylesheet/docbook-xsl \
             /sw/share/xml/xsl/docbook-xsl"

if [ -d "$oxygendir" ]; then
   docbookdirs="$oxygendir/frameworks/docbook/xsl"
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

dbtoepub=$docbook/epub/bin/dbtoepub

if [ ! -x "$dbtoepub" ]; then
   if [ ! -f "$dbtoepub" ]; then
      echo "$0: $dbtoepub: not found."
      exit 1
   else
      ruby=ruby
   fi
fi

$ruby $dbtoepub --output ROOTUsersGuide.epub ROOTUsersGuide.xml

exit 0
