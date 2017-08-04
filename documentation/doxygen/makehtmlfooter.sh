#!/bin/sh

# Generates the HTML footer

echo '<html>'
echo '<body>'
echo '<div id="footer" style="background-color:#E5EBF3;">'
echo '<small>'
echo '<img class="footer" src="rootlogo_s.gif" alt="root"/></a>'
# Doxygen unconditionally adds a space in front of $DOXYGEN_ROOT_VERSION
echo 'ROOT'$DOXYGEN_ROOT_VERSION' - Reference Guide Generated on $datetime (GVA Time) using Doxygen '`doxygen --version`'.'
echo '</small>'
echo '</div>'
echo '</body>'
echo '</html>'
