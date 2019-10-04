#!/bin/sh

# Generates the HTML footer

echo '<div id="nav-path" class="navpath">'
echo '<ul>'
echo '$navpath'
echo '<li class="footer">'
echo 'ROOT'$DOXYGEN_ROOT_VERSION' - Reference Guide Generated on $datetime (GVA Time) using Doxygen $doxygenversion &#160;&#160;'
echo '<img class="footer" src="rootlogo_s.gif" alt="root"/>'
echo '</li></ul>'
echo '</div>'
echo '<!--BEGIN !GENERATE_TREEVIEW-->'
echo '<hr class="footer"/><address class="footer"><small>'
echo 'ROOT'$DOXYGEN_ROOT_VERSION' - Reference Guide Generated on $datetime (GVA Time) using Doxygen $doxygenversion &#160;&#160;'
echo '<img class="footer" src="rootlogo_s.gif" alt="root"/>'
echo '</small></address>'
echo '<!--END !GENERATE_TREEVIEW-->'
echo '</body>'
echo '</html>'
