# Generates the HTML footer

echo '<html>'
echo '</body>'
echo '<div id="footer" style="background-color:#DDDDDD;">'
echo '<small>'
echo '<img class="footer" src="rootlogo_s.gif" alt="root"/></a>'
echo 'ROOT ('$DOXYGEN_ROOT_VERSION' ) Reference Guide Generated on $datetime.'
echo '</small>'
echo '</div>'
echo '</body>'
echo '</html>'