grep -v 'dot -' \
| grep -v '^$' \
| grep -v 'Checking for Graphviz' \
| grep -v 'dot: not found' \
| grep -v 'dot: command not found' \
| grep -v 'dot version' \
| grep -v 'RooFit' \
| grep -v 'Copyright' \
| grep -v 'roofit.sourceforge' \
| grep -v 'no dictionary for' \
| grep -v 't open file' \
| grep -v ' scanning ' \
| grep -v ' Times-Roman' \
| grep -v 'ldap_initialize() failed:' \
| grep -v 'ldap_sasl_bind_s() failed:' \
| grep -v '_Index.html'
