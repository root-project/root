sed -e 's,^(vector,(class vector,' -e 's,::iterator)[x[:xdigit:]]*$$,::iterator),' -e 's/const_iterator/class vector<unsigned int::const_iterator/' -e 's, @0x[a-fA-F0-9]*,,' \
