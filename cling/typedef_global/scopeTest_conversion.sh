grep -v Processing | sed -e s/myclbad.so/mycl.so/ -e s/10/12/ -e s/::ConstLink/ConstLink/ | sed -E "s,-?[0-9]+,N/A,"
