grep -v 'WARNING: IO rule for class' | sed -e "s|cms310/MAKEP|MAKEP|g" | sed -e "s|Shared lib cms310/cms310.so|cms310.so|g" | grep -v -e 'unused class rule' -e 'no dictionary for class pair'
