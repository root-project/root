grep -v -e 'unused class rule' -e 'no dictionary for class pair' | sed -e "s|aliceesd/MAKEP|MAKEP|g" | sed -e "s|Shared lib aliceesd/aliceesd.so|aliceesd.so|g"
