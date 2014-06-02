grep -v -e 'unused class rule' -e 'no dictionary for class pair' | sed -e "s|small_aod/MAKEP|MAKEP|g" | sed -e "s|Shared lib small_aod/small_aod.so|small_aod.so|g"
