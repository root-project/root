## Math Libraries


### Minuit2

-   Remove the TFitterMinuit class and the similar ones used to implement the `TVirtualFitter` interface using Minuit2. users should switch to use the
    `ROOT::Math::Minimizer` interface


All other changes in the Math packages have been applied also in the 5.34 patched versions of ROOT. See their release notes for the detailed list of applied improvements.

