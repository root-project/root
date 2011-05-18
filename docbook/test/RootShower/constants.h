// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/


#ifndef CONSTANTS_H
#define CONSTANTS_H

#define MAX_PARTICLE  100000
#define EMass         5.100e-04
#define CSpeed        29979245.8      // in cm/ms
#define Prop_B        5.26e-03        // in GeV/c pro 17.4 kGauss
#define MAX_GAMMA     1000000000000.

#define classic_electr_radius   2.8179409421853486e-12
#define electron_Compton_length 3.861593288718701e-10
#define fine_structure_const    0.00729735301337329
#define hbarc                   1.9732705406375647e-10

#define DEAD         -1
#define ALIVE         0
#define CREATED       1

// PDG code :
#define PHOTON              22  // Photon
#define ELECTRON            11  // Electron
#define POSITRON           -11  // Positron
#define MUON_MINUS          13  // Muon-
#define MUON_PLUS          -13  // Muon+
#define TAU_MINUS           15  // Tau-
#define TAU_PLUS           -15  // Tau+
#define NEUTRINO_E          12  // Neutrino e
#define ANTINEUTRINO_E     -12  // Anti Neutrino e
#define NEUTRINO_MUON       14  // Neutrino Muon
#define ANTINEUTRINO_MUON  -14  // Anti Neutrino Muon
#define NEUTRINO_TAU        16  // Neutrino Tau
#define ANTINEUTRINO_TAU   -16  // Anti Neutrino Tau
#define PION_ZERO          111  // Pion0
#define PION_PLUS          211  // Pion+
#define PION_MINUS        -211  // Pion-
#define ETA                221  // Eta
#define KAON_PLUS          321  // Kaon+
#define KAON_MINUS        -321  // Kaon-
#define KAON_LONG          130  // Kaon Long
#define KAON_SHORT         310  // Kaon Short
#define D_PLUS            -411  // D+
#define D_MINUS            411  // D-
#define D_ZERO             421  // D0
#define D_ZERO_BAR        -421  // D0 bar
#define PROTON            2212  // Proton
#define ANTIPROTON       -2212  // Anti Proton
#define NEUTRON           2112  // Neutron
#define ANTINEUTRON      -2112  // Anti Neutron

#define W_PLUS              24  // W+
#define W_MINUS            -24  // W-
#define Z_ZERO              23  // Z0
#define RHO_ZERO           113  // RHO0
#define RHO_PLUS           213  // RHO+
#define RHO_MINUS         -213  // RHO-
#define OMEGA              223  // OMEG
#define PHI                333  // PHI
#define J_PSI              443  // JPSI

#define LAMBDA            3122  // Lambda
#define LAMBDA_BAR       -3122  // Lambda bar
#define SIGMA_PLUS        3222  // Sigma+
#define SIGMA_BAR_MINUS  -3222  // Sigma bar -
#define SIGMA_ZERO        3212  // Sigma0
#define SIGMA_BAR_ZERO   -3212  // Sigma bar 0
#define SIGMA_MINUS       3112  // Sigma-
#define SIGMA_BAR_PLUS   -3112  // Sigma bar +
#define XI_ZERO           3322  // Xi0
#define XI_BAR_ZERO      -3322  // Xi bar 0
#define XI_MINUS          3312  // Xi-
#define XI_BAR_PLUS      -3312  // Xi bar +

#define Fe            0
#define Ne            1
#define Al            2
#define Pb            3
#define Polystyrene   4
#define BGO           5
#define CsI           6
#define NaI           7

#define UNDEFINE      0
#define STABLE        1
#define BREMS         2
#define CONVERSION    3
#define INTERACT      4
#define DECAY         5
#define PCUT          0.1

#endif // CONSTANTS_H
