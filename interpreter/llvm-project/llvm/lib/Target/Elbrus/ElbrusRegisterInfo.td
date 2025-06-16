//=== ElbrusRegisterInfo.td - Describe the Elbrus Register File - tablegen -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Elbrus Register file, defining the registers themselves,
// aliases between the registers, and the register classes built out of the
// registers.
//
//===----------------------------------------------------------------------===//

class E2KReg<string n> : Register<n> {
  let Namespace = "Elbrus";
}

//===----------------------------------------------------------------------===//
//  Register definitions...
//

def R0 : E2KReg<"r0">;

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

def GR32 : RegisterClass<"Elbrus", [i32], 32,
                         (add R0)>;

def GR64 : RegisterClass<"Elbrus", [i64], 64,
                         (add R0)>;

