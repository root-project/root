/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2019, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "RooFit/MultiProcess/Messenger.h"
#include <RooFit/MultiProcess/ProcessManager.h>  // ... JobManager::process_manager()

#include "gtest/gtest.h"

TEST(TestMPMessenger, Connections)
{
   RooFit::MultiProcess::ProcessManager pm(2);
   RooFit::MultiProcess::Messenger messenger(pm);
   messenger.test_connections(pm);
}
