#  @author Giovanni Petrucciani
#  @date 2024-12

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import time

class TaskProgressBar:
    def __init__(self, tasks, pollInterval=0.5):
        self._tasks = list(tasks)
        self._pollInterval = pollInterval
    def run(self) -> None:
        if not self._tasks:
            return
        try:
            import tqdm
            self._run_tqdm(tqdm.tqdm)
        except ImportError:
            self._run_basic()
        self._tasks.clear()
    def _run_tqdm(self, tqdm) -> None:
        total = len(self._tasks)
        with tqdm(total=total) as pbar:
            waited = 0
            last = 0
            while True:
                ndone = sum(f.done() for f in self._tasks)
                waited += 1
                if (ndone != last) or (waited > 10):
                    pbar.update(ndone-last)
                    last = ndone
                    waited = 0
                if ndone == total:
                    break
                time.sleep(self._pollInterval)
    def _run_basic(self) -> None:
        total = len(self._tasks)
        started = time.monotonic()
        waited = 0
        last = 0
        interval = int(max(1, 2/self._pollInterval)) # one dot every 2 s
        while True:
            ndone = sum(f.done() for f in self._tasks)
            waited += 1
            now = time.monotonic()
            if ndone != last:
                print(f"after {now-started:.0f}s, {ndone}/{total} done ({ndone/total*100:3.0}%%)", 
                      end = ("\n" if ndone == total else ""),
                      flush=True)
                last = ndone
                waited = 0
                if ndone == total:
                    break
            elif waited == interval:
                print(".", end="", flush=True)
                waited = 0
                if now - started > 10*60:
                    interval = int(max(1, 2*60/self._pollInterval)) # one dot every 2 mins
                elif now - started > 60:
                    interval = int(max(1, 10/self._pollInterval)) # one dot every 10 s
            time.sleep(self._pollInterval)
