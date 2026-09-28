# Author: ROOT Team / PyROOT modernization
#
################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import asyncio
import sys
import unittest
import ROOT


class AsyncioCooperativePumping(unittest.TestCase):
    """
    Test cooperative integration between Python's asyncio event loop
    and ROOT's event processing (ROOT.gSystem.ProcessEvents()).
    """

    def setUp(self):
        ROOT.EnableThreadSafety()

    def test_cooperative_event_pumping(self):
        """
        Verify that an asyncio event loop can cooperatively pump ROOT events
        alongside asynchronous background tasks without blocking.
        """
        processed_events_count = 0
        iterations_completed = 0

        async def root_event_pump(stop_event):
            nonlocal processed_events_count
            while not stop_event.is_set():
                ROOT.gSystem.ProcessEvents()
                processed_events_count += 1
                await asyncio.sleep(0.005)

        async def async_worker(stop_event):
            nonlocal iterations_completed
            for _ in range(5):
                await asyncio.sleep(0.01)
                iterations_completed += 1
            stop_event.set()

        async def main():
            stop_event = asyncio.Event()
            await asyncio.gather(root_event_pump(stop_event), async_worker(stop_event))

        asyncio.run(main())
        self.assertEqual(iterations_completed, 5)
        self.assertGreater(processed_events_count, 0)


class AsyncioThreadOffloading(unittest.TestCase):
    """
    Test offloading ROOT compute-heavy tasks into thread workers
    via asyncio.to_thread and loop.run_in_executor.
    """

    def setUp(self):
        ROOT.EnableThreadSafety()

    def test_asyncio_to_thread_th1(self):
        """
        Verify offloading TH1 creation and fill operations to worker threads
        using asyncio.to_thread.
        """

        def create_and_fill_hist(name, entries):
            h = ROOT.TH1F(name, f"Hist {name}", 50, -5, 5)
            for i in range(entries):
                h.Fill(i % 5 - 2)
            return h.GetEntries(), h.GetMean()

        async def main():
            res1, res2 = await asyncio.gather(
                asyncio.to_thread(create_and_fill_hist, "h_async_1", 100),
                asyncio.to_thread(create_and_fill_hist, "h_async_2", 200),
            )
            return res1, res2

        (entries1, mean1), (entries2, mean2) = asyncio.run(main())
        self.assertEqual(entries1, 100)
        self.assertEqual(entries2, 200)

    def test_asyncio_to_thread_tmath(self):
        """Test concurrent evaluation of TMath special functions in asyncio worker threads."""

        def compute_bessel(order, val):
            return ROOT.TMath.BesselI(order, val)

        async def main():
            tasks = [asyncio.to_thread(compute_bessel, i, 2.5) for i in range(4)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(main())
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIsInstance(r, float)


class AsyncioTaskGroup(unittest.TestCase):
    """
    Test Python 3.11+ structured concurrency with asyncio.TaskGroup.
    """

    def setUp(self):
        ROOT.EnableThreadSafety()

    @unittest.skipIf(sys.version_info < (3, 11), "asyncio.TaskGroup requires Python 3.11+")
    def test_task_group_concurrent_root_execution(self):
        """
        Verify that multiple concurrent ROOT operations managed by
        an asyncio.TaskGroup execute successfully and aggregate results.
        """

        def compute_integral(xmin, xmax):
            f = ROOT.TF1("f_async", "x*x + 2*x", -10, 10)
            return f.Integral(xmin, xmax)

        async def main():
            results = []
            async with asyncio.TaskGroup() as tg:
                t1 = tg.create_task(asyncio.to_thread(compute_integral, 0.0, 2.0))
                t2 = tg.create_task(asyncio.to_thread(compute_integral, 1.0, 3.0))

            results.append(t1.result())
            results.append(t2.result())
            return results

        integrals = asyncio.run(main())
        self.assertEqual(len(integrals), 2)
        # Integral of x^2 + 2x from 0 to 2 is [x^3/3 + x^2]_0^2 = 8/3 + 4 = 6.6666...
        self.assertAlmostEqual(integrals[0], 20.0 / 3.0, places=4)

    def test_asyncio_task_cancellation(self):
        """
        Verify that cancelling an in-flight asyncio task executing a ROOT operation
        raises asyncio.CancelledError and cleans up cleanly without leaking locks.
        """

        async def slow_root_task():
            await asyncio.sleep(2.0)
            return ROOT.TMath.Pi()

        async def main():
            task = asyncio.create_task(slow_root_task())
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
