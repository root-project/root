\defgroup Python Python Interface
\ingroup Python
\brief Python bindings and utilities for ROOT.


ROOT is a C++ framework used across HEP for data storage, analysis and visualisation. Its full API is available directly in Python through dynamic bindings powered by [cppyy](https://cppyy.readthedocs.io/). Every ROOT class you see in the
C++ documentation is accessible from Python under the `ROOT` module.

On top of that, a set of [pythonizations](@ref Pythonizations) adapt selected classes to feel more natively Pythonic: operator overloading, iterators, NumPy interoperability, and more.


# Installation

\htmlonly
<div class="install-tabs">
  <div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab(this, 'conda')">conda</button>
    <button class="tab-btn" onclick="switchTab(this, 'pip')">pip</button>
  </div>
  <div id="conda" class="tab-panel active">
    <pre><code>conda install -c conda-forge root</code></pre>
  </div>
  <div id="pip" class="tab-panel" style="display:none;">
    <pre><code>pip install root</code></pre>
    <p style="margin:6px 12px 10px;font-size:12px;color:#b45309;background:#fffbeb;
              border:1px solid #fcd34d;border-radius:4px;padding:6px 10px;">
      ⚠ Alpha - Linux only.
    </p>
  </div>
</div>

<style>
.install-tabs {
  border: 1px solid var(--page-foreground-color, #ccc);
  border-radius: 6px;
  overflow: hidden;
  max-width: 420px;
  font-family: monospace;
}
.tab-buttons {
  display: flex;
  background: var(--code-background, #f0f0f0);
  border-bottom: 1px solid var(--page-foreground-color, #ccc);
}
.tab-btn {
  padding: 6px 18px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  color: var(--page-foreground-color, #333);
  border-bottom: 2px solid transparent;
}
.tab-btn.active {
  border-bottom: 2px solid #1a73e8;
  font-weight: 600;
  color: #1a73e8;
}
.tab-panel pre {
  margin: 0;
  padding: 12px 16px;
  background: var(--code-background, #fff);
  color: var(--page-foreground-color, #333);
}
</style>

<script>
function switchTab(btn, id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  btn.classList.add('active');
  document.getElementById(id).style.display = 'block';
}
</script>
\endhtmlonly

See <a href="https://root.cern/install" style="color:#b45309;">root.cern/install</a> for all installation options.

# Quickstart

The entry point to ROOT in Python is one import:

~~~{.py}
import ROOT
~~~

Every ROOT class and function is available under the `ROOT` module.

Now let's create a histogram, fill it from a [NumPy array](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html) and write it to a file:

~~~{.py}
import numpy as np

# Create a 1D histogram
h = ROOT.TH1D("h", "Gaussian distribution;x;counts", 100, -5, 5)

# Fill it from a NumPy array
data = np.random.normal(0, 1, 10000)
h.Fill(data)

# Write it to a ROOT file
with ROOT.TFile.Open("output.root", "RECREATE") as f:
    h.Write()
~~~

Now we create an RDataFrame from scratch, define a new column with a Python lambda and draw a histogram:

~~~{.py}
import numpy as np

# Create an RDataFrame with 10000 rows
rdf = ROOT.RDataFrame(10000)

# Define a column x
rdf = rdf.Define("x", lambda : np.random.normal(0, 1))

# Draw a histogram of x
rdf.Histo1D("x").Draw()
~~~
