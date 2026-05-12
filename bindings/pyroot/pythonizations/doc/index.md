\defgroup Python Python Interface
\ingroup Python
\brief Python bindings and utilities for ROOT.


ROOT is a C++ framework used across HEP for data storage, analysis and visualisation. Its full API is available directly in Python through dynamic bindings powered by [cppyy](https://cppyy.readthedocs.io/). Every ROOT class you see in the
C++ documentation is accessible from Python under the `ROOT` module.

On top of that, a set of **[**pythonizations**](@ref Pythonizations)** adapt selected classes to feel more natively Pythonic: operator overloading, iterators, NumPy interoperability, and more.


# Installation

\htmlonly
<div class="install-tabs">
  <div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab(this, 'pip')">pip</button>
    <button class="tab-btn" onclick="switchTab(this, 'conda')">conda</button>
  </div>
  <div id="pip" class="tab-panel active">
    <pre><code>pip install root</code></pre>
  </div>
  <div id="conda" class="tab-panel" style="display:none;">
    <pre><code>conda install -c conda-forge root</code></pre>
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

# Quickstart

Open a ROOT file, apply a filter, draw a histogram:

~~~{.py}
import ROOT

df = ROOT.RDataFrame("events", "file.root")
df.Filter("pt > 20").Histo1D("px").Draw()
~~~

Read a histogram directly from a file:

~~~{.py}
with ROOT.TFile.Open("file.root") as f:
    h = f.Get("my_histogram")
    h.Draw()
~~~
