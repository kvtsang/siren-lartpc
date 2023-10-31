.. slar documentation master file, created by
   sphinx-quickstart on Wed Oct 26 19:52:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to slar's documentation!
===========================================
`slar` is implementation of `siren <https://www.vincentsitzmann.com/siren/>`_ for modeling physics of photon (signal) transport in Liquid Argon Time Projection Chamber (LArTPC) experiments.  In particular, this include siren and optimization scripts for modeling N-dimensional input to a floatin point feature.

For the installation and tutorial notebooks, please see the `software repository <https://github.com/CIDeR-ML/siren-lartpc>`_.


Getting started
---------------

You can find a quick guide to get started below.

Install ``slar``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/cider-ml/siren-lartpc
   cd siren-lartpc
   pip install . --user


You can install to your system path by omitting ``--user`` flag. 
If you used ``--user`` flag to install in your personal space, assuming ``$HOME``, you may have to export ``$PATH`` environment variable to find executables.

.. code-block:: bash
   
   export PATH=$HOME/.local/bin:$PATH


.


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :glob:

   slar <slar>

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`