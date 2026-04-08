
.. automodule:: plspy
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
===========

Core PLS
--------

plspy.pls
~~~~~~~~~

.. automodule:: plspy.pls
   :members:
   :undoc-members:
   :show-inheritance:

plspy.pls\_classes
~~~~~~~~~~~~~~~~~~

*Options*
""""""""
- :class:`Mean-Centred Task PLS <plspy.pls_classes._MeanCentreTaskPLS>`
- :class:`Behavioural PLS <plspy.pls_classes._RegularBehaviourPLS>`
- :class:`Contrast Task PLS <plspy.pls_classes._ContrastTaskPLS>`
- :class:`Contrast Behavioural PLS <plspy.pls_classes._ContrastBehaviourPLS>`
- :class:`Multiblock PLS <plspy.pls_classes._MultiblockPLS>`
- :class:`Contrast Multiblock PLS <plspy.pls_classes._ContrastMultiblockPLS>`

.. automodule:: plspy.pls_classes
   :private-members:
   :members: _MeanCentreTaskPLS, _RegularBehaviourPLS, _ContrastTaskPLS,  _ContrastBehaviourPLS, _MultiblockPLS, _ContrastMultiblockPLS
   :undoc-members:
   :show-inheritance:


Resampling
----------

plspy.bootstrap\_permutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: plspy.bootstrap_permutation
   :private-members:
   :members: _ResampleTestPLS
   :undoc-members:
   :show-inheritance:

plspy.resample
~~~~~~~~~~~~~~

.. automodule:: plspy.resample
   :members: resample_with_replacement, resample_without_replacement
   :undoc-members:
   :show-inheritance:


plspy.split_half_resampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: plspy.split_half_resampling
   :members: split_half, split_half_test_train
   :undoc-members:
   :show-inheritance:


Visualization
-------------
Provides classes to visualize PLS results. 

*Available plot types:*

- Singular Values Plot
- Permuted Singular Values Plot
- Design LV Plot
- Design Scores Plot
- Task PLS Brain Score Plot
- Correlation Plot
- Behaviour LV Plot
- Brain Scores vs Behaviour Plot


.. automodule:: plspy.visualize.visualize
   :members: visualize_classes
   :undoc-members:
   :show-inheritance:


Utilities
---------
plspy.check\_inputs
~~~~~~~~~~~~~~~~~~~

.. automodule:: plspy.check_inputs
   :members:
   :undoc-members:
   :show-inheritance:

plspy.decorators
~~~~~~~~~~~~~~~~

.. automodule:: plspy.decorators
   :members:
   :undoc-members:
   :show-inheritance:

plspy.exceptions
~~~~~~~~~~~~~~~~

.. automodule:: plspy.exceptions
   :members:
   :undoc-members:
   :show-inheritance:


