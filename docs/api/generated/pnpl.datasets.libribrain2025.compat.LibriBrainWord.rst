pnpl.datasets.libribrain2025.compat.LibriBrainWord
==================================================

.. currentmodule:: pnpl.datasets.libribrain2025.compat

.. autoclass:: LibriBrainWord

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~LibriBrainWord.__init__
      ~LibriBrainWord.calculate_standardization_params
      ~LibriBrainWord.clip_sample
      ~LibriBrainWord.close_h5_files
      ~LibriBrainWord.ensure_file
      ~LibriBrainWord.ensure_file_download
      ~LibriBrainWord.get_bids_raw_path
      ~LibriBrainWord.get_calibration_files
      ~LibriBrainWord.get_derivatives_path
      ~LibriBrainWord.get_events_path
      ~LibriBrainWord.get_h5_dataset
      ~LibriBrainWord.get_h5_path
      ~LibriBrainWord.get_headpos_path
      ~LibriBrainWord.get_preprocessed_path
      ~LibriBrainWord.get_sfreq_from_h5
      ~LibriBrainWord.init_continuous_h5
      ~LibriBrainWord.load_continuous_window
      ~LibriBrainWord.load_continuous_window_from_sample
      ~LibriBrainWord.load_head_positions
      ~LibriBrainWord.load_preprocessed_bids
      ~LibriBrainWord.load_raw_bids
      ~LibriBrainWord.prefetch_files
      ~LibriBrainWord.raw_bids_exists
      ~LibriBrainWord.setup_standardization
      ~LibriBrainWord.standardize
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~LibriBrainWord.HUGGINGFACE_FALLBACK_REPOS
      ~LibriBrainWord.HUGGINGFACE_REPO
      ~LibriBrainWord.broadcasted_means
      ~LibriBrainWord.broadcasted_stds
      ~LibriBrainWord.channel_means
      ~LibriBrainWord.channel_stds
      ~LibriBrainWord.label_info
      ~LibriBrainWord.n_channels
      ~LibriBrainWord.n_times
   
   