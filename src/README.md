# PviML
## Background
Separated from original PVI image reconstruction project. Only concerns with machine learning pipeline.

## Utility classes and functions
- `primitives.py`: defines frequently used constants, default objects, parameters, etc. for the entire project.
For example, `input_mode` for PVI data (images, bioz, etc.), `output_mode` for BP data (full waveform, systolic & diastolic),
and the sequence length `mask_key` are defined here.
- `h5io.py`: contains functions related to hdf5 write/read and transformation.
- `miscellaneous.py`: contains other helper functions that I don't know how to categorize.

## Core pipeline (dataset preparation  &rarr; training)

### Dataset discovery, loading, and transformation
- `ProjectPathManager`: sets up root project path + create directories for export. Can also detect the root path when
trained on CHPC clusters.

- `PviDatasetInventory`: surveys the available HDF5 datasets in the dataset directory (specified by ProjectPathManager).
Includes methods to filter datasets based on keywords in their names. 

- `PviRawDataset`: interfaces with the HDF5 files; only handles extracting and slicing raw tensors from individual *.h5 files.
First stage of the data pipeline. Does not transform/reshape the tensors.

- `PviConfiguredDataset`: inherits from `torch.utils.data.Dataset`.
Abstract base class to transform/reshape the tensors, specified by `input_mode` (pvi images or impedance signals);
`output_mode` (full bp waveform or only systolic and diastolic); and sequence length (1, 5, 10, or 15 consecutive periods).
Also declares `build` and `validate_build` protocols so the datasets can be used in training loop. Cannot be used directly.
Its subclasses are:
  
  - `PviSingleDataset`: Second stage in the data pipeline. Builds a ready-to-train dataset from a `PviRawDataset`.
  - `PviCompositeDataset`: Combines/stacks multiple `PviSingleDataset` and `PviCompositeDataset`. This is useful when
preparing a large dataset from multiple subjects. Can be used to compose a population dataset (if there is enough memory)
  - `PviLazyDataset`: Subclass of `PviConfiguredDataset` that can load data on-demand (during training). Has a different
interface than `Single` and `Composite` datasets. The `build` method does not actually build a complete dataset on memory,
rather loads their metadata and masks, and keeps track of the sample-source reference. Has a cache with
[LRU eviction policy](https://en.wikipedia.org/wiki/Page_replacement_algorithm#Least_recently_used).
Overrides the `__getitem__` method to slice data directly from h5 files and/or from cache.

### Dataset partition and sampling
- `GraphBipartitePartitioner`: partition the dataset sequences into train and test intervals without overlaps.
See [appendix C](#partition) for more discussion.
- `PviBatchSampler`: custom sampler to be plugged into argument `batch_sampler` of `torch.utils.data.DataLoader`. Used in
conjunction with `PviLazyDataset` to balance cache efficiency with batch diversity. See [appendix D](#sampler) for more
discussion.
  
### Neural network architectures
- `BasePviLearner`: abstract base class that provides a common interface to parse the data shapes and deciding on the
size of input and output layer. Also declares some protocol for the subclasses to implement, e.g. `_make_layers` and
`forward`. All models must inherit from this base class to use this interface. To implement the specific architecture,
e.g. linear regression, MLP, the models just need to override `_make_layers` and `forward` method.

  - `PviLinearRegression`: 1x fully connected layer mapping input &rarr; output, with pass-through (linear) activation.

  - `PviMLP`: 7x fully connected layers (input projection &rarr; 5x hidden &rarr; output prediction). The hidden layers have
  200 neurons each, with ReLU activation. Input and output layers have pass-through (linear) activation.

  - `PviCNN`: 3x ConvBlock (Conv &rarr; BatchNorm &rarr; MaxPool) &rarr; 2x fully connected. Convolution applies both
spatially and temporally, while pooling only applies on spatial dimension.

  - `PviCNNTransformer`: 2x ConvBlock (see above) &rarr; 1x Bi-LSTM &rarr; Transformer Encoder &rarr; 2x fully connected

  - `PviMamba`: work in progress

### Training utilities
- `TrainingWorkflow`: Main coordinator of the entire training progress. Encapsulates the classes listed below and only
interact with the model/dataset through them.
- `ModelTrainer`: Handles passing the data samples through the model during train/eval cycle, for a single epoch.
See [appendix E](#metrics) for some discussion of its methods.
- `EarlyStopCounter`: Implement an [early stopping](https://en.wikipedia.org/wiki/Early_stopping) policy based on a
target score. Can set activation threshold, reset threshold, and stopping patience.
- `MetricsTracker`: Records the relevant metrics (loss and bp_accuracy) through all epochs
- `TrainingCheckpoint`: Store model states + dataset partition states as *.pth files, so training can be resumed after
unexpected termination (e.g. CHPC timeout/preemption, or workstation shutdown/restart).
- `TrainingParamsLogger`: Compiles the training hyperparameters and the general model architecture, and exportss as *.json
files.

---
## <a name="workflow"></a> Typical workflow of a training session

The following steps give a high-level overview of the complete pipeline, from loading the h5 files, to configuring the datasets,
to setting up the training components (optimizer, loss, etc.), to initiating and running the training loop, and finally
exporting the results.

*Note 1*: See the scripts in `src/scripts` for specific examples on how this pipeline is used.

*Note 2*: Detailed implementation and more comments are found in the source code.

1. Call `ProjectPathManager` and set project `root` path if not using default. Also set `logdir` to export all artifacts
related to a training session.
2. Call `PviDatasetInventory` to define the dataset directory. Most of the time these stays the same for all training
sessions.
3. Build the dataset for training. There are several ways, depending on dataset mode:
   * Single h5 dataset: This is the simplest dataset form, with all samples contained in a single *.h5 file.
   To build the dataset, call `PviRawDataset` to load the h5 file with matching name, then call `PviSingleDataset` to
   configure the dataset's `input_mode`, `output_mode`, and `mask_key`. This configuration determines the input and
   output shape and helps the model decide the first and last layer.
   * Composite dataset: Contains samples from multiple *.h5 file.
   To build this type of dataset, first build multiple instances of `PviSingleDataset` following the procedure above. To
   be more selective on which h5 files to use, call `PviDatasetInventory.filter()` to get a subset of h5 files
   with matching keywords, e.g. all files from `subject001`, or all files from the `valsalva` experiments.
   Then call `PviCompositeDataset` to combine them into a larger dataset. All constituent `PviSingleDataset` must have
   the same configuration (`input_mode`, `output_mode`, `mask_key`), while `PviCompositeDataset` will take care of global
   indexing and make sure the data tensors are stored on a homogeneous memory block.
   * Lazy dataset: This dataset has the simplest interface (because everything is encapsulated in its implementation).
   To build this dataset, call `PviLazyDataset` and specify the inventory directory and configuration.
   During the build phase, the dataset will survey all *.h5 files found in the directory and construct a dictionary to map
   each sample to its source h5 file. During training, the lazy dataset will call `PviRawDataset` and `PviSingleDataset`,
   and/or slice the tensors directly from h5 file.

4. Set partition hyperparameters. Avoid splitting the dataset here. The actual split will take place in
`TrainingWorkflow.initiate_training()`, before training begins.
5. Set batch hyperparameters. Avoid calling `DataLoader` here. The `DataLoader` will be created in
`TrainingWorkflow.initiate_training()`, before training begins.
6. Select an existing model.
7. Select an optimizer, and wrap it around the models' parameters.
8. Select a loss function.
9. Select a learning-rate scheduler (Optional).
10. Call `EarlyStopCounter` and define the stopping hyperparameters.
11. Call `TrainingWorkflow`, and wrap everything as workflow arguments.
    The workflow object will call `ModelTrainer`, `MetricsTracker`, `TrainingCheckpoint` and `TrainingParamsLogger`. These
    objects do not have tunable hyperparameters and are thus encapsulated in the workflow class.
12. Call `TrainingWorkflow.initiate_training()`. The workflow will search for matching checkpoints in `logdir` and try
to resume.
    * If there is no matching checkpoint (or the checkpoint cannot be loaded), the dataset will be split into train and
    test subsets based on partition hyperparameters.
    * Otherwise, the dataset, model, optimizer, scheduler, stopping counter, and training history will be restored to
    The train/set partition state will also be restored, so the dataset will not be split again.
    * This method also calls `DataLoader` to create train and test loaders, based on batch hyperparameters. To allow for
    flexible sampling strategy, this step will always be called and is independent of checkpoint.
13. Call `TrainingWorkflow.run()`. Resume training (or training from scratch if no checkpoint is used).
    For inspection purposes, this method exports intermediate artifacts (history, inference results, and checkpoints).
    Currently, the export schedule is every 50 epochs, every 3 hours, or everytime a new high score is recorded in
    `EarlyStopCounter`.

---
## Appendices
### Appendix A: Project structure
**Note**: Only `src` is pushed to GitHub. Everything else are stored on local machines (with backups).
```
PviProject                  # as of 2025/07/31
├── src                     # this only part stored on github, containing entire codebase
    ├── ...                 # browse the current github page for details
├── setup.py
├── run_batch.sbatch        # for deployment on CHPC clusters
├── datasets                # main dataset folder
    ├── _holdout            # 5 subjects for validation
        ├── *_masked.h5
        ├── ...
    ├── *_masked.h5
    ├── ...
├── artifacts               # exported files from multiple ML training sessions
    ├── {ML session name}
        ├── checkpoints
            ├── *.pth       # training state (model weights, dataset partition, etc.)
        ├── configs
            ├── *.json      # tunable settings and hyperparameters
        ├── history
            ├── *.csv       # train and test metrics throughout epochs
        ├── results
            ├── *.csv       # inference results on test dataset
    ├── ...
```
### <a name="hdf5"></a> Appendix B: HDF5 naming convention and hierarchy
Overall, there are about 100 subjects. Each subject participates in one or more experiment sessions. Their data are
included in multiple *.h5 files, with the naming convention `{subjectID}_{experiment}_masked.h5`. For example, subject 1
participated in 3 experiments: baseline, valsalva, and cold pressor. So there are 3 files associated with subject 1:
* `subject001_baseline_masked.h5`
* `subject001_valsalva_masked.h5`
* `subject001_pressor_masked.h5`

On the other hand, subject 7 only participated in the baseline experiment, so the associated h5 files are:

* `subject007_baseline_masked.h5`
* ~~`subject007_valsalva_masked.h5`~~
* ~~`subject007_pressor_masked.h5`~~


As of 2025/07/31, the hierarchy of the hdf5 datasets (version 0.1.2) is as follows:
```
*_masked.h5                 # as of 2025/07/31
├── build
    ├── author
    ├── date
    ├── version
├── metadata
    ├── subject             # e.g. 'subject001', 'subject002',..., 'subject100'
    ├── session             # 'baseline', 'valsalva', 'pressor'
    ├── num_periods         # also denoted as N. Also counts uncleaned periods
    ├── period_length       # also denoted as T (typically T=50)
├── data
    ├── bp                  # blood pressure data group
        ├── signal          # shape: (1, T*N)
    ├── pviHP               # high-pass component of PVI data
        ├── img             # shape: (H, W, T*N), typically H=W=40
        ├── reactance       # shape: (Cr, T*N), typically Cr=32
        ├── resistance      # shape: (Cx, T*N), typically Cx=32
        ├── signal          # shape: (1, T*N)
    ├── pviLP               # low-pass component of PVI data
        ├── img             # shape: (H, W, T*N)
        ├── reactance       # shape: (Cr, T*N)
        ├── resistance      # shape: (Cx, T*N)
        ├── signal          # shape: (1, T*N)
├── masks                   # list of 2-tuples, containing period bounds of clean consecutive sequences
    ├── mask01              # shape: (M01, 2). In principle, 0 <= M01 <= N
    ├── mask05              # shape: (M05, 2). In principle, 0 <= M05 < M01
    ├── mask10              # shape: (M10, 2). In principle, 0 <= M10 < M05
    ├── mask15              # shape: (M15, 2). In principle, 0 <= M15 < M10
├── shapes                  # same hierarchy as 'data' group, containing the intended tensor shapes 
    ...
├── stats                   # temporal features. Does not include pviLP and bp
    ├── pviHP
        ├── duration        # shape: (1, N), duration of a full period (in seconds)
        ├── tMax            # shape: (1, N), peak time (in seconds)
```
#### Some more remarks about the h5 files:

* Unless stated otherwise, all numerical values are `float64` (`double`), and all string values are encoded in `utf-8`.
* The files were exported from `MATLAB`, with no chunkings and no compression.
* `MATLAB` tensors use column-major order (Fortran-style), while `h5py` reads data in row-major order (C/Pascal-style).
As such, before exporting, the tensor dimensions were already reversed to ensure the tensors read into `numpy` have the
intended shape. The expected shapes are described in the `shapes` group.
* The masks point to period indices, NOT frame/datapoint indices. When slicing along the time dimension, `period_length`
must be taken into account.
* The masks use 1-based index, with both ends being inclusive. When slicing in Python, the first index (but not the second)
should be offset by `-1`, e.g. `(3, 7) -> (2, 7)`.

### <a name="partition"></a> Appendix C: `GraphBipartitePartitioner`
To capture long-term variations in the sequences, we stitch multiple consecutive periods and use as an input sample.
These sequences are extracted with a sliding window along the full signal, excluding the portions that failed the
signal quality assessment (SQA). The valid (or clean) sequences are described by the `masks`. In the example below,
there are no sequences of 5 **clean** and **consecutive** periods between period `7` to `14`. It is possible that they
all failed SQA. But it is also possible that some of them passed SQA, but are scattered and disjoint.
```
mask05 = [(0, 5), (1, 6), (2, 7), (15, 20), (18, 23), (32, 37), ... ]
```
We then split the dataset (using the masks as proxy) into train and test sequences. If we just randomly pick some masks
as test sequences, it is very likely to have overlapping periods between train and test sequences. For example, if we
take the sequences `(1,6)` and `(18, 23)` as test samples and use the rest as train samples, we have periods
`1, 2, 3, 4, 5, 6, 18, 19` leaked from the test sequences into the trained sequences.

Alternatively, we can divide the data into disjoint train and test segments, e.g.
```
# for a dataset with 1000 total periods (including ones that failed SQA)
train_segments = [(0,200), (350, 400), (500, 600), (800, 1000)]
test_segments = [(200,350), (400, 500), (600, 800)]
```
and then pick the clean sequences (represented with the masks) that are fully contained within those segments. But this
approach inadvertently create locally biased samples. We thus need a way to truly randomize the partition, while
satisfying the following two constraints:
1. NO overlaps among train/test sequences of multiple consecutive periods;
2. Consistent train/test ratio as desired.

Considering only the first objective, the problem is straightforward: we compute the overlaps between train and test
sequences, and remove the conflicting sequences all at once. However, this will result in a very skewed train/test
ratio, or, in extreme cases, return an empty subset. In the example with `mask05` above, removing `(1, 6)` and `(18, 23)`
from the test sequences will resolve the leakage. But, if they are the only samples in the test set, this operation will
return an empty test set.

The second objective prevents this. We need a strategy that is more robust. Here are a few things to note:
* One sample (e.g. from the train subset) can overlap with more than one samples from the other subset (e.g. test).
* We seek to remove all conflicts, not all conflicting samples. In the example above, removing all conflicting samples
is equivalent to removing `(0, 5)`, `(2, 7)`, and `(15, 20)` from the train set and `(1, 6)` and `(18, 23)` from the
test set.

Essentially, we want to remove all the conflicts while keeping as many samples as possible. We thus target the few
samples with the most conflicts (an example of [Pareto principle](https://en.wikipedia.org/wiki/Pareto_principle)). 

#### Weighted bipartite graph

We can model this problem as a graph, with:
* nodes representing the data sequences (or by proxy, their interval masks)
* edges representing the overlaps/conflicts among sequences

The nodes include both train and test sequences. Since we are only interested in the train/test overlaps, and do not care
about the train/train and test/test overlaps, we can group the nodes into two distinct sets, thus creating a bipartite
graphs.

The edges, if only representing the presence of overlaps, can be unweighted. But if we are also interested in the degree
of overlaps (how many periods in common between two overlapping sequences), we can assign weights to the edges.

Our original problem is equivalent to finding the [minimum vertex cover](https://en.wikipedia.org/wiki/Vertex_cover)
of a graph, or its complement problem of finding the [maximum independent set](https://en.wikipedia.org/wiki/Independent_set_(graph_theory)).
It is [strongly NP-hard](https://en.wikipedia.org/wiki/Strong_NP-completeness). But for bipartite graph, we can approximate
an optimal solution by brute-force algorithm. Here are the steps:
1. Split the interval masks into provisional train/test sets (using e.g. `sklearn.model_selection.train_test_split`)
2. Compute an overlap matrix M representing the overlap lengths. Values in L should be >= 0 and <= sequence length
3. Compute the rowsum/colsum of M as the total overlaps.
4. Sort the rows/cols of M by its rowsum and colsum in descending order. The rows and columns with most conflicts will
appear first.
5. Decide to remove the most conflicting rows OR columns. Keep track of the current train/test ratio and the desired
ratio.
6. Repeat steps 3 to 5 until there is no more overlaps &rarr; M contains all 0.

#### Regarding performance for large datasets:
When applying this method for `PviCompositeDataset` or `PviLazyDataset` that can have up to ~200k sequences, we need
to do some housekeeping to reduce complexity. We note that these datasets contain sequences from multiple smaller datasets
that are disjoints. We can be sure there are no overlaps between `subject001` and `subject002`. The overlap matrix M thus
has a block-diagonal structure which can be used to our advantage, reducing the complexity from
`O((m1 + m2 + m3 + ...) * (n1 + n2 + n3 + ...))` to `O(m1*n1 + m2*n2 + m3*n3 + ...)`, with `m` and `n` being the initial
train and test size.

To implement this approach, `PviCompositeDataset` delegates the split operation to the constituent datasets and then
aggregates all the train and test masks to a global registry, while `PviLazyDataset` explicitly keeps track of the
dataset bounds and group the global masks into disjoint blocks.

### <a name="sampler"></a> Appendix D: `PviBatchSampler`

We compare following three approaches:
1. Random sampling: this is default behavior of `torch.utils.data.DataLoader`, with `shuffle=True`.
2. Sequential sampling: also default behavior, with `shuffle=False`
3. Stratified sampling with controlled batch randomization: objective of `PviBatchSampler`

For `PviLazyDataset` that loads samples on demand, it is expensive to use the default **random sampling**, because each
batch draws samples randomly from multiple source files, thus triggers the full _load &rarr; extract &rarr; unload_
process, just to yield a single sample in the current batch. With a fixed cache size, several samples can be compiled
before unloading loading a new source file, but that also depends on the arrival order of the samples in the batch.

In contrast, **sequential sampling** does not make use of the cache at all, because each batch draws all (or most)
of its samples sequentially from the same source, and only move on to the next file when there are no samples left.
In addition, the batches are heavily biased towards a specific source (subject).

Instead, we utilize the cache size to implement a controlled randomized approach: For every batch, we set a number
of maximum source files to draw samples from, i.e. diversity factor. This should be smaller than the cache size
to account for some extreme cases with small sources. To be truly source-agnostic, we randomly permute the source files
before caching and sampling, so that the batch combination also changes after each epoch.

#### Weighted and directed bipartite graph
This problem can be modeled as a (*weighted* and *directed*) bipartite graph, with:
* one set of nodes representing the data sources (*.h5 files),
* the other set of nodes representing the batches,
* the edges pointing from source &rarr; batch, representing ownership of the samples;
* the edge weights denoting the number of samples (>=1 and <= batch size).

Thus, summing all edge weights should give total number of samples from all sources, distributed in all batches. Now, we
can visualize the three scenarios above as following graph structures: 

1. True batch randomization: we have a dense graph with light edges, i.e. many edges, but with low weights. In the
extreme case, a batch can have as many incoming edges as its capacity, e.g. `batch_size=32` or `batch_size=64`, all
having weight of `1`. 
2. Fully sequential: we have sparse graph with heavy edges. Most batches have only one incoming edge, weighting the same
as their capacity, e.g. a single edge with weight `32` (or `64`).
3. Stratified sampling: we control the number of edges based on max cache size.

To implement scenario 3, we need to have knowledge of the global distribution of samples (their indices) at the start
of each epoch. Here are the steps to build a stratified sampler with cache-specific randomization:
1. Start with a master box containing all samples, in their natural order. (use global indices to avoid duplication)
2. Group the samples by their source files. The results is a box of smaller boxes, preserving the order in step 1.
3. First level of permutation: shuffle the small boxes in step 2, to change their order within the master box.
4. Grouping the small boxes into slightly larger boxes (but still smaller than the master box).
The number of small boxes to mix should be determined by the cache size (usually >=2 and <=5).
5. Second level of permutation: "unbox" step 4, and permute all the samples for each box. We now have local
randomization. This will create source-agnostic batches.
6. Unbox the master box, and send the samples to the batches.

### <a name="metrics"></a> Appendix E: Metrics computation during training
The class `ModelTrainer` computes the training metrics, i.e. loss and bp_accuracy for the train datasets, inside the method
`train_epoch`, which is meant for backpropagation. Meanwhile, there is another method `run_inference` which can also be
used to compute the metrics. Here are their differences: 
1. Train-mode vs. eval-mode computation: Dropout and batch normalization are active in train mode, thus producing
noisier predictions compared to eval-mode.
2. Running metrics vs. stacking results: Keeping a running score updated with += batch_score uses less memory than stacking
the predictions from all batches. For MSE and L1 error, the results are mathematically equivalent. However, running score
approach is incorrect for non-decomposable metrics like Pearson correlation or cosine similarity.
3. Progressively updated model states vs. final model state at the end of epoch.

Deciding where to compute the train metrics for logging/reporting depends on how they are interpreted and used:
* From `train_epoch`: If we want to visualize how the model navigates the loss landscape, and how regularization
(BatchNorm and Dropout) impacts training.
* From `run_inference`: If we want to more robust metrics that are comparable to the test metrics (which should ALWAYS
be computed in the inference loop).

However, for large datasets, the most important factor is time and memory spent on inference. For backpropagation, the model
already passes through the batches during `train_epoch` and compute the metrics (or at least the losses).
Computing the train metrics with `run_inference`  would require running the forward pass through the train batches again
--> 2x processing time, + prohibitively expensive memory for tensors stacking.