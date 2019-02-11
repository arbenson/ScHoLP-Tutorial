# Simplicial Closure and Higher-order Link Prediction tutorial

This Julia software accompanies the following paper:

- [Simplicial closure and higher-order link prediction](http://www.pnas.org/content/early/2018/11/08/1800683115).
  Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg.
  *Proceedings of the National Academy of Sciences*, 2018.

This tutorial code is not the main software library for simplicial closure and higher-order link prediction, which is [ScHoLP.jl](https://github.com/arbenson/ScHoLP.jl). Instead, the tutorial has the following goals:

1. Demonstrate how to use the package ScHoLP.jl for higher-order network analysis, in particular, for simplicial closure and higher-order link prediction.
2. Reproduce results and figures that appear in the paper.

### Setup

As discussed above, this tutorial shows how to use the ScHoLP.jl library for higher-order network analysis and reproduction of results. To get the ScHoLP.jl library and start using it in Julia:

```julia
import Pkg
Pkg.add("ScHoLP")
Pkg.test("ScHoLP")
```

Note that ScHoLP.jl has thread-level parallelism available for many features (using Julia's Base.Threads).

To get started with this tutorial:

```bash
git clone https://github.com/arbenson/ScHoLP-Tutorial.git
cd ScHoLP-Tutorial
```

To run this entire tutorial, you will also need several Julia packages (not all packages are needed for each component; you can add them as necessary).

```julia
import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Distributions")
Pkg.add("FileIO")
Pkg.add("GLM")
Pkg.add("HypothesisTests")
Pkg.add("JLD2")
Pkg.add("PyCall")
Pkg.add("ScikitLearn")
```

### Data

The package comes with a few example datasets.

```julia
using ScHoLP
ex = example_dataset("example1")  # example from figure 1 of paper
typeof(ex)  # should be HONData
ex.simplices, ex.nverts, ex.times, ex.name  # components of the data structure
chs = example_dataset("contact-high-school")  # another dataset
```

The structure is a name (a string) and three vectors of integers. The simplices vector contains the elements of the simplicies and the nverts vector says how many vertices are in each simplex. The times vector gives the corresponding times. Here's an example loop through the data structure

```julia
using ScHoLP
ex = example_dataset("example1")  # example from figure 1 of paper
let curr_ind = 0
	for (nvert, time) in zip(ex.nverts, ex.times)
    	simplex = ex.simplices[(curr_ind + 1):(curr_ind + nvert)]
	    curr_ind += nvert
	    println("time = $time; simplex = $simplex")
	end
end
```

The tutorial also comes with a few datasets, which will make it feasible to reproduce most of the results from the paper.

```julia
# starting from the main directory of tutorial code
include("common.jl")
hist_coauth = read_txt_data("coauth-MAG-History")
ndc_classes = read_txt_data("NDC-classes")
enron = read_txt_data("email-Enron")
```

The collection of datasets from the paper are available from [this web site](http://www.cs.cornell.edu/~arb/data/). You can also download them wholesale and use them as follows.

```bash
cd ScHoLP-Tutorial/data
wget https://github.com/arbenson/ScHoLP-Data/archive/1.0.tar.gz
tar -xzvf 1.0.tar.gz
gunzip ScHoLP-Data-1.0/*/*.gz
mv ScHoLP-Data/* .
```

### Simplicial closures

Here we show how to count the simplicial closures as a function of the configuration of the nodes.

```julia
using ScHoLP
enron = example_dataset("email-Enron")
closure_type_counts3(enron)  # 3-node configurations
```

The closures are written to a file where the first 3 columns identify the type of triangle (0, 1, or 2 for missing, weight 1, or weight 2+ edge). The fourth column is the number of open instances in the first 80% of the dataset and the fifth column is the number of instances that closed in the final 20% of the dataset.

```
$ cat email-Enron-3-node-closures-100.txt 
0	0	0	265772	90
0	0	1	40734	78
0	0	2	109207	334
0	1	1	2025	19
0	1	2	10669	132
0	2	2	11082	236
1	1	1	75	0
1	1	2	449	14
1	2	2	1162	56
2	2	2	888	49
```

In this dataset, there were 888 triangles with 3 strong ties in the first 80% of the data, of which 49 simplicially closed in the final 20% of the data.

We can also look at 4-node configurations.

```julia
closure_type_counts4(enron)  # 4-node configurations
```

This produces the another text file. The key of the configuration is given by the first 4 columns. If all are nonnegative integers, then it represents an open tetrahedron (all six edges are present). The 0/1/2 refer to 

- 0: an open simplicial tie, i.e., a 3-node subset that is a triangle but has not appeared in a simplex
- 1: a weak simplicial tie, i.e., a 3-node subset that has appeared in exactly 1 simplex
- 2: a strong simplicial tie, i.e., a 3-node subset that has appeared in at least 2 simplices

If the key contains "-1", then the configuration contains exactly 5 edges (which implies exactly 2 triangles). If the key contains "-2", the configuration contains 4 edges, and if the key contains a "-3", the configuration contains 3 edges. In the latter two cases, we assume that there is a triangle in the 4-node configuration.

```
$ cat email-Enron-4-node-closures-100.txt 
-3	-3	-3	0	194830	81
-3	-3	-3	1	206119	45
-3	-3	-3	2	230148	411
-2	-2	-2	0	80874	151
-2	-2	-2	1	83325	112
-2	-2	-2	2	79161	627
-1	-1	0	0	10000	30
-1	-1	0	1	16125	55
-1	-1	0	2	14650	203
-1	-1	1	1	6192	18
-1	-1	1	2	11308	72
-1	-1	2	2	5054	107
0	0	0	0	771	2
0	0	0	1	1709	14
0	0	0	2	1713	30
0	0	1	1	1344	1
0	0	1	2	2649	32
0	0	2	2	1261	32
0	1	1	1	340	2
0	1	1	2	893	19
0	1	2	2	843	15
0	2	2	2	225	6
1	1	1	1	6	0
1	1	1	2	47	0
1	1	2	2	73	0
1	2	2	2	46	1
2	2	2	2	16	0
```

### Higher-order link prediction

We now turn to higher-order link prediction. We begin by including the following file and generating a labeled dataset.

```julia
# starting from the main directory of tutorial code
include("open_triangle_prediction.jl")
enron = read_txt_data("email-Enron")  # read from data/email-Enron directory
collect_labeled_dataset(enron)
```

Notice that this generates some new files in the `prediction-output` directory.

Now we can generate scores of the open triangles from the first 80% of the dataset. This should add a bunch of score files to the `prediction-output` directory.

```julia
collect_local_scores(enron)  # scores based on local structural features
collect_walk_scores(enron)  # scores based on random walks and paths
collect_logreg_supervised_scores(enron)  # scores based on logistic regression
collect_Simplicial_PPR_decomposed_scores(enron)  # scores based on Simplicial PPR
```

We can evaluate how well these methods do compared to random guessing with respect to area under the precision-recall curve. This should reproduce the line for the email-Enron dataset in Table 2 of the paper.

```julia
evaluate(enron, ["harm_mean", "geom_mean", "arith_mean", "common", "jaccard", "adamic_adar", "proj_graph_PA", "simplex_PA", "UPKatz", "WPKatz", "UPPR", "WPPR", "SimpPPR_comb", "SimpPPR_grad", "SimpPPR_harm", "SimpPPR_curl", "logreg_supervised"])
```

We can also look at the top predictions made by the algorithms.

```julia
top_predictions(enron, "UPPR", 12)
```

This should produce the following output

```
1 (0.304992; 0): joe.stepenovitch@enron.com; don.baughman@enron.com; larry.campbell@enron.com
2 (0.272495; 0): joe.stepenovitch@enron.com; don.baughman@enron.com; benjamin.rogers@enron.com
3 (0.253992; 0): larry.campbell@enron.com; don.baughman@enron.com; benjamin.rogers@enron.com
4 (0.189678; 0): joe.parks@enron.com; eric.bass@enron.com; dan.hyvl@enron.com
5 (0.181085; 1): lisa.gang@enron.com; kate.symes@enron.com; bill.williams@enron.com
6 (0.179377; 0): joe.quenet@enron.com; chris.dorland@enron.com; jeff.king@enron.com
7 (0.176236; 0): joe.quenet@enron.com; jeff.king@enron.com; fletcher.sturm@enron.com
8 (0.175624; 1): lisa.gang@enron.com; holden.salisbury@enron.com; kate.symes@enron.com
9 (0.173160; 1): lisa.gang@enron.com; holden.salisbury@enron.com; bill.williams@enron.com
10 (0.170947; 0): geir.solberg@enron.com; holden.salisbury@enron.com; kate.symes@enron.com
11 (0.164845; 0): geir.solberg@enron.com; holden.salisbury@enron.com; bill.williams@enron.com
12 (0.162391; 0): lisa.gang@enron.com; cara.semperger@enron.com; kate.symes@enron.com
```

These are the top 12 predictions for the unweighted personalized PageRank scores. The tuple next to the ordered numbers, e.g., (0.304992; 0) in the first line, gives the score function value and a 0/1 indicator of whether or not the open triangle closed in the final 20% of the dataset (1 means that it closed). Here, we see that the triples of nodes with the 5th, 8th, and 9th highest scores went through a simplicial closure event.

### Summary statistics

There is some basic functionality for gathering summary statistics about the datasets.

```julia
chs = example_dataset("contact-high-school")
# print basic summary statistics (same as Table 1 in paper)
basic_summary_statistics(chs)
# compute more advanced statistics --> contact-high-school-statistics.csv
summary_statistics(chs);
```

The last command writes several summary statistics to a csv file. For example, "meansimpsize" is the mean number of nodes in each simplex, "projdensity" is the edge density of the projected graph, and "nclosedtri" and "nopentri" are the number of closed and open triangles. The first line of the csv file are the variables, the first line are the statistics for the full dataset and the second line are the statistics for the dataset restricted to only 3-node simplices.

```
$ cat contact-high-school-statistics.csv 
dataset,nnodes,nsimps,nconns,meansimpsize,maxsimpsize,meanprojweight,projdensity,nclosedtri,nopentri,nbbnodes,nbbconns,meanbbsimpsize,meanbbprojweight,meanbbconfigweight,bbconfigdensity,nbbconfigclosedtri,nbbconfigopentri
contact-high-school,327,172035,352718,2,5,32.644895,1.091537e-01,2370,31850,7937,18479,2.328210,2.302681,1.223864,2.044052e-01,3034,88912
contact-high-school-3-3,317,7475,22425,3,3,8.305556,5.390728e-02,2091,5721,2126,6378,3.000000,2.362222,1.132810,1.118476e-01,2094,18139
```

### Reproducing results in the main text

This section shows how to reproduce results from the paper.

##### Linear models for relationships in Figures 2D and 2E

We create linear models for the fraction of triangles in terms of the covariate log average degree (plus an intercept term). The following code snippet produces these models.

```julia
# starting from the main directory of tutorial code
include("statistical_tests_and_models.jl")
model_fig_2D, model_fig_2E = fracopen_logavedeg_linear_models();
r2(model_fig_2D)  # roughly 0.38
r2(model_fig_2E)  # roughly 0.85
```

##### Hypothesis tests for fewer strong ties vs. more weaker ties

Here we are testing hypotheses on whether stronger but fewer ties or weaker but more ties are more indicative of simplicial closure.

```julia
# starting from the main directory of tutorial code
include("statistical_tests_and_models.jl")
simplicial_closure_tests()
# run at significance level 1e-3 instead
simplicial_closure_tests(1e-3)
```

##### Table 1 (basic dataset statistics)

We saw how to get these numbers in the summary statistics section above. The `basic_summary_statistics()` function produces the numbers.

##### Table 2 (logistic regression for system domain classification)

Egonet data was collected with the function call `collect_egonet_data(100, 20)` in the file `egonet_analysis.jl`. This takes some time, so we pre-computed the data output and stored it in the directory `output/egonets`. We can reproduce the performance of the logistic regression models with the following code snippet. Performance numbers might be slightly different due to randomization.

```julia
include("egonet_analysis.jl")
egonet_predict([LOG_DENSITY, LOG_AVE_DEG, FRAC_OPEN])
egonet_predict([LOG_AVE_DEG, FRAC_OPEN])
egonet_predict([LOG_DENSITY, FRAC_OPEN])
egonet_predict([LOG_DENSITY, LOG_AVE_DEG])
```

##### Table 3 (Higher-order link prediction performance)

The numbers in this table came from using the higher-order link prediction methods outlined above. Note that some of the score functions are computationally expensive. The necessary julia functions are 

- `collect_labeled_dataset()` to generate the labeled dataset based on an 80/20 split of the data
- `collect_local_scores()` to generate scores based on local structural features
- `collect_walk_scores() ` to generate scores based on random walks and paths
- `collect_logreg_supervised_scores()` to generate scores from the supervised learning method

After collecting the data, we can reproduce results in the table with the following commands.

```julia
include("open_triangle_prediction.jl")
enron = example_dataset("email-Enron")
evaluate(enron, ["harm_mean", "geom_mean", "arith_mean", "adamic_adar", "proj_graph_PA", "UPKatz", "UPPR", "logreg_supervised"])
```

##### Figure 1 (small example of higher-order network)

The example higher-order network in Figure 1 is one of the examples included with the library. Here we show how to list the simplices and compute the weighted projected graph.

```julia
using ScHoLP
ex_fig1 = example_dataset("example1")

# Print out simplices
function print_simplices()
	ind = 1
	for (nv, t) in zip(ex_fig1.nverts, ex_fig1.times)
    	simplex = ex_fig1.simplices[ind:(ind + nv - 1)]
	    ind += nv
	   	println("$t $simplex")
	end
end
print_simplices()

# Get the weighted projected graph
basic_matrices(ex_fig1)[3]
```

##### Figure 2A (simplex size distribution)

Here is a sample code snippet for computing the simplex size distribution for the email-Enron dataset.

```julia
# Example code for computing simplex size distribution
using ScHoLP
using StatsBase: countmap
enron = example_dataset("email-Enron")
num_verts, counts = [collect(v) for v  in zip(sort(countmap(enron.nverts))...)]
tot = sum(counts)
fracs = [count / tot for count in counts];
for (nv, f) in zip(num_verts, fracs)
    println("$nv: $f")
end
```

For reproducing the figure, we have pre-computed the distributions in the files `output/simplex-size-dists/*-simplex-size-dist.jld2`. The following produces the simplex size distribution and saves the figure.

```julia
# starting from the main directory of tutorial code
include("paper_plots.jl")
# produce figure 2A --> simplex-size-dist.pdf
simplex_size_dist_plot()
```

##### Figure 2B—E (basic dataset structure)

These figures rely on using the `summary_statistics()` function for all of the datasets. For some of the larger datasets, this can take a while. For this tutorial, we include the pre-computed statistics are in the `output/summary-stats/` directory. The following code snippet reproduces the figure.

```julia
# starting from the main directory of tutorial code
include("paper_plots.jl")
# produce figures 2BCDE --> dataset-structure.pdf
dataset_structure_plots()
```

##### Figure 3 (logistic regression decision boundary)

Plot the decision boundary for the logistic regression classifier. 

```julia
include("paper_plots.jl")
logreg_decision_boundary()
```

##### Figure 4 (model simulation)

These figures require running simulations. Since the simulations are random, the output may not be exactly the same. The following will re-run the simulations and write the results to `simulation.jld2`.

```julia
# starting from the main directory of tutorial code
include("simulations.jl")
# run the simulations (takes several minutes)
simulate()  # --> stores in output/simulation/simulation.jld2
```

The simulation results used in the paper are stored in `output/simulation/simulation.jld2` for convenience. The above code should produce something similar but not exactly the same (due to randomness in the simulation). The following code snippet reproduces figures 2GH.

```julia
# starting from the main directory of tutorial code
include("paper_plots.jl")
simulation_plot()  # reproduce Figure 4 --> simulation.pdf
```

##### Figure 5 (lifecycles)

Producing results from Figure 5A uses the `process_lifecycles()` function from the ScHoLP.jl library. Figures 5B—D use the `lifecycle()` function in`lifecycle_analysis.jl`. 

```julia
include("lifecycle_analysis.jl")
# read dataset from data/coauth-MAG-History directory
hist = read_txt_data("coauth-MAG-History")  
# Get data for Figure 5A (this may take a couple minutes)
closed_transition_counts, open_transition_counts = process_lifecycles(hist)
# direct transitions to simplicial closure from each state
closed_transition_counts[end, 1:(end-1)]
# read data from data/NDC-classes directory
ndc_classes = read_txt_data("NDC-classes")  
node_labels = read_node_labels("NDC-classes")
node_labels[[44, 74, 76]]  # nodes in Figure 5B
lifecycle(ndc_classes, 44, 74, 76)
```

The simplex labels in the last function call are NDC codes. For example, the first one is 67296-1236. This corresponds to Reyataz as produced by Redpharm Drug, Inc. in 2003, as recorded [here](https://ndclist.com/ndc/67296-1236/package/67296-1236-4).

##### Figure 6 (3-node and 4-node configuration closure probabilities)

This figure is constructed from the simplicial closure probabilities on 3-node configurations. Above, we showed how to compute these. We have pre-computed the probabilities for each dataset in the directory `output/3-node-closures/`.

```julia
# starting from the main directory of tutorial code
include("paper_plots.jl")
three_node_scatter_plot()  # Figures 6ABC --> closure-prob-scatter-3.pdf
four_node_scatter_plot()   # Figures 6DEF --> closure-prob-scatter-4.pdf
```

##### Figure 7 (generalized means)

We first show how to collect the data for generalized means. The following code snippet should produce an output file `prediction-output/email-Enron-open-tris-80-100-genmeans-perf.jld2`.

```julia
# starting from the main directory of tutorial code
include("open_triangle_prediction.jl")
enron = read_txt_data("email-Enron")  # read dataset from data/email-Enron directory
collect_labeled_dataset(enron)  # generate 80/20 split on triangles
ps, improvements = collect_generalized_means(enron)
```

We pre-computed the generalized mean scores for all of the datasets in the paper, which are in the directory `output/generalized-means/`. To reproduce Figure 6, you can then use the following code snippet.

```julia
# starting from the main directory of tutorial code
include("paper_plots.jl")
generalized_means_plot()  # --> generalized-means-perf.pdf
```



### Reproduce results in the supplementary material

##### Table S1 (temporal asynchroncity)

To measure temporal asynchroncity in the datasets, we look at the number of "active interval" overlaps in the open triangles. The active interval is the time interval corresponding to the interval of time between the first and last simplices (in time) containing the two nodes.

```julia
using ScHoLP
enron = example_dataset("email-Enron")
interval_overlaps(enron)
```

This should produce the following output:

```
dataset & # open triangles & 0 overlaps & 1 overlap & 2 overlaps & 3 overlaps
email-Enron & 3317 & 0.008 & 0.130 & 0.151 & 0.711
```

##### Table S2 (dependence of tie strength and edge density at different points in time)

The results from this table just use the core ScHoLP.jl functionality and the same function we saw above for the simplicial closure probabilities. We just provide an extra input parameter to the function `closure_type_counts3()` for pre-filtering the dataset to just start with the first X% of timestamped simplices.

```julia
using ScHoLP
enron = example_dataset("email-Enron")
for X in [40, 60, 80]
	closure_type_counts3(enron, X)  # start with first X% of data
end
```

This creates text files `email-Enron-3-node-closures-{40,60,80}.txt`. For convenience, we provide all of the pre-computed closure statistics in the directory `output/{3,4}-node-closures/`. The following code snippet shows how to run the hypothesis tests that are reported in the table.

```julia
# starting from the main directory of tutorial code
include("statistical_tests_and_models.jl")
only_3_node_tests = true
for X in [40, 60, 80, 100]
    simplicial_closure_tests(1e-5, X, only_3_node_tests)
end
```

##### Table S3 (Simplicial closure probabilities at different points in time)

In describing how to reproduce Table S2, we showed how to get the closure probabilities at different points in time. The following code snippet prints out some of the statistics for other datasets, which are pre-computed and stored in the `output/3-node-closures/` directory.

```julia
# starting from the main directory of tutorial code
include("common.jl")
function closure_stats_over_time(dataset::String)
    # Read closures 
    stats = [read_closure_stats(dataset, 3, X) for X in [40, 60, 80, 100]]
    keys = stats[1][1]  # same across each
    for i in 1:length(keys)
        nsamples = [stat[2][i] for stat in stats]
        nclosures = [stat[3][i] for stat in stats]
        frac_closed = nclosures ./ nsamples
        println("$(keys[i]) $frac_closed")
    end
end
closure_stats_over_time("DAWN")
closure_stats_over_time("tags-stack-overflow")
```

##### Table S5 (extra results from models)

Example to get all of the results form the Enron dataset.

```julia
include("open_triangle_prediction.jl")
enron = read_txt_data("email-Enron")  # read from data/email-Enron directory
collect_labeled_dataset(enron)
collect_local_scores(enron)  # scores based on local structural features
collect_walk_scores(enron)  # scores based on random walks and paths
collect_logreg_supervised_scores(enron)  # scores based on logistic regression
collect_Simplicial_PPR_decomposed_scores(enron)  # scores based on Simplicial PPR
evaluate(enron, ["harm_mean", "geom_mean", "arith_mean", "common", "jaccard", "adamic_adar", "proj_graph_PA", "simplex_PA", "UPKatz", "WPKatz", "UPPR", "WPPR", "SimpPPR_comb", "logreg_supervised"])
```

##### Table S6 (extra results from the Hodge decomposition)

This table shows the results from using the Hodge decomposition to further decompose the simplicial personalized PageRank scores. Note that this software uses the newer normalization method described in the following paper:

- [Random walks on simplicial complexes and the normalized Hodge Laplacian](https://arxiv.org/abs/1807.05044). Michael T. Schaub, Austin R. Benson, Paul Horn, Gabor Lippner, and Ali Jadbabaie. *arXiv:1807.05044*, 2018.

Here is how one would reproduce the line for the NDC-classes dataset.

```julia
# starting from the main directory of tutorial code
include("open_triangle_prediction.jl")
# read data from data/NDC-classes directory
ndc_classes = read_txt_data("NDC-classes")  
# collect the data from the 80/20 split
collect_labeled_dataset(ndc_classes)  
# collect scores
collect_Simplicial_PPR_decomposed_scores(ndc_classes)  
# print relative scores
evaluate(ndc_classes, ["SimpPPR_comb", "SimpPPR_grad", "SimpPPR_harm", "SimpPPR_curl"]) 
```

##### Table S7 (output predictions)

We showed how to look at the top predictions in the higher-order link prediction section above. Here is the specific command to reproduce Table S7.

```julia
include("open_triangle_prediction.jl")
dawn = read_txt_data("DAWN")  # need to download DAWN data to data/ directory
collect_labeled_dataset(dawn)
collect_local_scores(dawn)
top_predictions(dawn, "adamic_adar", 25)
```

##### Figure S1 (heat map of 3-node closures)

```julia
include("paper_plots.jl")
closure_probs_heat_map(3)
```

##### Figure S2 (heat map of 4-node closures)

```julia
include("paper_plots.jl")
closure_probs_heat_map(3)
```

##### Figure S3 (heat map of 3-node closures at different points in time)

```julia
include("paper_plots.jl")
for X in [40, 60, 80, 100]
	closure_probs_heat_map(3, X)
end
```

