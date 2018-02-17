# Simplicial closure and higher-order link prediction tutorial

This Julia software accompanies the paper.

- Simplicial closure and higher-order link prediction.
  Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg.
  In preparation.

This tutorial code is not the main software library for simplicial closure and higher-order link prediction. Instead, the tutorial has the following goals:

1. Demonstrate how to use the package ScHoLP.jl for higher-order network analysis, in particular, for simplicial closure and higher-order link prediction.
2. Reproduce results and figures that appear in the paper.

### Setup

As discussed above, this tutorial shows how to use the ScHoLP.jl library for higher-order network analysis and reproduction of results. To get the ScHoLP.jl library and start using it:

```julia
Pkg.clone("https://github.com/arbenson/ScHoLP.jl.git")
Pkg.test("ScHoLP")
using ScHoLP
```

ScHoLP.jl has thread-level parallelism available for many features (using Julia's Base.Threads).

### Data

The package comes with a few example datasets.

```julia
ex = example_dataset("example1")  # example from figure 1
typeof(ex)  # should be ScHoLP.HONData
ex.simplices, ex.nverts, ex.times, ex.name  # components of the data structure
chs = example_dataset("contact-high-school") # another dataset
```

The tutorial also comes with a few datasets.

```julia
include("common.jl")
hist_coauth = read_txt_data("coauth-MAG-History")
ndc_classes = read_txt_data("NDC-classes")
enron = read_txt_data("email-Enron")
```

The collection of datasets from the paper are available at http://www.cs.cornell.edu/~arb/data/

### Simplicial closures

Here we show how to count the simplicial closures as a function of the configuration of the nodes.

```julia
using ScHoLP
enron = example_dataset("email-Enron")
closure_type_counts3(enron)  # 3-node configurations
```

The closures are written to a file where the first 3 columns identify the type of triangle (0, 1, or 2 for missing, weight 1, or weight 2+). The fourth column is the number of open instances in the first 80% of the dataset and the fifth column is the number of instances that closed in the final 20% of the dataset.

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

So there were 888 triangles with 3 strong ties in the first 80% of the data, of which 49 simplicially closed in the final 20% of the data.

We can also look at 4-node configurations.

```julia
closure_type_counts4(enron)  # 4-node configurations
```

This produces the another text file. The key of the configuration is given by the first 4 columns. If all are nonnegative integers, then it represents an open tetrahedron (all six edges are present). The 0/1/2 refer to 

- 0: an open simplicial tie, i.e., 3-node subset is a triangle but has not appeared in a simplex
- 1: a weak simplicial tie, i.e., 3-node subset has appeared in exactly 1 simplex
- 2: a strong simplicial tie, i.e., 3-node subset has appeared in at least 2 simplices

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
include("open_triangle_prediction.jl")
enron = read_txt_data("email-Enron")  # read from data/email-Enron directory
collect_labeled_dataset(enron)
```

Notice that this generates some new files in the `prediction-output` directory.

Now we can generate a bunch of scores of the open triangles from the first 80% of the dataset.

```julia
collect_local_scores(enron)  # scores based on local structural features
collect_walk_scores(enron) # scores based on random walks and paths
collect_Simplicial_PPR_combined_scores(enron) # scores based on Simplicial PPR
collect_logreg_supervised_scores(enron) # scores based on logistic regression supervised method
```

This should add a bunch of score files to the `prediction-output` directory.

Since enron is a small dataset, we can afford to decompose the Simplicial PPR scores into the gradient, curl, and harmonic components:

```julia
collect_Simplicial_PPR_decomposed_scores(enron)
```

We can evaluate how well these methods do compared to random guessing with respect to area under the precision-recall curve:

```julia
evaluate(enron, ["harm_mean", "geom_mean", "arith_mean", "common", "jaccard", "adamic_adar", "proj_graph_PA", "simplex_PA", "UPKatz", "WPKatz", "UPPR", "WPPR", "SimpPPR_comb", "logreg_supervised", "SimpPPR_grad", "SimpPPR_harm", "SimpPPR_curl"])
```

This should reproduce the line for the email-Enron dataset in Table 2.

### Summary statistics

There is some basic functionality for gathering summary statistics about the datasets.

```julia
chs = example_dataset("contact-high-school")
basic_summary_statistics(chs) # prints basic summary statistics (same as Table 1 in paper)
summary_statistics(chs) # more advanced statistics (produces contact-high-school-statistics.csv)
```

The last command writes several summary statistics to a csv file. For example, "meansimpsize" is the mean number of nodes in each simplex, "projdensity" is the edge density of the projected graph, and "nclosedtri"/"nopentri" are the number of closed and open triangles. The first line of the csv file are the variables, the first line are the statistics for the full dataset and the second line are the statistics for the dataset restricted to only 3-node simplices.

```
$ cat contact-high-school-statistics.csv 
dataset,nnodes,nsimps,nconns,meansimpsize,maxsimpsize,meanprojweight,projdensity,nclosedtri,nopentri,nbbnodes,nbbconns,meanbbsimpsize,meanbbprojweight,meanbbconfigweight,bbconfigdensity,nbbconfigclosedtri,nbbconfigopentri
contact-high-school,327,172035,352718,2,5,32.644895,1.091537e-01,2370,31850,7937,18479,2.328210,2.302681,1.223864,2.044052e-01,3034,88912
contact-high-school-3-3,317,7475,22425,3,3,8.305556,5.390728e-02,2091,5721,2126,6378,3.000000,2.362222,1.132810,1.118476e-01,2094,18139
```



## Reproducing research

This section is dedicated to showing how to reproduce results from the paper.

##### Table 1

We saw this above in the summary statistics. The `summary_statistics()` command produces the numbers.

##### Table 2

These numbers came from using the higher-order link prediction methods outlined above. Some of the methods are computationally expensive and take days to complete on a large memory server with 64 threads. The functions are 

- `collect_labeled_dataset()` to generate the labeled dataset based on an 80/20 split of the data
- `collect_local_scores()` to generate scores based on local structural features
- `collect_walk_scores() ` to generate scores based on random walks and paths
- `collect_Simplicial_PPR_combined_scores()` to generate scores based on simplicial PPR
- `collect_logreg_supervised_scores()` to generate scores from the supervised learning method

##### Figure 1

##### Figure 2

##### Figure 3

##### Figure 4

##### Figure 5

##### Figure 6

##### Table S1 (temporal asynchroncity)

To measure temporal asynchroncity in the datasets, we look at the number of "active interval" overlaps in the open triangles. The active interval is the time interval corresponding to the interval of time between the first and last simplices (in time) containing the two nodes.

```julia
enron = example_dataset("email-Enron")
interval_overlaps(enron)
```

This should produce the following output:

```
dataset & # open triangles & 0 overlaps & 1 overlap & 2 overlaps & 3 overlaps
email-Enron & 3317 & 0.008 & 0.130 & 0.151 & 0.711
```

##### Table S2

##### Table S3

##### Table S4

##### Table S5

##### Table S6



