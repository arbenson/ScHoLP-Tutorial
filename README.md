# Simplicial closure and higher-order link prediction tutorial

This Julia software accompanies the paper.

- Simplicial closure and higher-order link prediction.
  Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg.
  In preparation.

This tutorial code is not the main software library for simplicial closure and higher-order link prediction. Instead, the tutorial has the following goals:

1. Demonstrate how to use the package ScHoLP.jl for higher-order network analysis, in particular, for simplicial closure and higher-order link prediction.
2. Reproduce results and figures that appear in the paper.

### Setup

As discussed above, this tutorial shows how to use the ScHoLP.jl library for higher-order network analysis and reproduction of results. For the 

```julia
using ScHoLP
```

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

The datasets from the paper are available from http://www.cs.cornell.edu/~arb/data/

### Simplicial closure probabilities

Here we show how to compute the simplicial closure probabilities.

```julia
closure_type_counts3("email-Enron")  # 3-node configurations
closure_type_counts4("email-Enron")  # 4-node configurations
```

This should give the following output. The first 3 columns identify the type of triangle (the tie strength). The fourth column is the number of open instances in the first 80% of the dataset and the fifth column is the number of instances that closed in the final 20% of the dataset.

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



### Closure prediction

This section relies on the following julia file.

```julia
include("open_triangle_prediction.jl")
```

We first generate a labeled dataset.





### Reproducing research

##### Basic summary statistics of datasets

This reproduces the statistics in Table 1 of the paper.

```julia
basic_summary_statistics("email-Enron")
```

This should produce the following output.

```
dataset & # nodes & # edges in proj. graph & # simplices & # unique simplices
email-Enron & 143 & 1800 & 10883 & 1542
```



We can also compute some more advanced summary statistics (this takes a bit longer for larger datasets, but you can make use multiple threads).

```julia
summary_statistics("email-Enron")
```

This provides many more statistics and writes them to a csv file. For example, "meansimpsize" is the mean number of nodes in each simplex, "projdensity" is the edge density of the projected graph, and "nclosedtri"/"nopentri" are the number of closed and open triangles. The first line of the csv file are the variables, the first line are the statistics for the full dataset and the second line are the statistics for the dataset restricted to only 3-node simplices.

```
$ cat email-Enron-statistics.csv 
dataset,nnodes,nsimps,nconns,meansimpsize,maxsimpsize,meanprojweight,projdensity,nclosedtri,nopentri,nbbnodes,nbbconns,meanbbsimpsize,meanbbprojweight,meanbbconfigweight,bbconfigdensity,nbbconfigclosedtri,nbbconfigopentri
email-Enron,143,10883,26841,2,18,16.037222,1.772875e-01,6578,3317,1542,4648,3.014267,4.222778,2.028355,3.647198e-01,10460,36999
email-Enron-3-3,125,1231,3693,3,3,7.129344,6.683871e-02,317,331,324,972,3.000000,1.876448,1.167883,1.060645e-01,312,949
```





##### Temporal asynchronicity

To measure temporal asynchroncity in the datasets, we look at the number of "active interval" overlaps in the open triangles. The active interval is the time interval corresponding to the interval of time between the first and last simplices (in time) containing the two nodes.

```julia
interval_overlaps("email-Enron")
```

This should produce the following output:

```
dataset & # open triangles & 0 overlaps & 1 overlap & 2 overlaps & 3 overlaps
email-Enron & 3317 & 0.008 & 0.130 & 0.151 & 0.711
```

