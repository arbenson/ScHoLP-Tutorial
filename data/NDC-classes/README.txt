This dataset consists of the pharmaceutical classes used to classify drugs in
the National Drug Code Directory maintained by the Food and Drug
Administration. The original data was downloaded from
https://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm. Each simplex
corresponds to an NDC code for a drug, where the nodes are the classes applied
to the drug.  Timestamps are in days and represent when the drug was first
marketed. Note that the same drug substance can have more than one NDC code. For
example, different dosages of the same drug may result in multiple NDC codes.


The file NDC-classes-node-labels.txt maps the node IDs to the classes.

The nth line in NDC-classes-simplex-labels.txt is the name of the drug
corresponding to the nth simplex.
