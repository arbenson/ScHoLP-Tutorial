The simplices in this dataset are constructed from the Enron email dataset,
downloaded from https://www.cs.cmu.edu/~./enron/.

Each simplex corresponds to an email. The nodes in the simplex are the Enron
employees that are senders or recipients of the email. Note that the nodes are
only Enron employees, but the emails may also have had non-Enron employees on
them. We only include emails that are sent by an Enron employee and have at
least one recipient who is an Enron employee. Timestamps are in milliseconds.

The file email-Enron-addresses.txt provides the map of node ids to email
addresses.
