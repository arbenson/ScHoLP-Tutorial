using Base.Threads
using Combinatorics
using DelimitedFiles
using FileIO
using JLD2
using ScHoLP
using StatsBase

function read_txt_data(dataset::String)
    function read(filename::String)
        ret = Int64[]
        open(filename) do f
            for line in eachline(f)
                push!(ret, parse(Int64, line))
            end
        end
        return ret
    end
    return HONData(read("data/$(dataset)/$(dataset)-simplices.txt"),
                   read("data/$(dataset)/$(dataset)-nverts.txt"),
                   read("data/$(dataset)/$(dataset)-times.txt"),
                   dataset)
end

function read_node_labels(dataset::String)
    labels = Vector{String}()
    open("data/$(dataset)/$(dataset)-node-labels.txt") do f
        for line in eachline(f)
            push!(labels, join(split(line)[2:end], " "))
        end
    end
    return labels
end

function read_simplex_labels(dataset::String)
    labels = Vector{String}()
    open("data/$(dataset)/$(dataset)-simplex-labels.txt") do f
        for line in eachline(f)
            push!(labels, join(split(line)[2:end], " "))
        end
    end
    return labels
end

function read_closure_stats(dataset::String, simplex_size::Int64, initial_cutoff::Int64=100)
    keys = []
    probs, nsamples, nclosed = Float64[], Int64[], Int64[]
    data = readdlm("output/$(simplex_size)-node-closures/$(dataset)-$(simplex_size)-node-closures.txt")
    if initial_cutoff < 100
        data = readdlm("output/$(simplex_size)-node-closures/$(dataset)-$(simplex_size)-node-closures-$(initial_cutoff).txt")
    end
    for row_ind in 1:size(data, 1)
        row = convert(Vector{Int64}, data[row_ind, :])
        push!(keys, tuple(row[1:simplex_size]...))
        push!(nsamples, row[end - 1])
        push!(nclosed, row[end])
    end
    return (keys, nsamples, nclosed)
end

# This is just a convenient wrapper around all of the formatting parameters for
# making plots.
function all_datasets_params()
    green  = "#1b9e77"
    orange = "#d95f02"
    purple = "#7570b3"
    plot_params = [["coauth-DBLP",            "x", green],
                   ["coauth-MAG-Geology",     "x", orange],
                   ["coauth-MAG-History",     "x", purple],
                   ["music-rap-genius",       "v", green],
                   ["tags-stack-overflow",    "s", green],
                   ["tags-math-sx",           "s", orange],
                   ["tags-ask-ubuntu",        "s", purple],
                   ["threads-stack-overflow", "o", green],
                   ["threads-math-sx",        "o", orange],
                   ["threads-ask-ubuntu",     "o", purple],
                   ["NDC-substances",         "<", green],
                   ["NDC-classes",            "<", orange],
                   ["DAWN",                   "p", green],
                   ["congress-bills",         "*", green],
                   ["congress-committees",    "*", orange],
                   ["email-Eu",               "P", green],
                   ["email-Enron",            "P", orange],
                   ["contact-high-school",    "d", green],
                   ["contact-primary-school", "d", orange],
                   ]
    return plot_params
end
;
