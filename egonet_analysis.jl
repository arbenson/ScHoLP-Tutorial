include("common.jl")

using DataFrames
using GLM
using Printf
using Random
using SparseArrays
using Statistics

using ScikitLearn
@sk_import linear_model: LogisticRegression

# Construct HONData for a given ego
function egonet_dataset(dataset::HONData, ego::Int64, B::SpIntMat)
    in_egonet = zeros(Bool, size(B, 1))
    in_egonet[ego] = true
    in_egonet[findnz(B[:, ego])[1]] .= true

    node_map = Dict{Int64, Int64}()
    function get_key(x::Int64)
        if haskey(node_map, x); return node_map[x]; end
        n = length(node_map) + 1
        node_map[x] = n
        return n
    end
    ego_key = get_key(ego)
    
    new_simplices = Int64[]
    new_nverts = Int64[]
    new_times = Int64[]
    curr_ind = 1
    for (nvert, time) in zip(dataset.nverts, dataset.times)
        end_ind = curr_ind + nvert - 1
        simplex = dataset.simplices[curr_ind:end_ind]
        curr_ind += nvert
        simplex_in_egonet = [v for v in simplex if in_egonet[v]]
        if length(simplex_in_egonet) > 0
            mapped_simplex = [get_key(v) for v in simplex_in_egonet]
            append!(new_simplices, mapped_simplex)
            push!(new_nverts, length(mapped_simplex))
            push!(new_times, time)
        end
    end

    return HONData(new_simplices, new_nverts, new_times, "egonet")
end

function egonet_stats(dataset_name::String, num_egos::Int64)
    # read data
    dataset = read_txt_data(dataset_name)
    A1, At1, B1 = basic_matrices(dataset.simplices, dataset.nverts)
    
    # Get eligible egos
    n = size(B1, 1)
    tri_order = proj_graph_degree_order(B1)
    in_tri = zeros(Int64, n, Threads.nthreads())
    Threads.@threads for i = 1:n
        for (j, k) in neighbor_pairs(B1, tri_order, i)
            if B1[j, k] > 0
                tid = Threads.threadid()
                in_tri[[i, j, k], tid] .= 1
            end
        end
    end
    eligible_egos = findall(vec(sum(in_tri, dims=2)) .> 0)
    num_eligible = length(eligible_egos)
    println("$num_eligible eligible egos")
    
    # Sample from eligible egos
    sampled_egos =
        eligible_egos[StatsBase.sample(1:length(eligible_egos),
                                       num_egos, replace=false)]

    # Collect statistics
    X = zeros(Float64, NUM_FEATS, length(sampled_egos))
    for (j, ego) in enumerate(sampled_egos)
        print(stdout, "$j \r")
        flush(stdout)
        egonet = egonet_dataset(dataset, ego, B1)
        A, At, B = basic_matrices(egonet.simplices, egonet.nverts)

        num_nodes = sum(sum(At, dims=1) .> 0)
        no, nc = num_open_closed_triangles(A, At, B)

        # log average degree
        X[LOG_AVE_DEG, j] = log.(nnz(B) / num_nodes)
        # log edge density
        X[LOG_DENSITY, j] = log.(nnz(B) / (num_nodes^2 - num_nodes))
        # frac. open tris
        X[FRAC_OPEN, j] = no / (no + nc)
    end
    
    return convert(SpFltMat, X')
end

function collect_egonet_data(num_egos::Int64, trial::Int64)
    Random.seed!(1234 * trial)  # reproducibility
    dataset_names = [row[1] for row in all_datasets_params()]
    ndatasets = length(dataset_names)
    X = zeros(Float64, 0, NUM_FEATS)
    labels = Int64[]
    for (ind, dname) in enumerate(dataset_names)
        println("$dname...")
        label = nothing
        if     (dname == "coauth-DBLP" ||
                dname == "coauth-MAG-Geology" ||
                dname == "coauth-MAG-History");      label = 0;
        elseif (dname == "tags-stack-overflow" ||
                dname == "tags-math-sx"        ||
                dname == "tags-ask-ubuntu");         label = 1;
        elseif (dname == "threads-stack-overflow" ||
                dname == "threads-math-sx"        ||
                dname == "threads-ask-ubuntu");      label = 2;
        elseif (dname == "contact-high-school" ||
                dname == "contact-primary-school");  label = 3;
        elseif (dname == "email-Eu" ||
                dname == "email-Enron");             label = 4;
        end
        if label != nothing
            X = [X; egonet_stats(dname, num_egos)]
            append!(labels, ones(Int64, num_egos) * label)
        end
    end
    save("output/egonets/egonet-data-$trial.jld2",
         Dict("X" => X, "labels" => labels))
end

function egonet_predict(feat_cols::Vector{Int64})
    accs_mlr = Float64[]
    accs_rnd = Float64[]

    for trial in 1:20
        (X_train, X_test, y_train, y_test) = egonet_train_test_data(trial)[1:4]
        X_train = X_train[:, feat_cols]
        X_test = X_test[:, feat_cols]
        model = LogisticRegression(fit_intercept=true, multi_class="multinomial",
                                   C=10, solver="newton-cg", max_iter=10000)
        ScikitLearn.fit!(model, X_train, y_train)
        rand_prob =
            sum([(sum(y_train .== l) / length(y_train))^2 for l in unique(y_train)])
        push!(accs_mlr, ScikitLearn.score(model, X_test, y_test))
        push!(accs_rnd, rand_prob)
    end

    @printf("%0.2f +/- %0.2f\n", mean(accs_mlr), std(accs_mlr))
    @printf("%0.2f +/- %0.2f\n", mean(accs_rnd), std(accs_rnd))
end
