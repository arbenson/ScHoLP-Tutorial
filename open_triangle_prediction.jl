include("common.jl")

using ScHoLP

using LinearAlgebra
using Printf
using PyCall

using ScikitLearn
@sk_import linear_model: LogisticRegression
@sk_import metrics: average_precision_score

const OUTDIR = "prediction-output"
basename_str(dataset::String) = "$(OUTDIR)/$dataset-open-tris-80-100"

function read_data(dataset::HONData, prcntl1::Int64, prcntl2::Int64)
    fname = "$(OUTDIR)/$(dataset.name)-open-tris-$prcntl1-$prcntl2.jld2"
    data = load(fname)["data"]
    dataT = data'
    ntri = size(dataT, 2)
    triangles = Vector{NTuple{3,Int64}}(undef, ntri)
    labels = Vector{Int64}(undef, ntri)
    for i in 1:ntri
        triangles[i] = (dataT[1, i], dataT[2, i], dataT[3, i])
        labels[i] = dataT[4, i]
    end
    return triangles, labels
end

function write_scores(dataset::HONData, score_type::String, scores::Vector{Float64})
    basename = basename_str(dataset.name)
    save("$basename-scores-$score_type.jld2", Dict("scores" => scores))
end

function read_scores(dataset::HONData, score_type::String)
    basename = basename_str(dataset.name)    
    data = load("$basename-scores-$score_type.jld2")
    return convert(Vector{Float64}, data["scores"])
end

function collect_local_scores(dataset::HONData)
    triangles = read_data(dataset, 80, 100)[1]
    old_simplices, old_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 80, 100)[1:2]
    A, At, B = basic_matrices(old_simplices, old_nverts)
    
    println("harmonic mean...")
    write_scores(dataset, "harm_mean", harmonic_mean(triangles, B))
    
    println("arithmetic mean...")
    write_scores(dataset, "arith_mean", arithmetic_mean(triangles, B))
    
    println("geometric mean...")
    write_scores(dataset, "geom_mean", geometric_mean(triangles, B))

    degrees = vec(sum(make_sparse_ones(B), dims=1))
    println("projected graph preferential attachment...")
    write_scores(dataset, "proj_graph_PA", pref_attach3(triangles, degrees))
    
    simp_degrees = vec(sum(At, dims=1))
    println("simplex preferential attachment...")    
    write_scores(dataset, "simplex_PA", pref_attach3(triangles, simp_degrees))
    
    println("common...")
    common_nbrs = common_neighbors_map(B, triangles)
    write_scores(dataset, "common", common3(triangles, common_nbrs))
    
    println("jaccard...")
    write_scores(dataset, "jaccard", jaccard3(triangles, common_nbrs, degrees))
    
    println("adamic-adar...")        
    write_scores(dataset, "adamic_adar", adamic_adar3(triangles, common_nbrs, degrees))
end

function collect_walk_scores(dataset::HONData)
    triangles = read_data(dataset, 80, 100)[1]
    old_simplices, old_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 80, 100)[1:2]
    A, At, B = basic_matrices(old_simplices, old_nverts)
    basename = basename_str(dataset.name)
    #dense_solve = size(B, 2) < 10000
    dense_solve = false
    
    println("Unweighted personalized Katz...")
    scores, S = PKatz3(triangles, B, true, dense_solve)
    write_scores(dataset, "UPKatz", scores)
    save("$basename-UPKatz.jld2", Dict("S" => S))

    println("Weighted personalized Katz...")    
    scores, S = PKatz3(triangles, B, false, dense_solve)    
    write_scores(dataset, "WPKatz", scores)
    save("$basename-WPKatz.jld2", Dict("S" => S))

    println("Unweighted personalized PageRank...")
    scores, S = PPR3(triangles, B, true, dense_solve)
    write_scores(dataset, "UPPR", scores)
    save("$basename-UPPR.jld2", Dict("S" => S))

    println("Weighted personalized PageRank...")    
    scores, S = PPR3(triangles, B, false, dense_solve)    
    write_scores(dataset, "WPPR", scores)
    save("$basename-WPPR.jld2", Dict("S" => S))
end

function collect_logreg_supervised_scores(dataset::HONData)
    function feature_matrix(triangles::Vector{NTuple{3,Int64}},
                            At::SpIntMat, B::SpIntMat)
        degrees = vec(sum(make_sparse_ones(B), dims=1))
        simp_degrees = vec(sum(At, dims=1))
        common_nbrs = common_neighbors_map(B, triangles)
        ntriangles = length(triangles)
        X = zeros(Float64, 26, ntriangles)
        Threads.@threads for ind = 1:ntriangles
            i, j, k = triangles[ind]
            X[1:3, ind] = [B[i, j]; B[j, k]; B[i, k]]
            X[4:6, ind] = degrees[[i, j, k]]
            X[7:9, ind] = simp_degrees[[i, j, k]]
            common_ij = common_nbr_set(common_nbrs, i, j)
            common_ik = common_nbr_set(common_nbrs, i, k)
            common_jk = common_nbr_set(common_nbrs, j, k)
            X[10, ind] = length(common_ij)
            X[11, ind] = length(common_ik)
            X[12, ind] = length(common_jk)
            X[13, ind] = length(intersect(common_ij, common_ik, common_jk))
            X[14:22, ind] = log.(X[1:9, ind])
            X[23:26, ind] = log.(X[10:13, ind] .+ 1.0)
        end
        return Matrix(X')
    end
    
    triangles = read_data(dataset, 80, 100)[1]
    simplices = dataset.simplices
    nverts = dataset.nverts
    times = dataset.times
    old_simplices, old_nverts = split_data(simplices, nverts, times, 80, 100)[1:2]
    A, At, B = basic_matrices(old_simplices, old_nverts)
    basename = basename_str(dataset.name)

    train_triangles, val_labels = read_data(dataset, 60, 80)
    train_simplices, train_nverts = split_data(simplices, nverts, times, 60, 80)[1:2]
    At_train, B_train = basic_matrices(train_simplices, train_nverts)[2:3]
    X_train = feature_matrix(train_triangles, At_train, B_train)
    model = LogisticRegression(fit_intercept=true, solver="liblinear")
    ScikitLearn.fit!(model, X_train, val_labels)
    X = feature_matrix(triangles, At, B)
    learned_scores = ScikitLearn.predict_proba(model, X)[:, 2]
    write_scores(dataset, "logreg_supervised", learned_scores)
end

function collect_Simplicial_PPR_combined_scores(dataset::HONData)
    triangles = read_data(dataset, 80, 100)[1]
    old_simplices, old_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 80, 100)[1:2]
    A = basic_matrices(old_simplices, old_nverts)[1]
    basename = basename_str(dataset.name)    

    (scores_comb, S_comb, edge_map) = Simplicial_PPR3_combined(triangles, A, true, 0.85)
    write_scores(dataset, "SimpPPR_comb", scores_comb)
    save("$basename-SimpPPR_comb.jld2",
         Dict("S" => S_comb, "edge_map" => edge_map))
end

function collect_Simplicial_PPR_decomposed_scores(dataset::HONData)
    triangles = read_data(dataset, 80, 100)[1]
    old_simplices, old_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 80, 100)[1:2]
    A = basic_matrices(old_simplices, old_nverts)[1]
    basename = basename_str(dataset.name)
    
    (scores_comb, scores_curl, scores_grad, scores_harm,
     S_comb,      S_curl,      S_grad,      S_harm, edge_map) =
         Simplicial_PPR3_decomposed(triangles, A, true, 0.85)
    write_scores(dataset, "SimpPPR_comb", scores_comb)
    write_scores(dataset, "SimpPPR_grad", scores_grad)
    write_scores(dataset, "SimpPPR_curl", scores_curl)
    write_scores(dataset, "SimpPPR_harm", scores_harm)           
    save("$basename-SimpPPR_comb.jld2",
         Dict("S" => S_comb, "edge_map" => edge_map))
    save("$basename-SimpPPR_grad.jld2",
         Dict("S" => S_grad, "edge_map" => edge_map))
    save("$basename-SimpPPR_curl.jld2",
         Dict("S" => S_curl, "edge_map" => edge_map))
    save("$basename-SimpPPR_harm.jld2",
         Dict("S" => S_harm, "edge_map" => edge_map))
end

function collect_generalized_means(dataset::HONData)
    triangles, labels = read_data(dataset, 80, 100)
    old_simplices, old_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 80, 100)[1:2]
    B = basic_matrices(old_simplices, old_nverts)[3]
    basename = basename_str(dataset.name)    

    ps = convert(Vector{Float64}, [-Inf; collect(-4:0.25:4); Inf])
    all_scores = generalized_means(triangles, B, ps)
    rand_rate = sum(labels .== 1) / length(labels)
    
    improvements = Float64[]
    for (r, p) in enumerate(ps)
        scores = all_scores[:, r]
        ave_prec = average_precision_score(labels, scores)
        improvement = ave_prec / rand_rate            
        push!(improvements, improvement)
        println("($p): $improvement")
    end
    save("$basename-genmeans-perf.jld2",
         Dict("improvements" => improvements, "ps" => ps))
    return (ps, improvements)
end

function evaluate(dataset::HONData, score_types::Vector{String})
    triangles, labels = read_data(dataset, 80, 100)
    rand_rate = sum(labels .== 1) / length(labels)
    @printf("random: %0.2e\n", rand_rate)
    for score_type in score_types
        scores = read_scores(dataset, score_type)
        ave_prec = average_precision_score(labels, scores)
        improvement = ave_prec / rand_rate
        @printf("%s: %0.2f\n", score_type, improvement)
    end
end

"""
top_predictions
---------------

Look at the top predictions of a given score function.

top_predictions(dataset::HONData, score_type::String, topk::Int64=10)

Input parameters:
- dataset::HONData: the dataset
- score_type::String: the score type (scores must have been precomputed)
- topk::Int64=10: the number of top predictions to examine
"""
function top_predictions(dataset::HONData, score_type::String, topk::Int64=10)
    triangles, labels = read_data(dataset, 80, 100)
    scores = read_scores(dataset, score_type)
    sp = sortperm(scores, alg=QuickSort, rev=true)
    node_labels = read_node_labels(dataset.name)
    for rank = 1:topk
        ind = sp[rank]
        i, j, k = triangles[ind]
        @printf("%d (%f; %d): %s; %s; %s\n", rank, scores[ind],
                labels[ind], node_labels[i], node_labels[j], node_labels[k])
    end
end

"""
collect_labeled_dataset
-----------------------

Collects the open triangles in the first 80% of the data as well as a label of
whether or not it closes.

collect_labeled_dataset(data::HONData)

Input parameters:
- dataset::HONData: The dataset.
"""
function collect_labeled_dataset(dataset::HONData)
    function write_dataset(old_simplices::Vector{Int64}, old_nverts::Vector{Int64},
                           new_simplices::Vector{Int64}, new_nverts::Vector{Int64},
                           output_name::String)
        new_closed_tris = new_closures(old_simplices, old_nverts, new_simplices, new_nverts)
        open_tris = enum_open_triangles(old_simplices, old_nverts)
        output_data = zeros(Int64, 4, length(open_tris))
        for (i, tri) in enumerate(open_tris)
            output_data[1:3, i] = collect(tri)
            output_data[4, i] = (tri in new_closed_tris)
        end
        basename = basename_str(dataset.name)
        save("$(OUTDIR)/$(dataset.name)-$(output_name).jld2",
             Dict("data" => output_data'))
    end
    
    old_simplices, old_nverts, new_simplices, new_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 80, 100)
    write_dataset(old_simplices, old_nverts, new_simplices, new_nverts,
                  "open-tris-80-100")
    train_simplices, train_nverts, val_simplices, val_nverts =
        split_data(dataset.simplices, dataset.nverts, dataset.times, 60, 80)
    write_dataset(train_simplices, train_nverts, val_simplices, val_nverts,
                  "open-tris-60-80")
end
;
