include("common.jl")

using CSV
using HypothesisTests
using DataFrames
using Printf
using GLM

function simplicial_closure_tests(significance::Float64=1e-5, X::Int64=100, only_3_node::Bool=false)
    datasets = [param[1] for param in all_datasets_params()]
    density_tests3 = [(1, 2), (2, 4), (3, 5), (4, 7), (5, 8), (6, 9)]
    strength_tests3 = [(2, 3), (4, 5), (5, 6), (7, 8), (8, 9), (9, 10)]
    strength_density_tests3 = [(6, 7)]

    density_tests4 = [(1, 4), (2, 5), (3, 6),  # 3 --> 4
                      (4, 7), (5, 8), (6, 9),  # 4 --> 5
                      (7, 13), (8, 14), (9, 15), (10, 16), (11, 17), (12, 18)  # 5 --> 6
                      ]
    strength_tests4_3edge = [(1, 2), (2, 3)]
    strength_tests4_4edge = [(4, 5), (5, 6)]
    strength_tests4_5edge = [(7, 8), (8, 9), (10, 11), (11, 12), (8, 10), (9, 11)]
    strength_tests4_6edge = [(13, 14), (14, 15), (16, 17), (17, 18), (19, 20), (20, 21), (21, 22),
                             (23, 24), (24, 25), (25, 26), (26, 27), (14, 16), (15, 17), (16, 19),
                             (17, 20), (18, 21), (19, 23), (20, 24), (21, 25), (22, 26)]
    strength_density_tests4 = [(12, 19)]
    
    for (test_type, comparisons, simp_size) in [("density (3-node)", density_tests3, 3),
                                                ("tie strength (3-node)", strength_tests3, 3),
                                                ("strength vs. density (3-node)", strength_density_tests3, 3),
                                                ("density (4-node)", density_tests4, 4),
                                                ("tie strength (4-node, 3-edge)", strength_tests4_3edge, 4),
                                                ("tie strength (4-node, 4-edge)", strength_tests4_4edge, 4),
                                                ("tie strength (4-node, 5-edge)", strength_tests4_5edge, 4),
                                                ("tie strength (4-node, 6-edge)", strength_tests4_6edge, 4),
                                                ("strength vs. density (4-node)", strength_density_tests4, 4)]
        if only_3_node && simp_size != 3; continue; end
        sig_count1 = 0
        sig_count2 = 0
        raw = 0
        total = 0
        for dataset in datasets
            (keys, nsamples, nclosed) = read_closure_stats(dataset, simp_size, X)
            for (ind1, ind2) in comparisons
                k1, k2 = "$(keys[ind1])", "$(keys[ind2])"
                n1, n2 = nsamples[[ind1, ind2]]
                if min(n1, n2) <= 20; continue; end
                x1, x2 = nclosed[[ind1, ind2]]
                p1, p2 = nothing, nothing
                if max(x1, x2) <= 5
                    p1 = pvalue(FisherExactTest(x1, x2, n1, n2), tail=:left)
                    p2 = pvalue(FisherExactTest(x1, x2, n1, n2), tail=:right)
                else
                    phat = (x1 + x2) / (n1 + n2)
                    n = n1 + n2
                    stddev = sqrt(n * phat * (1 - phat) * (1.0 / n1 + 1.0 / n2))
                    p1 = pvalue(OneSampleZTest(x1 / n1 - x2 / n2, stddev, n), tail=:left)
                    p2 = pvalue(OneSampleZTest(x1 / n1 - x2 / n2, stddev, n), tail=:right)
                end
                if p1 < significance; sig_count1 += 1; end
                if p2 < significance; sig_count2 += 1; end
                if (test_type == "strength vs. density (3-node)" || test_type == "strength vs. density (4-node)")
                    if p1 < significance; println("$test_type (density more likely):  $dataset"); end
                    if p2 < significance; println("$test_type (strength more likely): $dataset"); end
                end                
                if x1 / n1 < x2 / n2; raw += 1; end
                total += 1
            end
        end
        @printf("%s (left): %d of %d tests significant at < %g\n",
                test_type, sig_count1, total, significance)
        @printf("%s (right): %d of %d tests significant at < %g\n",
                test_type, sig_count2, total, significance)
        @printf("%s (raw): %d of %d\n", test_type, raw, total)
    end
end

function fracopen_logavedeg_linear_models()
    datasets = [row[1] for row in all_datasets_params()]
    frac_open  = Float64[]
    ave_deg    = Float64[]
    frac_open3 = Float64[]
    ave_deg3   = Float64[]
    for dataset in datasets
        data = CSV.read("output/summary-stats/$dataset-statistics.csv")
        no = data[1, :nopentri]
        nc = data[1, :nclosedtri]
        push!(frac_open, no / (no + nc))
        pd = data[1, :projdensity]
        nn = data[1, :nnodes]
        push!(ave_deg, pd * (nn - 1))

        if dataset != "congress-committees"
            no = data[2, :nopentri]
            nc = data[2, :nclosedtri]
            push!(frac_open3, no / (no + nc))
            pd = data[2, :projdensity]
            nn = data[2, :nnodes]
            push!(ave_deg3, pd * (nn - 1))
        end
    end

    data  = DataFrame(X=log.(ave_deg),  Y=frac_open)
    data3 = DataFrame(X=log.(ave_deg3), Y=frac_open3)
    model  = lm(@formula(Y ~ X), data)
    model3 = lm(@formula(Y ~ X), data3)
    return (model, model3)
end
;
