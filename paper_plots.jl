include("common.jl")

using CSV
using DataFrames
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch

using ScikitLearn
@sk_import linear_model: LogisticRegression

function dataset_structure_plots()
    plot_params = all_datasets_params()
    datasets = [row[1] for row in plot_params]
    
    frac_open  = Float64[]
    density    = Float64[]
    ave_deg    = Float64[]
    frac_open3 = Float64[]
    density3   = Float64[]
    ave_deg3   = Float64[]
    for dataset in datasets
        data = CSV.read("output/summary-stats/$dataset-statistics.csv")
        no = data[1, :nopentri]
        nc = data[1, :nclosedtri]
        push!(frac_open, no / (no + nc))
        pd = data[1, :projdensity]
        nn = data[1, :nnodes]
        push!(density, pd)
        push!(ave_deg, pd * (nn - 1))

        no = data[2, :nopentri]
        nc = data[2, :nclosedtri]
        push!(frac_open3, no / (no + nc))
        push!(density3, data[1, :projdensity])
        pd = data[2, :projdensity]
        nn = data[2, :nnodes]
        push!(density3, pd)
        push!(ave_deg3, pd * (nn - 1))
    end

    PyPlot.pygui(true)
    close()
    markers = [row[2] for row in plot_params]
    colors  = [row[3] for row in plot_params]

    fsz=10
    # Fraction open triangles vs. density
    subplot(221)
    for i in 1:length(datasets)
        semilogx(density[i], frac_open[i], markers[i], color=colors[i])
    end
    xlabel("Edge density in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("25 or fewer nodes per simplex", fontsize=fsz)

    # Fraction open triangles vs. density
    subplot(223)
    for i in 1:length(datasets)
        semilogx(ave_deg[i], frac_open[i], markers[i], color=colors[i])
    end
    xlabel("Average degree in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("25 or fewer nodes per simplex", fontsize=fsz)

    # Fraction open triangles vs. density (just 3-node simplices)
    subplot(222)
    for i in 1:length(datasets)
        # Skip congress-committees, which has no 3-node simplices
        if datasets[i] != "congress-committees"
            semilogx(density3[i], frac_open3[i], markers[i], color=colors[i])
        end
    end
    xlabel("Edge density in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("Exactly 3 nodes per simplex", fontsize=fsz)

    subplot(224)
    for i in 1:length(datasets)
        # Skip congress-committees, which has no 3-node simplices
        if datasets[i] != "congress-committees"
            semilogx(ave_deg3[i], frac_open3[i], markers[i], color=colors[i])
        end
    end
    xlabel("Average degree in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("Exactly 3 nodes per simplex", fontsize=fsz)

    tight_layout()
    savefig("dataset-structure.pdf")
    show()
end

function simulation_plot()
    data = load("output/simulation/simulation.jld2")
    all_n         = data["n"]
    all_b         = data["b"]
    all_density   = data["density"]
    all_ave_deg   = data["ave_deg"]
    all_frac_open = data["frac_open"]
    
    close()
    figure()
    # Edge density
    for (n, cm, marker) in [(200, ColorMap("Purples"), "d"),
                            (100, ColorMap("Reds"),    "<"),
                            (50,  ColorMap("Greens"),  "s"),
                            (25,  ColorMap("Blues"),   "o"),
                            ]
        inds = findall(all_n .== n)
        curr_b    = all_b[inds]
        density   = all_density[inds]
        frac_open = all_frac_open[inds]
        scatter(density, frac_open, c=curr_b, marker=marker, label="$n", s=14,
                vmin=minimum(curr_b) - 0.5, vmax=maximum(curr_b) + 0.5, cmap=cm)
        
    end
    ax = gca()
    ax[:set_xscale]("log")
    fsz = 20
    ax[:tick_params]("both", labelsize=fsz-5, length=5, width=1)
    legend()
    xlabel("Edge density in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    tight_layout()
    savefig("simulation.pdf", bbox_inches="tight")
    show()
end

function simplex_size_dist_plot()
    plot_params = all_datasets_params()
    datasets = [row[1] for row in plot_params]
    markers  = [row[2] for row in plot_params]
    colors   = [row[3] for row in plot_params]
    
    # Simplex size distribution
    subplot(221)
    for i in 1:length(datasets)
        dataset = datasets[i]
        if dataset != "congress-committees" && dataset != "music-rap-genius"        
            data = load("output/simplex-size-dists/$dataset-simplex-size-dist.jld2")
            nvert = data["nvert"]
            counts = data["counts"]
            tot = sum(counts)
            fracs = [count / tot for count in counts]
            ms = 4
            loglog(nvert, fracs, marker=markers[i], color=colors[i],
                   linewidth=0.5, markersize=ms)
        end
    end
    fsz = 10
    xlabel("Number of nodes in simplex", fontsize=fsz)
    ylabel("Fraction of simplices", fontsize=fsz)
    title("Simplex size distribution", fontsize=fsz)

    # Fake subplot for legend
    subplot(223)
    for i in 1:length(datasets)
        plot(0.5, 0.5, markers[i], color=colors[i], label=datasets[i])
    end
    legend(datasets, fontsize=4)
    tight_layout()
    savefig("simplex-size-dist.pdf")
    show()
end

function min_max_val(probs1::Vector{Float64}, probs2::Vector{Float64})
    probs = [probs1; probs2]
    return (minimum([p for p in probs if p > 0]), maximum(probs))
end

function closure_probs_heat_map(simplex_size::Int64, initial_cutoff::Int64=100)
    plot_params = all_datasets_params()
    datasets = [param[1] for param in plot_params]

    keys, nsamples, nclosed = read_closure_stats(datasets[1], simplex_size, initial_cutoff)
    probs = nclosed ./ nsamples
    P = zeros(length(datasets), length(keys))
    insufficient_sample_inds = []
    for (ind, dataset) in enumerate(datasets)
        keys, nsamples, nclosed = read_closure_stats(dataset, simplex_size, initial_cutoff)
        P[ind, :] = nclosed ./ nsamples
        for (key_ind, (key, nsamp)) in enumerate(zip(keys, nsamples))
            if nsamp <= 20
                println("$dataset: $key has only $nsamp samples.")
                push!(insufficient_sample_inds, (ind, key_ind))
            end
        end
    end
    close()
    PyPlot.pygui(true)

    minval = max(1e-9, minimum([v for v in P[:] if v > 0]))
    P[P[:] .== 0] .= minval
    for (i, j) in insufficient_sample_inds; P[i, j] = 0; end

    cm = ColorMap("Blues")
    gray = (0.8, 0.8, 0.8, 1)
    cm[:set_bad](color=gray)
    
    maxval = 1
    imshow(P, norm=matplotlib[:colors][:LogNorm](vmin=minval, vmax=maxval),
           cmap=cm)
    ax = gca()
    # Add patches for bad cases
    for (i, j) in insufficient_sample_inds
        ax[:add_patch](patch.Rectangle((j-1.5, i-1.5), 1, 1, hatch="///////", fill=true, snap=false,
                                       facecolor=gray))
    end
    ax[:set_yticks](0:(length(datasets)-1))
    ax[:set_yticklabels](datasets, rotation=10, fontsize=(simplex_size == 4 ? 4 : 7))
    ax[:set_xticks](0:(length(probs)-1))
    ax[:tick_params](axis="both", length=3)
    ax[:set_xticklabels](["" for _ in 0:(length(probs)-1)])
    cb = colorbar()
    cb[:ax][:tick_params](labelsize=9)
    tight_layout()
    savefig("closure-probs-$(simplex_size)-$(initial_cutoff).pdf")
    show()
end

function three_node_scatter_plot()
    plot_params = all_datasets_params()
    datasets = [row[1] for row in plot_params]
    probs011  = Float64[]
    probs022  = Float64[]
    probs111  = Float64[]
    probs112  = Float64[]
    for dataset in datasets
        nsamples, nclosed = read_closure_stats(dataset, 3)[2:3]
        probs = nclosed ./ nsamples
        push!(probs011, probs[4])
        push!(probs022, probs[6])
        push!(probs111, probs[7])
        push!(probs112, probs[8])
    end

    markers = [row[2] for row in plot_params]
    colors = [row[3] for row in plot_params]

    close()
    fsz=10
    PyPlot.pygui(true)
    subplot(221)
    minval, maxval = min_max_val(probs011, probs111)
    loglog([minval, maxval], [minval, maxval], "black", lw=0.5)
    for i in 1:length(datasets)
        loglog(probs111[i], probs011[i], markers[i], color=colors[i])
    end
    xlabel("Closure probability (111)", fontsize=fsz)
    ylabel("Closure probability (011)", fontsize=fsz)    
    
    subplot(222)
    minval, maxval = min_max_val(probs112, probs111)
    loglog([minval, maxval], [minval, maxval], "black", lw=0.5)
    for i in 1:length(datasets)
        loglog(probs112[i], probs111[i], markers[i], color=colors[i])        
    end
    xlabel("Closure probability (112)", fontsize=fsz)
    ylabel("Closure probability (111)", fontsize=fsz)    

    subplot(223)
    minval, maxval = min_max_val(probs022, probs111)
    loglog([minval, maxval], [minval, maxval], "black", lw=0.5)
    for i in 1:length(datasets)
        loglog(probs022[i], probs111[i], markers[i], color=colors[i])
    end
    xlabel("Closure probability (022)", fontsize=fsz)
    ylabel("Closure probability (111)", fontsize=fsz)    

    tight_layout()
    savefig("closure-prob-scatter-3.pdf")
    show()
end

function four_node_scatter_plot()
    plot_params = all_datasets_params()
    datasets = [row[1] for row in plot_params]
    markers  = [row[2] for row in plot_params]
    colors   = [row[3] for row in plot_params]

    probs00   = Float64[]
    probs22   = Float64[]
    probs0000 = Float64[]
    probs0001 = Float64[]
    probs0111 = Float64[]    
    for (i, dataset) in enumerate(datasets)
        nsamples, nclosed = read_closure_stats(dataset, 4)[2:3]
        probs = nclosed ./ nsamples
        push!(probs00,   probs[7])
        push!(probs22,   probs[12])
        push!(probs0000, probs[13])
        push!(probs0001, probs[14])
        push!(probs0111, probs[19])
    end

    fsz=10
    close()
    PyPlot.pygui(true)
    subplot(221)
    minval, maxval = min_max_val(probs00, probs0000)
    loglog([minval, maxval], [minval, maxval], "black", lw=0.5)
    for i in 1:length(datasets)
        loglog(probs0000[i], probs00[i], markers[i], color=colors[i])
    end
    xlabel("Closure probability (0000)", fontsize=fsz)
    ylabel("Closure probability (00)", fontsize=fsz)    

    subplot(222)
    minval, maxval = min_max_val(probs0000, probs0001)
    loglog([minval, maxval], [minval, maxval], "black", lw=0.5)
    for i in 1:length(datasets)
        loglog(probs0001[i], probs0000[i], markers[i], color=colors[i])        
    end
    xlabel("Closure probability (0001)", fontsize=fsz)
    ylabel("Closure probability (0000)", fontsize=fsz)    

    subplot(223)
    minval, maxval = min_max_val(probs0111, probs22)
    loglog([minval, maxval], [minval, maxval], "black", lw=0.5)
    for i in 1:length(datasets)
        loglog(probs22[i], probs0111[i], markers[i], color=colors[i])
    end
    xlabel("Closure probability (0111)", fontsize=fsz)
    ylabel("Closure probability (22)", fontsize=fsz)    

    tight_layout()
    savefig("closure-prob-scatter-4.pdf")
    show()
end

function generalized_means_plot()
    close()
    fsz=10
    function make_subplot(datasets)
        axvline(x=-1, ls="--", color="black", lw=1.0, label="harmonic")
        axvline(x=0,  ls="-",  color="black", lw=1.0, label="geometric")
        axvline(x=1,  ls=":",  color="black", lw=1.0, label="arithmetic")
        for param in all_datasets_params()
            dataset = param[1]
            if dataset in datasets
                basename = "output/generalized-means/$dataset-open-tris-80-100"
                data = load("$basename-genmeans-perf.jld2")
                ps = data["ps"]
                improvements = data["improvements"]
                plot(ps[2:end-1], improvements[2:end-1],
                     marker=param[2], ms=3, lw=0.5, color=param[3])
            end
        end
        ylabel("Relative performance", fontsize=fsz)
        xlabel("p", fontsize=fsz)
        ax = gca()
        ax[:set_xticks](-4:1:4)
        ax[:tick_params](axis="both", length=3)
    end

    set1 = ["threads-stack-overflow", "threads-math-sx", "threads-ask-ubuntu"]
    set2 = ["tags-stack-overflow", "tags-math-sx", "tags-ask-ubuntu",
            "contact-high-school", "contact-primary-school",
            "DAWN", "NDC-substances", "NDC-classes"]
    set3 = ["coauth-MAG-History", "coauth-MAG-Geology", "coauth-DBLP",
            "email-Enron", "email-Eu", "congress-bills"]
    subplot(221)
    make_subplot(set1)
    legend(fontsize=fsz)
    subplot(222)
    make_subplot(set2)    
    subplot(223)
    make_subplot(set3)    
    tight_layout()
    savefig("generalized-means-perf.pdf")
    show()
end

function logreg_decision_boundary(trial::Int64=1)
    (X, _, y, _, yf, _) = egonet_train_test_data(trial)
    X = X[:, [LOG_AVE_DEG, FRAC_OPEN]]
    model = LogisticRegression(fit_intercept=true, multi_class="multinomial", C=10,
                               solver="newton-cg", max_iter=1000)
    ScikitLearn.fit!(model, X, y)

    dim = 500
    minval1, maxval1 = minimum(X[:, 1]) - 0.5, maximum(X[:, 1]) * 1.02
    minval2, maxval2 = minimum(X[:, 2]) - 0.01, maximum(X[:, 2]) + 0.05
    grid_feats = zeros(Float64, 2, dim * dim)
    grid_ind = 1
    xx = [(i - 1) * (maxval1 - minval1) / dim + minval1 for i in 1:dim]
    yy = [(j - 1) * (maxval2 - minval2) / dim + minval2 for j in 1:dim]
    for x in xx, y in yy
        grid_feats[1, grid_ind] = x
        grid_feats[2, grid_ind] = y
        grid_ind += 1
    end

    close()
    figure()
    Z = reshape(ScikitLearn.predict(model, Matrix(grid_feats')), dim, dim)
    labels = Dict(0 => "coauthorship", 1 => "tags", 2 => "threads",
                  3 => "contact",      4 => "email")
    greys = ["#f7f7f7", "#d9d9d9", "#bdbdbd", "#969696", "#636363"]
    contourf(exp.(xx), yy, Z, colors=greys)
    params = all_datasets_params()
    label2domain = Dict(0  => 0,  1  => 0,  2  => 0,
                        3  => -1,
                        4  => 1,  5  => 1,  6  => 1,
                        7  => 2,  8  => 2,  9  => 2,
                        10 => -1, 11 => -1, 12 => -1, 13 => -1, 14 => -1,
                        15 => 3,  16 => 3,
                        17 => 4,  18 => 4)
    colors_full = ["#ed5e5f", "#e41a1c", "#9f1214", 
                   "no-op",
                   "#69a3d2", "#377eb8", "#25567d", 
                   "#80c87d", "#4daf4a", "#357933", 
                   "no-op", "no-op", "no-op", "no-op", "no-op",
                   "#984ea3", "#68356f",
                   "#d37a48", "#a65628"]
    for label in sort(unique(yf))
        inds = findall(yf .== label)
        scatter(exp.(X[inds, 1]), X[inds, 2],
                color=colors_full[label],
                marker="o",
                label=params[label][1],
                s=14)
    end
    fsz = 18
    #legend(fontsize=fsz-4)
    ax = gca()
    ax[:set_xscale]("log")
    xlabel("Average degree", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("Decision boundary", fontsize=fsz)
    ax[:set_xlim](1.8, 400)
    ax[:tick_params](axis="both", length=3, labelsize=14)
    tight_layout()
    savefig("decision.pdf")
    show()
end
;
