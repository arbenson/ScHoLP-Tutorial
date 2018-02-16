include("common.jl")

using DataFrames
using HypothesisTests
using MAT
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch

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
        data = readtable("output/$dataset-statistics.csv")
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

function simulation_plots()
    data = matread("output/simulation.mat")
    all_n         = data["n"]
    all_b         = data["b"]
    all_density   = data["density"]
    all_ave_deg   = data["ave_deg"]
    all_frac_open = data["frac_open"]
    
    close()

    # Edge density
    subplot(221)
    for (n, cm, marker) in [(200, ColorMap("Purples"), "d"),
                            (100, ColorMap("Reds"),    "<"),
                            (50,  ColorMap("Greens"),  "s"),
                            (25,  ColorMap("Blues"),   "o"),
                            ]
        inds = find(all_n .== n)
        curr_b    = all_b[inds]
        density   = all_density[inds]
        frac_open = all_frac_open[inds]
        scatter(density, frac_open, c=curr_b, marker=marker, label="$n", s=6,
                vmin=minimum(curr_b) - 0.5, vmax=maximum(curr_b) + 0.5, cmap=cm,)
    end
    ax = gca()
    ax[:set_xscale]("log")
    fsz = 10
    xlabel("Edge density in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("Exactly 3 nodes per simplex (simulated)", fontsize=fsz)

    # Average degree
    subplot(223)
    for (n, cm, marker) in [(200, ColorMap("Purples"), "d"),
                            (100, ColorMap("Reds"),    "<"),
                            (50,  ColorMap("Greens"),  "s"),
                            (25,  ColorMap("Blues"),   "o"),
                            ]
        inds = find(all_n .== n)
        curr_b    = all_b[inds]
        ave_deg   = all_ave_deg[inds]
        frac_open = all_frac_open[inds]
        scatter(ave_deg, frac_open, c=curr_b, marker=marker, label="$n", s=6,
                vmin=minimum(curr_b) - 0.5, vmax=maximum(curr_b) + 0.5, cmap=cm)
    end
    ax = gca()
    ax[:set_xscale]("log")
    fsz = 10
    xlabel("Average degree in projected graph", fontsize=fsz)
    ylabel("Fraction of triangles open", fontsize=fsz)
    title("Exactly 3 nodes per simplex (simulated)", fontsize=fsz)    

    # legend
    subplot(224)
    for (n, color, marker) in [(200, "purple", "d"),
                               (100, "red",    "<"),
                               (50,  "green",  "s"),
                               (25,  "blue",   "o"),
                               ]
        scatter([1], [1], marker=marker, color=color, label="n = $n")
    end
    legend()
    
    tight_layout()
    savefig("model-structure.pdf")
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
        (_, nverts, _) = read_txt_data(dataset)
        nvert, counts = zip(sort(countmap(nverts))...)
        tot = sum(counts)
        fracs = [count / tot for count in counts]
        ms = (length(dataset) > 8 && dataset[1:8] == "congress") ? 6 : 2
        loglog(nvert, fracs, marker=markers[i], color=colors[i],
               linewidth=0.5, markersize=ms)
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
    legend(names, fontsize=4)
    tight_layout()
    savefig("simplex-size-dist.pdf")
end

function min_max_val(probs1::Vector{Float64}, probs2::Vector{Float64})
    probs = [probs1; probs2]
    return (minimum([p for p in probs if p > 0]), maximum(probs))
end

function closure_probs_heat_map(simplex_size::Int64)
    plot_params = all_datasets_params()
    datasets = [param[1] for param in plot_params]

    keys, nsamples, nclosed = read_closure_stats(datasets[1], simplex_size)
    probs = nclosed ./ nsamples
    P = zeros(length(datasets), length(keys))
    insufficient_sample_inds = []
    for (ind, dataset) in enumerate(datasets)
        keys, nsamples, nclosed = read_closure_stats(dataset, simplex_size)
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
    P[P[:] .== 0] = minval
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
    ax[:set_yticklabels](datasets, rotation=10, fontsize=(simplex_size == 4 ? 4 : 5))
    ax[:set_xticks](0:(length(probs)-1))
    ax[:tick_params](axis="both", length=3)
    ax[:set_xticklabels](["" for _ in 0:(length(probs)-1)])
    cb = colorbar(orientation="horizontal")
    cb[:ax][:tick_params](labelsize=(simplex_size == 4 ? 18 : 17))
    tight_layout()
    savefig("topological-closure-probs-$(simplex_size)-nodes.pdf")
    show()
end

function closure_probs_line(simplex_size::Int64)
    close()
    PyPlot.pygui(true)

    probs = nothing
    for param in all_datasets_params()
        keys, nsamples, nclosed = read_closure_stats(param[1], simplex_size)
        probs = nclosed ./ nsamples
        for (key_ind, (key, nsamp)) in enumerate(zip(keys, nsamples))
            if nsamp <= 20; probs[key_ind] = 0; end
        end
        semilogy(collect(0:(length(probs)-1)), probs, label=param[1],
                 marker=param[2], ms=4,
                 lw=0.5, color=param[3])
    end

    ax = gca()    
    ax[:set_xticks](0:(length(probs)-1))
    ax[:tick_params](axis="both", length=3)
    ax[:set_xticklabels](["" for _ in 0:(length(probs)-1)])
    ylabel("Closure probability")
    tight_layout()
    savefig("closure-probs-$(simplex_size).pdf")
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
    savefig("closure-prob-scatter.pdf")
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
        loglog(probs0111[i], probs22[i], markers[i], color=colors[i])
    end
    xlabel("Closure probability (0111)", fontsize=fsz)
    ylabel("Closure probability (22)", fontsize=fsz)    

    tight_layout()
    savefig("closure-prob-scatter-4-node.pdf")
    show()
end

function gen_means_plot()
    close()
    fsz=10
    function make_subplot(datasets)
        axvline(x=-1, ls="--", color="black", lw=1.0, label="harmonic")
        axvline(x=0,  ls="-",  color="black", lw=1.0, label="geometric")
        axvline(x=1,  ls=":",  color="black", lw=1.0, label="arithmetic")
        for param in all_datasets_params()
            dataset = param[1]
            if dataset in datasets
                basename = "prediction-output/$dataset-open-triangles-80-100"
                data = matread("$basename-genmeans-perf.mat")
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
        
    set1 = ["threads-stack-overflow-25", "threads-math-sx", "threads-ask-ubuntu"]
    set2 = ["tags-stack-overflow", "tags-math-sx", "tags-ask-ubuntu", "music-rap-genius-25",
            "contact-high-school", "contact-primary-school",
            "DAWN", "NDC-substances-25", "NDC-classes-25"]
    set3 = ["coauth-MAG-History-25", "coauth-MAG-Geology-25", "coauth-DBLP-25",
            "email-Enron-25", "email-Eu-25", "congress-committees-25", "congress-bills-25"]
    subplot(221)
    make_subplot(set1)
    legend(fontsize=fsz)
    subplot(222)
    make_subplot(set2)    
    subplot(223)
    make_subplot(set3)    
    
    tight_layout()
    savefig("genmeans-perf.pdf")
    show()
end

function structure_3d()
    plot_params = all_datasets_params()
    datasets = [row[1] for row in plot_params]
    
    frac_open  = Float64[]
    nodes      = Float64[]
    ave_deg    = Float64[]
    frac_open3 = Float64[]
    nodes3     = Float64[]
    ave_deg3   = Float64[]
    for dataset in datasets
        data = readtable("output/$dataset-statistics.csv")
        no = data[1, :nopentri]
        nc = data[1, :nclosedtri]
        push!(frac_open, no / (no + nc))
        pd = data[1, :projdensity]
        nn = data[1, :nnodes]
        push!(nodes, nn)
        push!(ave_deg, pd * (nn - 1))

        no = data[2, :nopentri]
        nc = data[2, :nclosedtri]
        push!(frac_open3, no / (no + nc))
        pd = data[2, :projdensity]
        nn = data[2, :nnodes]
        push!(nodes3, nn)
        push!(ave_deg3, pd * (nn - 1))
    end

    PyPlot.pygui(true)    
    close()
    markers = [row[2] for row in plot_params]
    colors  = [row[3] for row in plot_params]

    fsz=10
    for i in 1:length(datasets)
        scatter3D([log(nodes3[i])], [log(ave_deg3[i])], [frac_open3[i]], marker=markers[i], c=colors[i], s=[30])
    end
    xlabel("Log number of nodes", fontsize=fsz)
    ylabel("Log average degree", fontsize=fsz)    
    zlabel("Fraction of triangles open", fontsize=fsz)
    title("Exactly 3 nodes per simplex", fontsize=fsz)

    tight_layout()
    savefig("data-3d-only3.pdf")
    show()
end
