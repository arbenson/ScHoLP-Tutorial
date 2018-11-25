include("common.jl")

using Distributions
using SparseArrays

function simulate_summary_stats(n::Int64, p::Float64)
    bin = Binomial(binomial(n, 3), p)
    num_closed = max(rand(bin), 1)
    
    simplices = Int64[]
    flips = rand(binomial(n, 3))
    total = 0
    for (ind, (i, j, k)) in enumerate(combinations(1:n, 3))
        if flips[ind] <= p
            push!(simplices, i, j, k)
            total += 1
        end
    end
    nverts = 3 * ones(Int64, total)
    
    A, At, B = basic_matrices(simplices, nverts)
    no, nc = num_open_closed_triangles(A, At, B)
    frac_open = no / (no + nc)
    density = nnz(B) / (n^2 - n)
    return frac_open, density
end

function simulate()
    all_n = Int64[]
    all_b = Float64[]
    all_frac_open = Float64[]
    all_density = Float64[]
    all_ave_deg = Float64[]
    for n in [25, 50, 100, 200]
        for b = 0.8:0.02:1.8
            println("$b...")
            p = 1.0 / n^b
            for _ in 1:5
                fo, de = simulate_summary_stats(n, p)
                ad = de * (n^2 - n) / n
                push!(all_n, n)
                push!(all_b, b)
                push!(all_frac_open, fo)
                push!(all_density, de)
                push!(all_ave_deg, ad)
            end
        end
    end
    save("output/simulation/simulation.jld2",
         Dict("n" => all_n, "b" => all_b, "density" => all_density,
              "ave_deg" => all_ave_deg, "frac_open" => all_frac_open))
end
