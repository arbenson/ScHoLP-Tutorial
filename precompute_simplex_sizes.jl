include("common.jl")

function main()
    datasets = [row[1] for row in all_datasets_params()]
    for dataset in datasets
        println("$dataset...")
        if dataset != "congress-committees" && dataset != "music-rap-genius"
            hd = read_txt_data(dataset)
            nv_counts = Dict{Int64,Int64}()
            for v in hd.nverts
                if !haskey(nv_counts, v); nv_counts[v]  = 1
                else                      nv_counts[v] += 1
                end
            end
            nverts = [x[1] for x in nv_counts]
            counts = [x[2] for x in nv_counts]
            save("output/simplex-size-dists/$dataset-simplex-size-dist.jld2",
                 Dict("nvert" => nverts, "counts" => counts))
        end
    end
end
