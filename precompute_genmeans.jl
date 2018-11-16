include("open_triangle_prediction.jl")

function main()
    datasets = [row[1] for row in all_datasets_params()]
    for dataset in datasets
        println("$dataset...")
        if dataset != "congress-committees" && dataset != "music-rap-genius"
            hd = read_txt_data(dataset)
            collect_labeled_dataset(hd)
            collect_generalized_means(hd)
        end
    end
end
