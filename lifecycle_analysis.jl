include("common.jl")

# NDC-classes
# 44 breast cancer resistance protein inhibitors [moa]
# 74 hiv protease inhibitors [moa]
# 76 ugt1a1 inhibitors [moa]
# u, v, w = 44, 74, 76

# tags-ask-ubuntu
# 55 colors
# 86 icons
# 1740 16.04
#u, v, w = 55, 86, 1740

# music-rap-genius
# 2551 Gucci Mane (13)
# 2615 Travis Scott (20185)
# 2628 Young Thug (20503)
#u, v, w = 2551, 2615, 2628

"""
lifecycle
---------

Prints out lifecycle information for a given triple of nodes.

lifecycle(dataset::HONData, u::Int64, v::Int64, w::Int64)

Input parameters
- dataset::HONData: the dataset
- u::Int64: the first node
- v::Int64: the second node
- w::Int64: the third node
"""
function lifecycle(dataset::HONData, u::Int64, v::Int64, w::Int64)
    names = Vector{String}()
    basename = dataset.name
    open("data/$basename/$basename-simplex-labels.txt") do f
        for line in eachline(f); push!(names, line); end
    end
    node_label_map = Dict{Int64, String}()
    open("data/$basename/$basename-node-labels.txt") do f
        for line in eachline(f)
            info = split(line)
            node_label_map[parse(Int64, info[1])] = join(info[2:end], " ")
        end
    end

    simplices = dataset.simplices
    nverts = dataset.nverts
    times = dataset.times
    A, At, B = basic_matrices(simplices, nverts)
    simplex_order = simplex_degree_order(At)
    triangle_order = proj_graph_degree_order(B)

    uv_simplices  = Int64[]
    uw_simplices  = Int64[]
    vw_simplices  = Int64[]
    uvw_simplices = Int64[]
    for simplex in nz_row_inds(At, u)
        # Check if v and w are in the simplex
        has_v = A[v, simplex] > 0
        has_w = A[w, simplex] > 0
        if  has_v && !has_w; push!(uv_simplices,  simplex); end
        if !has_v &&  has_w; push!(uw_simplices,  simplex); end
        if  has_v &&  has_w; push!(uvw_simplices, simplex); end
    end
    for simplex in nz_row_inds(At, v)
        has_u = A[u, simplex] > 0
        has_w = A[w, simplex] > 0
        if has_w && !has_u; push!(vw_simplices, simplex); end
    end

    closure_time = minimum([times[simplex] for simplex in uvw_simplices])
    keep_simplices = Int64[]
    before_closure(sid::Int64) = times[sid] <= closure_time
    append!(keep_simplices, filter(before_closure, uv_simplices))
    append!(keep_simplices, filter(before_closure, uw_simplices))
    append!(keep_simplices, filter(before_closure, vw_simplices))
    append!(keep_simplices, filter(before_closure, uvw_simplices))
    sort!(keep_simplices, by=(sid -> times[sid]))
    
    for simplex in keep_simplices
        simplex_nodes = [node_label_map[node] for node in nz_row_inds(A, simplex) if node in [u, v, w]]        
        simplex_nodes = join(simplex_nodes, "; ")
        simplex_name = names[simplex]
        println("$simplex_name: $simplex_nodes")
    end
end
;
