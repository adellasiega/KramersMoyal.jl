module KramersMoyal

export estimate_kramers_moyal, estimate_kramers_moyal_ensemble, estimate_kramers_moyal_time, estimate_kramers_moyal_time_ensemble

using Statistics

"""
    estimate_kramers_moyal(
        y::Vector{Float64},
        lag::Float64, 
        order::Int=2,
        bins::Int=100)

Estimate Kramers-Moyal coefficients of a given time series 'y' using 'lag' to compute differences.
Returns a dictionary of 'order => coefficient array'.
"""

function estimate_kramers_moyal(y::Vector{Float64},
                                 lag::Int=1,
                                 order::Int=2,
                                 n_bins::Int=100,
                                 y_min::Union{Nothing,Float64}=nothing,
                                 y_max::Union{Nothing,Float64}=nothing)

    N = length(y)
    if N <= lag
        error("Time series too short for chosen lag")
    end

    y_min = isnothing(y_min) ? minimum(y) : y_min
    y_max = isnothing(y_max) ? maximum(y) : y_max

    bin_edges = range(y_min, y_max, length=n_bins+1)
    bin_centers = 0.5 .* (bin_edges[1:end-1] .+ bin_edges[2:end])

    # assign each state to a bin
    bin_ids = clamp.(searchsortedlast.(Ref(bin_edges), y), 1, n_bins)

    # prepare bins
    Δy_bins = [Float64[] for _ in 1:n_bins]

    # collect increments
    for t in 1:(N-lag)
        b = bin_ids[t]
        push!(Δy_bins[b], y[t+lag] - y[t])
    end

    # KM coefficients
    D = Dict{Int, Vector{Float64}}()
    
    # Counts for evaluating weights
    counts = zeros(Int, n_bins)

    for i in 1:n_bins
        Δy = Δy_bins[i]
        counts[i] = length(Δy)
    end

    for n in 1:order
        D[n] = Float64[]
        for i in 1:n_bins
            Δy = Δy_bins[i]
            counts[i] = length(Δy)
            if !isempty(Δy)
                push!(D[n], mean(Δy.^n)/(factorial(n)*lag))
            else
                push!(D[n], 0.0)  # empty bins → 0
            end
        end
    end

    return (y=bin_centers, coefficients=D, counts=counts)
end

function estimate_kramers_moyal_ensemble(Y::Matrix{Float64},
                                         lag::Int=1,
                                         order::Int=2,
                                         n_bins::Int=100)

    # global min/max for consistent binning
    y_min = minimum(Y)
    y_max = maximum(Y)
    n_trajectories = size(Y,1)

    # compute per-trajectory KM coefficients
    results = [estimate_kramers_moyal(Y[i,:], lag, order, n_bins, y_min, y_max)
               for i in 1:n_trajectories]

    y = results[1].y
    n_y = length(y)

    ensemble_coeffs = Dict{Int, Vector{Float64}}()
    
    for n in 1:order
        coeffs = zeros(n_y)
        weights = zeros(n_y)

        for r in results
            c = r.coefficients[n]
            w = r.counts
            for i in 1:n_y
                if !isnan(c[i])
                    coeffs[i] += c[i] * w[i]
                    weights[i] += w[i]
                end
            end
        end

        # normalize weighted average
        coeffs ./= max.(weights, 1)  # avoid division by 0
        ensemble_coeffs[n] = coeffs
    end

    # ensemble average
    #for n in 1:order
    #    all_coeffs = hcat([r.coefficients[n] for r in results]...)
    #    ensemble_coeffs[n] = mean(all_coeffs, dims=2)[:]
    #end

    return (y=y, coefficients=ensemble_coeffs)
end

"""
    estimate_kramers_moyal_time(
        y::Vector{Float64},
        t::Vector{Float64},
        lag::Int=1,
        order::Int=2,
        n_y_bins::Int=100,
        n_t_bins::Int=50)

Estimate time-dependent Kramers–Moyal coefficients D(y,t).
Returns (y_centers, t_centers, coefficients::Dict{Int, Matrix}).
Each coefficient[n] is an n_y_bins × n_t_bins matrix.
"""
function estimate_kramers_moyal_time(
    y::Vector{Float64},
    t::Vector{Float64},
    lag::Int=1,
    order::Int=2,
    n_y_bins::Int=100,
    n_t_bins::Int=50
)

    N = length(y)
    @assert length(t) == N "y and t must have same length"

    # Bin edges
    y_edges = range(minimum(y), stop=maximum(y), length=n_y_bins+1)
    y_centers = 0.5 .* (y_edges[1:end-1] .+ y_edges[2:end])

    t_edges = range(minimum(t), stop=maximum(t), length=n_t_bins+1)
    t_centers = 0.5 .* (t_edges[1:end-1] .+ t_edges[2:end])

    # Initialize dict of coefficient matrices
    coeffs = Dict{Int, Matrix{Float64}}()
    for n in 1:order
        coeffs[n] = zeros(n_y_bins, n_t_bins)
    end

    # Loop over bins
    for i in 1:n_y_bins
        for j in 1:n_t_bins
            # Find indices belonging to this (y,t) bin
            bin_idx = findall(
                (y .>= y_edges[i]) .& (y .< y_edges[i+1]) .&
                (t .>= t_edges[j]) .& (t .< t_edges[j+1])
            )

            if !isempty(bin_idx)
                # Discard indices too close to end for forward lag
                valid_idx = filter(k -> k <= N-lag, bin_idx)

                Δy = y[valid_idx .+ lag] .- y[valid_idx]

                for n in 1:order
                    coeffs[n][i,j] = sum(Δy.^n) / (length(Δy)*factorial(n)*lag)
                end
            end
        end
    end

    return (y=y_centers, t=t_centers, coefficients=coeffs)
end

function estimate_kramers_moyal_time_ensemble(
    Y::Matrix{Float64}, 
    t::Vector{Float64}, 
    lag::Int=1, 
    order::Int=2, 
    n_y_bins::Int=100, 
    n_t_bins::Int=50
)
    n_traj = size(Y, 1)

    # Compute per-trajectory results
    results = [estimate_kramers_moyal_time(Y[i, :], t, lag, order, n_y_bins, n_t_bins) for i in 1:n_traj]

    y = results[1].y
    τ = results[1].t
    n_y = length(y)
    n_t = length(τ)

    # Init ensemble dict
    coeffs = Dict{Int, Matrix{Float64}}()
    for n in 1:order
        coeffs[n] = zeros(n_y, n_t)
    end

    # Average over trajectories
    for n in 1:order
        stacked = cat([r.coefficients[n] for r in results]..., dims=3) # shape (n_y, n_t, n_traj)
        coeffs[n] .= mean(stacked, dims=3)[:, :, 1]
    end

    return (y=y, t=τ, coefficients=coeffs)
end


end # module KramersMoyal
