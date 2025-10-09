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
        t::Vector{Float64};
        lag::Int=1,
        order::Int=2,
        n_y_bins::Int=100,
        n_t_bins::Int=50,
        y_min::Union{Nothing,Float64}=nothing,
        y_max::Union{Nothing,Float64}=nothing,
        t_min::Union{Nothing,Float64}=nothing,
        t_max::Union{Nothing,Float64}=nothing,
    )

Estimate time-dependent Kramers–Moyal coefficients D_n(y,t).

Returns:
    (y = y_centers,
     t = t_centers,
     coefficients = Dict{Int, Matrix{Float64}},   # each n_y_bins × n_t_bins
     counts = Matrix{Int})                        # sample counts per bin
"""
function estimate_kramers_moyal_time(
    y::Vector{Float64},
    t::Vector{Float64};
    lag::Int=1,
    order::Int=2,
    n_y_bins::Int=100,
    n_t_bins::Int=50,
    y_min::Union{Nothing,Float64}=nothing,
    y_max::Union{Nothing,Float64}=nothing,
    t_min::Union{Nothing,Float64}=nothing,
    t_max::Union{Nothing,Float64}=nothing,
)

    N = length(y)
    @assert length(t) == N "y and t must have same length"
    @assert N > lag "Time series too short for chosen lag"

    # Bin edges (optionally fixed to external ranges)
    ylo = isnothing(y_min) ? minimum(y) : y_min
    yhi = isnothing(y_max) ? maximum(y) : y_max
    tlo = isnothing(t_min) ? minimum(t) : t_min
    thi = isnothing(t_max) ? maximum(t) : t_max

    y_edges = range(ylo, stop=yhi, length=n_y_bins+1)
    t_edges = range(tlo, stop=thi, length=n_t_bins+1)
    y_centers = 0.5 .* (y_edges[1:end-1] .+ y_edges[2:end])
    t_centers = 0.5 .* (t_edges[1:end-1] .+ t_edges[2:end])

    # Allocate coefficient matrices and counts
    coeffs = Dict{Int, Matrix{Float64}}()
    for n in 1:order
        coeffs[n] = fill(NaN, n_y_bins, n_t_bins)
    end
    counts = zeros(Int, n_y_bins, n_t_bins)

    # Precompute bin ids for each sample (y,t)
    # clamp to ensure edge cases fall into last bin
    y_ids = clamp.(searchsortedlast.(Ref(y_edges), y), 1, n_y_bins)
    t_ids = clamp.(searchsortedlast.(Ref(t_edges), t), 1, n_t_bins)

    # For each bin, collect valid indices k such that k ≤ N - lag
    # Use vectors of vectors to avoid repeated findall
    idx_bins = [Int[] for _ in 1:(n_y_bins*n_t_bins)]
    @inbounds for k in 1:(N-lag)
        bi = y_ids[k]
        bj = t_ids[k]
        push!(idx_bins[(bj-1)*n_y_bins + bi], k)
    end
    """ 
    # Compute per-bin statistics
    @inbounds for j in 1:n_t_bins, i in 1:n_y_bins
        bin_list = idx_bins[(j-1)*n_y_bins + i]
        m = length(bin_list)
        counts[i, j] = m
        if m == 0
            continue
        end
        Δy = y[bin_list .+ lag] .- y[bin_list]
    
        for n in 1:order
            coeffs[n][i, j] = mean(Δy .^ n) / (factorial(n) * lag)
        end
    end
    """
    # Compute per-bin statistics
    @inbounds for j in 1:n_t_bins, i in 1:n_y_bins
        bin_list = idx_bins[(j-1)*n_y_bins + i]
        m = length(bin_list)
        counts[i, j] = m
        if m == 0
            continue
        end
        Δy = y[bin_list .+ lag] .- y[bin_list]

        # first-order (drift) -- compute and store
        d1 = mean(Δy) / (1 * lag)   # factorial(1)=1
        coeffs[1][i, j] = d1

        # higher orders
        for n in 2:order
            if n == 2
                # subtract τ * D^{(1)} before squaring
                centered = Δy .- (lag .* d1)
                coeffs[2][i, j] = mean(centered .^ 2) / (factorial(2) * lag)
            else
                # fallback: raw moment / (n! * lag)
                # (for n>2 you'd normally use central moments or Kramers–Moyal definition)
                coeffs[n][i, j] = mean(Δy .^ n) / (factorial(n) * lag)
            end
        end
    end
     

    return (y=y_centers, t=t_centers, coefficients=coeffs, counts=counts)
end


"""
    estimate_kramers_moyal_time_ensemble(
        Y::Matrix{Float64},
        t::Vector{Float64};
        lag::Int=1,
        order::Int=2,
        n_y_bins::Int=100,
        n_t_bins::Int=50
    )

Ensemble-average time-dependent Kramers–Moyal coefficients using
sample-count weighting across trajectories.

Returns:
    (y, t, coefficients::Dict{Int, Matrix{Float64}}, counts::Matrix{Int})
"""
function estimate_kramers_moyal_time_ensemble(
    Y::Matrix{Float64},
    t::Vector{Float64};
    lag::Int=1,
    order::Int=2,
    n_y_bins::Int=100,
    n_t_bins::Int=50
)
    n_traj, N = size(Y)
    @assert length(t) == N "Each trajectory must have same length as t"
    @assert N > lag "Time series too short for chosen lag"

    # Global ranges to ensure consistent binning across trajectories
    y_min = minimum(Y)
    y_max = maximum(Y)
    t_min = minimum(t)
    t_max = maximum(t)

    # Per-trajectory estimates with shared binning
    results = Vector{Any}(undef, n_traj)
    for r in 1:n_traj
        results[r] = estimate_kramers_moyal_time(
            Y[r, :], t;
            lag=lag, order=order,
            n_y_bins=n_y_bins, n_t_bins=n_t_bins,
            y_min=y_min, y_max=y_max, t_min=t_min, t_max=t_max
        )
    end

    y = results[1].y
    τ = results[1].t
    n_y = length(y)
    n_t = length(τ)

    # Weighted ensemble average: weight = per-bin count
    coeffs = Dict{Int, Matrix{Float64}}()
    total_counts = zeros(Int, n_y, n_t)

    for n in 1:order
        num = zeros(n_y, n_t)
        wts = zeros(Int, n_y, n_t)
        for r in results
            C = r.coefficients[n]
            W = r.counts
            # accumulate only where C is finite
            @inbounds for j in 1:n_t, i in 1:n_y
                w = W[i, j]
                if w > 0 && isfinite(C[i, j])
                    num[i, j] += C[i, j] * w
                    wts[i, j] += w
                end
            end
        end
        total_counts .+= wts
        # finalize averages; leave NaN where total weight is zero
        avg = fill(NaN, n_y, n_t)
        @inbounds for j in 1:n_t, i in 1:n_y
            if wts[i, j] > 0
                avg[i, j] = num[i, j] / wts[i, j]
            end
        end
        coeffs[n] = avg
    end

    return (y=y, t=τ, coefficients=coeffs, counts=total_counts)
end

end # module KramersMoyal
