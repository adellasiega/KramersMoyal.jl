# KramersMoyal.jl

A Julia package for estimating Kramers-Moyal coefficients from time series data. The Kramers-Moyal expansion provides a systematic way to analyze stochastic processes by estimating drift and diffusion coefficients from empirical data.

## Features

- **Standard Kramers-Moyal estimation**: Estimate coefficients from single time series
- **Ensemble averaging**: Combine estimates from multiple trajectories with proper weighting
- **Time-dependent coefficients**: Analyze non-stationary processes with time-varying dynamics  
- **Flexible binning**: Customizable spatial and temporal discretization
- **Robust statistics**: Handles empty bins and missing data gracefully

## Installation

```julia
# Add the package (assuming it's registered or available locally)
using Pkg
Pkg.add("KramersMoyal")
```

## Quick Start

```julia
using KramersMoyal

# Generate sample data (e.g., from an Ornstein-Uhlenbeck process)
N = 10000
dt = 0.01
y = cumsum(randn(N)) * sqrt(dt)  # Random walk approximation

# Estimate Kramers-Moyal coefficients
result = estimate_kramers_moyal(y, lag=1, order=2, n_bins=50)

# Access results
y_bins = result.y                    # Bin centers
D1 = result.coefficients[1]          # First-order coefficient (drift)
D2 = result.coefficients[2]          # Second-order coefficient (diffusion)
counts = result.counts               # Sample counts per bin
```

## API Reference

### Core Functions

#### `estimate_kramers_moyal`

Estimate Kramers-Moyal coefficients from a single time series.

```julia
estimate_kramers_moyal(
    y::Vector{Float64},
    lag::Int=1,
    order::Int=2,
    n_bins::Int=100,
    y_min::Union{Nothing,Float64}=nothing,
    y_max::Union{Nothing,Float64}=nothing
)
```

**Parameters:**
- `y`: Time series data
- `lag`: Time lag for computing differences (default: 1)
- `order`: Maximum order of coefficients to estimate (default: 2)
- `n_bins`: Number of spatial bins (default: 100)
- `y_min`, `y_max`: Optional range limits for binning

**Returns:** Named tuple with:
- `y`: Bin centers
- `coefficients`: Dictionary mapping order → coefficient array
- `counts`: Sample counts per bin

#### `estimate_kramers_moyal_ensemble`

Ensemble average of Kramers-Moyal coefficients from multiple trajectories.

```julia
estimate_kramers_moyal_ensemble(
    Y::Matrix{Float64},
    lag::Int=1,
    order::Int=2,
    n_bins::Int=100
)
```

**Parameters:**
- `Y`: Matrix where each row is a trajectory
- Other parameters same as single trajectory case

**Returns:** Named tuple with ensemble-averaged coefficients.

#### `estimate_kramers_moyal_time`

Estimate time-dependent Kramers-Moyal coefficients D_n(y,t).

```julia
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
    t_max::Union{Nothing,Float64}=nothing
)
```

**Parameters:**
- `y`: Time series values
- `t`: Time points
- `n_y_bins`: Number of spatial bins
- `n_t_bins`: Number of temporal bins
- Range parameters for custom binning limits

**Returns:** Named tuple with:
- `y`: Spatial bin centers
- `t`: Temporal bin centers  
- `coefficients`: Dictionary of coefficient matrices (n_y_bins × n_t_bins)
- `counts`: Sample count matrix

#### `estimate_kramers_moyal_time_ensemble`

Ensemble average of time-dependent coefficients with sample-count weighting.

```julia
estimate_kramers_moyal_time_ensemble(
    Y::Matrix{Float64},
    t::Vector{Float64};
    lag::Int=1,
    order::Int=2,
    n_y_bins::Int=100,
    n_t_bins::Int=50
)
```

## Usage Examples

### Single Time Series Analysis

```julia
using KramersMoyal
using Plots

# Simulate Ornstein-Uhlenbeck process: dX = -θX dt + σ dW
θ, σ = 0.1, 0.5
dt = 0.01
N = 5000
X = zeros(N)
for i in 2:N
    X[i] = X[i-1] - θ*X[i-1]*dt + σ*sqrt(dt)*randn()
end

# Estimate coefficients
result = estimate_kramers_moyal(X, lag=1, order=2, n_bins=30)

# Plot drift and diffusion
plot(result.y, result.coefficients[1], label="D₁ (drift)", xlabel="X", ylabel="Coefficient")
plot!(result.y, result.coefficients[2], label="D₂ (diffusion)")

# Theoretical values: D₁(x) = -θx, D₂(x) = σ²/2
plot!(result.y, -θ*result.y, label="Theory D₁", linestyle=:dash)
hline!([σ²/2], label="Theory D₂", linestyle=:dash)
```

### Ensemble Analysis

```julia
# Generate multiple trajectories
n_traj = 50
Y = zeros(n_traj, N)
for i in 1:n_traj
    X = zeros(N)
    for j in 2:N
        X[j] = X[j-1] - θ*X[j-1]*dt + σ*sqrt(dt)*randn()
    end
    Y[i, :] = X
end

# Ensemble estimation with proper weighting
ensemble_result = estimate_kramers_moyal_ensemble(Y, lag=1, order=2, n_bins=30)

# Compare single vs ensemble estimates
plot(result.y, result.coefficients[1], label="Single trajectory", alpha=0.5)
plot!(ensemble_result.y, ensemble_result.coefficients[1], label="Ensemble average", linewidth=2)
```

### Time-Dependent Analysis

```julia
# Simulate process with time-varying parameters
t = (0:N-1) * dt
X = zeros(N)
for i in 2:N
    θ_t = 0.1 + 0.05*sin(0.1*t[i])  # Time-varying drift
    X[i] = X[i-1] - θ_t*X[i-1]*dt + σ*sqrt(dt)*randn()
end

# Estimate time-dependent coefficients
time_result = estimate_kramers_moyal_time(X, t, lag=1, order=2, n_y_bins=20, n_t_bins=25)

# Visualize with heatmap
using Plots
heatmap(time_result.t, time_result.y, time_result.coefficients[1], 
        xlabel="Time", ylabel="X", title="Time-dependent drift D₁(x,t)")
```

## Theory

The Kramers-Moyal expansion approximates a stochastic differential equation:

```
dX(t) = D₁[X,t]dt + √(2D₂[X,t])dW(t) + ...
```

The coefficients are estimated as:

```
D_n(x,t) = (1/n!) ⟨(ΔX)ⁿ⟩ / Δt
```

where `ΔX = X(t+Δt) - X(t)` and `⟨·⟩` denotes the conditional expectation given `X(t) = x, t=\tau`.

For the ensemble case, coefficients are weighted by sample counts to properly handle varying data density across bins and trajectories.

