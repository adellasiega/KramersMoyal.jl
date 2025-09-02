using KramersMoyal
using Pkg
Pkg.add("Plots")
Pkg.add("DifferentialEquations")
Pkg.add("Statistics")
using Plots
using DifferentialEquations
using Statistics

# Define a SDE
# du = f(u)dt + g(u)dw
f(u, p, t) = -0.3*u
g(u, p, t) = 0.1

# Set integration conditions
u0 = 0.0
dt = 0.001 
tspan = (0.0, 10.0)

# Set SDEProblem
prob = SDEProblem(f, g, u0, tspan)

## Single Trajectory
solution = solve(prob, EM(), dt=dt)
Y = solution.u
results = estimate_kramers_moyal(Y, 2)

# Plot results
plot_trajectory = plot(Y)

# Drift: estimated vs true
plot_drift = plot(
    results.y,
    results.coefficients[1] ./ dt,
    label = "Estimated drift",
    title = "Drift comparison",
    xlabel = "y",
    ylabel = "Drift"
)
plot!(results.y, [f(y, 0, 0) for y in results.y], label = "True drift")

# Diffusion: estimated vs true
plot_diff = plot(
    results.y,
    sqrt.(results.coefficients[2] .* 2 ./ dt),
    label = "Estimated diffusion",
    title = "Diffusion comparison",
    xlabel = "y",
    ylabel = "Diffusion"
)
plot!(results.y, [g(y, 0, 0) for y in results.y], label = "True diffusion")

# Save figures
savefig(plot_trajectory, "figures/single_solution")
savefig(plot_drift, "figures/single_estimation_drift")
savefig(plot_diff, "figures/single_estimation_diffusion")


## Ensemble of Trajectories
# Define how to randomize initial conditions
rand_initial_condition() = u0 .+ rand()
prob_func = (prob, i, repeat) -> remake(prob, u0 = rand_initial_condition())

ensemble_prob = EnsembleProblem(prob; prob_func = prob_func)
solution = solve(ensemble_prob, EM(), dt=dt, trajectories=500)
summary = EnsembleSummary(solution)
Y = Matrix(hcat([Float64.(sol.u) for sol in solution]...)')

# Perform KM estimation
results = estimate_kramers_moyal_ensemble(Y, 2, 2, 100)

# Plot results
plot_summary = plot(summary)

# Drift: estimated vs true
plot_drift = plot(
    results.y,
    results.coefficients[1] ./ dt,
    label = "Estimated drift",
    title = "Drift comparison",
    xlabel = "y",
    ylabel = "Drift"
)
plot!(results.y, [f(y, 0, 0) for y in results.y], label = "True drift")

# Diffusion: estimated vs true
plot_diff = plot(
    results.y,
    sqrt.(results.coefficients[2] .* 2 ./ dt),
    label = "Estimated diffusion",
    title = "Diffusion comparison",
    xlabel = "y",
    ylabel = "Diffusion"
)
plot!(results.y, [g(y, 0, 0) for y in results.y], label = "True diffusion")

# Save figures
savefig(plot_summary, "figures/ensemble_solution")
savefig(plot_drift, "figures/ensemble_estimation_drift")
savefig(plot_diff, "figures/ensemble_estimation_diffusion")

#=
# Plot results
plot_summary = plot(summary)

plot_drift_est = scatter(results.y, results.coefficients[1]/dt, title="Estimated drift")
plot_drift_true = scatter(results.y, [f(y, 0, 0) for y in results.y], title="True drift")
plot_drift = plot(plot_drift_est, plot_drift_true, layout=(1,2), size=(900,400))

plot_diff_est = scatter(results.y, results.coefficients[2]/dt, title="Estimated diffusion")
plot_diff_true = scatter(results.y, [g(y, 0, 0) for y in results.y], title="True diffusion")
plot_diff = plot(plot_diff_est, plot_diff_true, layout=(1,2), size=(900,400))

savefig(plot_summary, "figures/ensemble_solution")
savefig(plot_drift, "figures/estimation_drift")
savefig(plot_diff, "figures/estimation_diffusion")
=#

#=
# Define a time dependant SDE
# du = f(u)dt + g(u)dw
f(u, p, t) = -0.3 * u + 0.9 * t
g(u, p, t) = 0.2

time = Array(range(start=tspan[1], step=dt, stop=tspan[2]))
result = estimate_kramers_moyal_time_ensemble(Y, time)

# Create matrices on the same grid
F_true = [f(y, τ, τ) for y in result.y, τ in result.t]
G_true = [g(y, τ, τ) for y in result.y, τ in result.t]

# Drift
p1 = heatmap(result.y, result.t, result.coefficients[1]'/dt,
             xlabel="y", ylabel="t", title="Estimated Drift f(y,t)")
p2 = heatmap(result.y, result.t, F_true',
             xlabel="y", ylabel="t", title="True Drift f(y,t)")
h1 = plot(p1, p2, layout=(1,2), size=(900,400))
savefig(h1, "figures/time_drift")

# Diffusion
q1 = heatmap(result.y, result.t, result.coefficients[2]'/dt,
             xlabel="y", ylabel="t", title="Estimated Diffusion g(y,t)")
q2 = heatmap(result.y, result.t, G_true',
             xlabel="y", ylabel="t", title="True Diffusion g(y,t)")
h2 = plot(q1, q2, layout=(1,2), size=(900,400))
savefig(h2, "figures/time_diff")
=#
