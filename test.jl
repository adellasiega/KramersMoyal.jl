using KramersMoyal
using Pkg
Pkg.add("Plots")
Pkg.add("DifferentialEquations")
Pkg.add("Statistics")
using Plots: plot, plot!, savefig, heatmap
using DifferentialEquations: SDEProblem, solve, EM, EnsembleProblem, EnsembleSummary, remake
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


# Define a time dependant SDE
# du = f(u,t)dt + g(u)dw
f(u, p, t) = -0.3 * u * t
g(u, p, t) = 0.1 * t

prob = SDEProblem(f, g, u0, tspan)
rand_initial_condition() = u0 .+ rand()
prob_func = (prob, i, repeat) -> remake(prob, u0 = rand_initial_condition())

ensemble_prob = EnsembleProblem(prob; prob_func = prob_func)
solution = solve(ensemble_prob, EM(), dt=dt, trajectories=500)
summary = EnsembleSummary(solution)
plot_summary_time = plot(summary)
savefig(plot_summary_time, "figures/ensemble_solution_time")
Y = Matrix(hcat([Float64.(sol.u) for sol in solution]...)')

time = Array(range(start=tspan[1], step=dt, stop=tspan[2]))
result = estimate_kramers_moyal_time_ensemble(Y, time)

# Create matrices on the same grid
F_true = [f(y, τ, τ) for y in result.y, τ in result.t]
G_true = [g(y, τ, τ) for y in result.y, τ in result.t]

F_est = result.coefficients[1]'/dt
G_est = sqrt.(result.coefficients[2]' .* 2 ./ dt)

F_min = -2.0#min(minimum(F_true), minimum(F_est))-0.1
F_max = 2.0#max(maximum(F_true), maximum(F_est))+0.1

G_min = 0.0 #min(minimum(G_true), minimum(G_est))-0.1
G_max = 1.0 #max(maximum(G_true), maximum(G_est))+0.1

# Drift
p1 = heatmap(result.y, result.t, F_est,
             xlabel="y", ylabel="t", title="Estimated Drift f(y,t)",
             clims=(F_min, F_max))
p2 = heatmap(result.y, result.t, F_true',
             xlabel="y", ylabel="t", title="True Drift f(y,t)",
             clims=(F_min, F_max))
h1 = plot(p1, p2, layout=(1,2), size=(900,400))
savefig(h1, "figures/ensemble_time_drift")

# Diffusion
q1 = heatmap(result.y, result.t, G_est,
             xlabel="y", ylabel="t", title="Estimated Diffusion g(y,t)",
             clims=(G_min, G_max))
q2 = heatmap(result.y, result.t, G_true',
             xlabel="y", ylabel="t", title="True Diffusion g(y,t)",
             clims=(G_min, G_max))
h2 = plot(q1, q2, layout=(1,2), size=(900,400),)
savefig(h2, "figures/ensemble_time_diffusion")
