using DifferentialEquations
using LinearAlgebra
using PyPlot
using PyCall
@pyimport matplotlib.pyplot as pyplt


function sample(a₀::Float64,a₁::Float64,b₀::Float64,b₁::Float64;T = 1000)
    cost = zeros(T)
    gain = zeros(T)
    avg = zeros(T)
    u = zeros(T)
    u₀ = 1
    x = zeros(T+1)
    x₀ = 1
    θ = [[a₀;a₁;b₀;b₁] for t = 1:(T+1)]
    ϕ = [[0.0;0.0;0.0;0.0] for t = 1:T]
    r = 0

    for t in 1:T
        if t == 1
            u[t] = -1/θ[t][3] * (θ[t][1]*x[t] + θ[t][2]*x₀ + θ[t][4]*u₀)
            ϕ[t] = [x[t];x₀;u[t];u₀]
            r = r + transpose(ϕ[t]) * ϕ[t]
            w = randn()
            x[t+1] = a₀ * x[t] + a₁ * x₀ + b₀ * u[t] + b₁ * u₀ + w
            gain[t] = -u[t]/x[t]
            cost[t] = x[t] * x[t]
        else
            u[t] = -1/θ[t][3] * (θ[t][1]*x[t] + θ[t][2]*x[t-1] + θ[t][4]*u[t-1])
            ϕ[t] = [x[t];x[t-1];u[t];u[t-1]]
            r = r + transpose(ϕ[t]) * ϕ[t]
            w = randn()
            x[t+1] = a₀ * x[t] + a₁ * x[t-1] + b₀ * u[t] + b₁ * u[t-1] + w
            gain[t] = -u[t]/x[t]
            cost[t] = cost[t-1] + x[t] * x[t]
        avg[t] = cost[t]/t
        θ[t+1] = θ[t] + ϕ[t]/r * (x[t+1] - transpose(ϕ[t]) * θ[t])
        end
    end
    return avg,gain


end

function simulation(a₀::Float64,a₁::Float64,b₀::Float64,b₁::Float64; T = 1000, N = 1)
    avg  = [ zeros(T) for n = 1:N ]
    gain = [ zeros(T) for n = 1:N ]

    for n = 1:N
        avg[n],gain[n]= sample(a₀,a₁,b₀,b₁; T=T)

        pyplt.plot([t for t = 1:T],avg[n])
        pyplt.axis([0,T,0,10])
        pyplt.xlabel("t")
        pyplt.ylabel("cost/t")
        pyplt.title("cost function value/t vs t ")
        pyplt.savefig("cost function value.png")

        pyplt.plot([t for t = 1:T],gain[n])
        pyplt.axis([0,T,0,10])
        pyplt.xlabel("t")
        pyplt.ylabel("gain")
        pyplt.title("gain vs t ")
        pyplt.savefig("gain.png")

    end

end
