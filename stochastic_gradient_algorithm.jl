using DifferentialEquations
using LinearAlgebra
using PyPlot
using PyCall
using Statistics: mean
using DataFrames
using ControlSystems
@pyimport matplotlib.pyplot as pyplt


function sample(a₀::Float64,a₁::Float64,b₀::Float64,b₁::Float64;T = 10000)
    cost = zeros(T)
    regret = zeros(T)
    gain = zeros(T)
    avg_cost = zeros(T)
    j_optimal = tr(dare([a₀;a₁],[b₀;b₁],1.0,1.0))
    print("jopt",j_optimal)
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
            r = r + ϕ'[t]* ϕ[t]
            w = randn()
            x[t+1] = a₀ * x[t] + a₁ * x₀ + b₀ * u[t] + b₁ * u₀ + w
            gain[t] = -u[t]/x[t]
            cost[t] = x[t] * x[t]
        else
            u[t] = -1/θ[t][3] * (θ[t][1]*x[t] + θ[t][2]*x[t-1] + θ[t][4]*u[t-1])
            ϕ[t] = [x[t];x[t-1];u[t];u[t-1]]
            r = r + ϕ'[t] * ϕ[t]
            w = randn()
            x[t+1] = a₀ * x[t] + a₁ * x[t-1] + b₀ * u[t] + b₁ * u[t-1] + w
            gain[t] = -u[t]/x[t]
            cost[t] = cost[t-1] + x[t] * x[t]
        avg_cost[t] = cost[t]/t
        θ[t+1] = θ[t] + ϕ[t]/r * (x[t+1] - ϕ'[t] * θ[t])
        end
    end


    for t in 1:T
        if t == 1
            regret[t] = cost[t] - avg_cost[T]
        else
            regret[t] = regret[t-1] + cost[t] - avg_cost[T]
        end
    end

    return avg_cost,gain,regret

end

function simulation(a₀::Float64,a₁::Float64,b₀::Float64,b₁::Float64; T = 10000, N = 1000)
    avg_cost  = [ zeros(T) for n = 1:N ]
    gain = [ zeros(T) for n = 1:N ]
    regret = [ zeros(T) for n = 1:N ]
    avg_regret = zeros(T)
    sum = 0.0
    pyplt.clf()

    for n = 1:N
        avg_cost[n],gain[n],regret[n]= sample(a₀,a₁,b₀,b₁; T=T)


        # pyplt.plot([t for t = 1:T],avg[n])
        # pyplt.axis([0,T,0,10])
        # pyplt.xlabel("t")
        # pyplt.ylabel("cost/t")
        # pyplt.title("cost function value/t vs t ")
        # pyplt.savefig("cost function value.png")

        # pyplt.plot([t for t = 1:T],gain[n])
        # pyplt.axis([0,T,0,10])
        # pyplt.xlabel("t")
        # pyplt.ylabel("gain")
        # pyplt.title("gain vs t ")
        # pyplt.savefig("gain.png")

    end

    for t = 1:T
        for n = 1:N
            sum = sum + regret[n][t]
        avg_regret[t] = log(abs(sum/N))
        end
    end



    # pyplt.plot([log(t) for t = 1:T],avg_regret)
    # pyplt.xlabel("logt")
    # pyplt.ylabel("log(average regret)")
    # pyplt.title("log(average regret) vs log(t) ")
    # pyplt.savefig("average regret.png")

end
