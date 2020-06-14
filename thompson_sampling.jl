using PyPlot
using PyCall
using Distributions
using LinearAlgebra
@pyimport matplotlib.pyplot as pyplt

function sample(a::Float64,b::Float64;T = 1000)
    cost = zeros(T)
    gain = zeros(T)
    avg = zeros(T)
    θ = [[0.0;1.0] for t = 1:(T+1)]
    θ̂ = [[0.0;1.0] for t = 1:T]
    Σ = [Symmetric([1.0 0.0;0.0 1.0]) for t = 1:(T+1)]
    z = [[0.0;0.0] for t = 1:T]
    u = zeros(T+1)
    x = zeros(T+1)

    for t in 1:T
        θ̂[t] = rand(MvNormal(θ[t], Σ[t]))
        gain[t] = θ̂[t][1]/θ̂[t][2]
        u[t] = -gain[t] * x[t]
        z[t] = [x[t];u[t]]
        w = randn()
        x[t+1] = a * x[t] + b * u[t] + w
        θ[t+1] = θ[t] + (Σ[t]*z[t]*(x[t+1]-transpose(z[t])*θ[t]))/(1+transpose(z[t])*Σ[t]*z[t])
        Σ[t+1] = Σ[t] - Symmetric((Σ[t]*z[t]*transpose(z[t])*Σ[t]))/(1+transpose(z[t])*Σ[t]*z[t])
        if t == 1
            cost[t] = x[t] * x[t]
            avg[t] = cost[t]
        else
            cost[t] = cost[t-1] + x[t]*x[t]
            avg[t] = cost[t]/t
        end
    end
    return avg,gain
end
function simulate(a::Float64,b::Float64;T = 1000, N = 100)
    avg  = [ zeros(T) for n = 1:N ]
    gain = [ zeros(T) for n = 1:N ]
    pyplt.clf()
    for n in 1:N
        avg[n],gain[n] = sample(a,b)
        plot_data(avg[n])
    end

end

function plot_data(data;T = 1000)
    pyplt.plot([t for t = 1:T],data)
    pyplt.axis([0,T,0,10])
    pyplt.xlabel("t")
    pyplt.ylabel("cost/t")
    pyplt.title("cost/t vs t ")
    pyplt.savefig("average cost vs t ts.png")

end
