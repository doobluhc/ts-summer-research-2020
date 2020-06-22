using PyPlot
using PyCall
using ControlSystems
using LinearAlgebra
using Statistics
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
@sk_import linear_model: LinearRegression
@sk_import model_selection: train_test_split
@pyimport matplotlib.pyplot as pyplt


function sample(a::Float64, b::Float64; T = 10000)
    cost = zeros(T)
    regret = zeros(T)
    avg_cost = zeros(T)
    gain = zeros(T)
    j_optimal = tr(dare(a, b, 1.0, 1.0))

    x = zeros(T + 1)
    u = zeros(T)
    θ̂ = [[a; b] for t = 1:(T+1)]
    Σ = [[1.0 0.0; 0.0 1.0] for t = 1:(T+1)]
    â = zeros(T)
    b̂ = zeros(T)

    for t = 1:T
        â[t] = θ̂[t][1]
        b̂[t] = θ̂[t][2]
        gain[t] = θ̂[t][1] / θ̂[t][2]
        u[t] = -gain[t] * x[t]
        w = randn()
        z = [x[t]; u[t]]

        x[t+1] = a * x[t] + b * u[t] + w

        if t == 1
            cost[t] = x[t] * x[t]
            regret[t] = cost[t] - j_optimal
        else
            cost[t] = cost[t-1] + x[t] * x[t]
            regret[t] = regret[t-1] + cost[t] - j_optimal
        end

        avg_cost[t] = cost[t] / t

        θ̂[t+1] = θ̂[t] + Σ[t] * z * (x[t+1] - z' * θ̂[t]) / (1 + z' * Σ[t] * z)
        Σ[t+1] = Σ[t] - Σ[t] * z * z' * Σ[t] / (1 + z' * Σ[t] * z)
    end


    return avg_cost, gain, â, b̂, regret

end

function simulation(a::Float64, b::Float64; T = 10000, N = 100)
    avg_cost = [zeros(T) for n = 1:N]
    gain = [zeros(T) for n = 1:N]
    â = [zeros(T) for n = 1:N]
    b̂ = [zeros(T) for n = 1:N]
    regret = [zeros(T) for n = 1:N]
    sum = 0.0
    avg_regret = zeros(T)
    pyplt.clf()
    for n = 1:N
        avg_cost[n], gain[n], â[n], b̂[n], regret[n] = sample(a, b)

        # pyplt.plot([t for t = 1:T],avg_cost[n])
        # pyplt.axis([0,T,0,100])
        # pyplt.xlabel("t")
        # pyplt.ylabel("cost/t")
        # pyplt.title("cost function value/t vs t ")
        # pyplt.savefig("average cost vs t for CE.png")
    end

    #plot log(average regret) vs log t
    for t = 1:T
        temp = zeros(N)
        for n = 1:N
            temp[n] = regret[n][t]
        end
        if mean(temp) < 0
            avg_regret[t] = 0
        else
            avg_regret[t] = log(10,mean(temp))
        end
    end

    X = reshape([log(t) for t = 100:T],9901,1)
    Y = reshape(avg_regret[100:T],9901,1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)
    pyplt.scatter(X, Y, color ="blue")
    pyplt.plot(X, y_pred, color ="red")
    pyplt.text(5,7.5,"slope = $slope")
    pyplt.text(5,6.5,"intercept = $intercept")
    pyplt.xlabel("logt")
    pyplt.ylabel("log(average regret)")
    pyplt.title("log(average regret) vs log(t) for CE")
    pyplt.savefig("average cost vs log t CE.png")



end
