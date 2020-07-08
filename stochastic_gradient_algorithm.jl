using DifferentialEquations
using LinearAlgebra
using PyPlot
using PyCall
using Statistics: mean
using DataFrames
using ControlSystems
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
@sk_import linear_model: LinearRegression
@pyimport matplotlib.pyplot as pyplt


function sample(a₀::Float64,a₁::Float64,b₀::Float64,b₁::Float64;T = 100000)
    cost = zeros(T)
    regret = zeros(T)
    gain = zeros(T)
    avg_cost = zeros(T)
    j_optimal = 1
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
            regret[t] = x[t] * x[t] - j_optimal
        else
            u[t] = -1/θ[t][3] * (θ[t][1]*x[t] + θ[t][2]*x[t-1] + θ[t][4]*u[t-1])
            ϕ[t] = [x[t];x[t-1];u[t];u[t-1]]
            r = r + ϕ'[t] * ϕ[t]
            w = randn()
            x[t+1] = a₀ * x[t] + a₁ * x[t-1] + b₀ * u[t] + b₁ * u[t-1] + w
            gain[t] = -u[t]/x[t]
            cost[t] = cost[t-1] + x[t] * x[t]
            regret[t] = regret[t-1] + x[t] * x[t] - j_optimal
        avg_cost[t] = cost[t]/t
        θ[t+1] = θ[t] + ϕ[t]/r * (x[t+1] - ϕ'[t] * θ[t])
        end
    end

    return avg_cost,gain,regret

end

function simulation(a₀::Float64,a₁::Float64,b₀::Float64,b₁::Float64; T = 100000, N = 1000)
    j_optimal = 1
    avg_cost  = [ zeros(T) for n = 1:N ]
    gain = [ zeros(T) for n = 1:N ]
    regret = [ zeros(T) for n = 1:N ]
    avg_regret = zeros(T)
    bot = zeros(T)
    top = zeros(T)
    pyplt.clf()

    for n = 1:N
        avg_cost[n],gain[n],regret[n]= sample(a₀,a₁,b₀,b₁; T=T)
        println(n)

        # pyplt.plot([t for t = 1:T],avg_cost[n])
        # pyplt.axis([0,T,0,10])
        # pyplt.xlabel("t")
        # pyplt.ylabel("cost/t")
        # pyplt.title("cost function value/t vs t (optimal cost = $j_optimal)")
        # pyplt.savefig("average cost vs t for SG.png")

        # pyplt.plot([t for t = 1:T],gain[n])
        # pyplt.axis([0,T,0,10])
        # pyplt.xlabel("t")
        # pyplt.ylabel("gain")
        # pyplt.title("gain vs t ")
        # pyplt.savefig("gain.png")

    end
    #plot log(average regret) vs log t
    # for t = 1:T
    #     temp = zeros(N)
    #     for n = 1:N
    #         temp[n] = regret[n][t]
    #     end
    #     if mean(temp) < 0
    #         avg_regret[t] = 0
    #     else
    #         avg_regret[t] = log(10,mean(temp))
    #     end
    # end
    #
    # X = reshape([log(10,t) for t = 100:T],9901,1)
    # Y = reshape(avg_regret[100:T],9901,1)
    # regr = LinearRegression()
    # fit!(regr,X,Y)
    # y_pred = predict(regr,X)
    # slope = float(regr.coef_)
    # intercept = float(regr.intercept_)
    # pyplt.scatter(X, Y, color ="blue")
    # pyplt.plot(X, y_pred, color ="red")
    # pyplt.text(2,1.4,"slope = $slope")
    # print(slope)
    # pyplt.xlabel("logt")
    # pyplt.ylabel("log(average regret)")
    # pyplt.title("log(average regret) vs log(t) for SG")
    # pyplt.savefig("log average regret vs log t SG.png")

    #plot average regret) vs sqrt t
    for t = 1:T
        temp = zeros(N)
        for n = 1:N
            temp[n] = regret[n][t]
        end
        avg_regret[t] = quantile!(temp,0.5)
        bot[t] = quantile!(temp,0.25)
        top[t] = quantile!(temp,0.75)
    end



    X = reshape([sqrt(t) for t = 100:T],(T-99),1)
    Y = reshape(avg_regret[100:T],(T-99),1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)
    pyplt.fill_between([sqrt(t) for t = 100:T],bot[100:T],top[100:T],color="gray")
    pyplt.plot(X, Y, color ="blue")
    pyplt.plot(X, y_pred, color ="red")
    print(slope)
    pyplt.xlabel("sqrtt")
    pyplt.ylabel("average regret")
    pyplt.title("average regret vs sqrt t for SG(slope = $slope)")
    pyplt.savefig("average regret vs sqrt t SG.png")

end
