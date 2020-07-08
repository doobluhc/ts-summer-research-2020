using PyPlot
using PyCall
using Distributions
using LinearAlgebra
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
using ControlSystems
@sk_import linear_model: LinearRegression
@pyimport matplotlib.pyplot as pyplt

function sample(a::Float64,b::Float64;T = 100000)
    cost = zeros(T)
    gain = zeros(T)
    avg = zeros(T)
    regret = zeros(T)


    θ = [0.0;1.0]
    Σ = Symmetric([1.0 0.0;0.0 1.0])
    j_optimal = 1
    x = 0.0

    for t in 1:T

        if t == 1
            cost[t] = x * x
            regret[t] = cost[t] - j_optimal
            avg[t] = cost[t]
        else
            cost[t] = cost[t-1] + x*x
            regret[t] = regret[t-1] + x*x - j_optimal
            avg[t] = cost[t]/t
        end

        θ̂ = rand(MvNormal(θ, Σ))
        gain[t] = θ̂[1]/θ̂[2]
        u = -gain[t] * x
        z = [x;u]
        w = randn()
        x = a * x + b * u + w
        θ = θ + Σ*z*(x-z'*θ)/(1+z'*Σ*z)
        Σ = Σ - Symmetric(Σ*z*z'*Σ)/(1+z'*Σ*z)

    end
    return avg,gain,regret
end
function simulation(a::Float64,b::Float64;T = 100000, N = 1000)
    j_optimal = 1
    avg_cost  = [ zeros(T) for n = 1:N ]
    gain = [ zeros(T) for n = 1:N ]
    regret = [zeros(T) for n = 1:N]
    avg_regret = zeros(T)
    bot = zeros(T)
    top = zeros(T)
    pyplt.clf()
    for n in 1:N
        avg_cost[n],gain[n],regret[n] = sample(a,b)
        println(n)
        # pyplt.plot([t for t = 1:T],avg_cost[n])
        # pyplt.axis([0,T,0,10])
        # pyplt.xlabel("t")
        # pyplt.ylabel("cost/t")
        # pyplt.title("cost function value/t vs t (optimal cost = $j_optimal)")
        # pyplt.savefig("average cost vs t for TS.png")
    end

    #plot log(average regret) vs log t
    # for t = 1:T
    #     temp = zeros(N)
    #     for n = 1:N
    #         temp[n] = regret[n][t]
    #     end
    #     avg_regret[t] = quantile!(temp,0.5)
    #     bot[t] = quantile!(temp,0.25)
    #     top[t] = quantile!(temp,0.75)
    # end
    #
    # X = reshape([log(10,t) for t = 100:T],99901,1)
    # Y = reshape(avg_regret[100:T],99901,1)
    # regr = LinearRegression()
    # fit!(regr,X,Y)
    # y_pred = predict(regr,X)
    # slope = float(regr.coef_)
    # intercept = float(regr.intercept_)
    # pyplt.fill_between([sqrt(t) for t = 100:T],bot[100:T],top[100:T],color="gray")
    # pyplt.plot(X, Y, color ="blue")
    # pyplt.plot(X, y_pred, color ="red")
    # print(slope)
    # pyplt.xlabel("logt")
    # pyplt.ylabel("log(average regret)")
    # pyplt.title("log(average regret) vs log(t) for TS(slope = $slope)")
    # pyplt.savefig("log average regret vs log t TS.png")


    #plot average regret vs sqrt t
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
    pyplt.title("average regret vs sqrt t for TS(slope = $slope)")
    pyplt.savefig("average regret vs sqrt t TS.png")

end
