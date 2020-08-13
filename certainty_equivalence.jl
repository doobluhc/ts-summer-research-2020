
using ControlSystems
using LinearAlgebra
using Statistics
using PyPlot
using ScikitLearn
using PyCall
using LsqFit
@pyimport matplotlib.pyplot as pyplt
@sk_import linear_model: LinearRegression


abstract type Algorithm end


mutable struct CE <: Algorithm
    A::Float64
    B::Float64
    Q::Float64
    R::Float64
end


function sample(A::Array{Float64},B::Array{Float64},Q::Array{Float64},R::Array{Float64},T::Int64,N::Int64)

    p = size(B,1)
    q = size(B,2)
    if size(B) == (size(B,1),)
        B = reshape(B,size(B,1),1)
    end
    if size(R) == (size(R,1),)
        R = reshape(R,size(R,1),1)
    end


    S = dare(A,B,Q,R)
    j_optimal = tr(S)
    # println("optimal cost",j_optimal)
    # println("optimal gain" , (R + B'*S*B)\(B'*S*A))
    global_avg_cost = [zeros(T) for n = 1:N]
    global_cost = [zeros(T) for n = 1:N]
    global_regret = [zeros(T) for n = 1:N]
    sum_gain = zeros(Float64,q,p)
    global_gain = [zeros(Float64,q,p) for n = 1:N]

    for n = 1:N
        x = zeros(Float64,p,1)
        u = ones(Float64,q,1)
        w = zeros(Float64,p,1)
        θ̂ = vcat(zeros(Float64,p,p),ones(Float64,q,p))
        Σ = Symmetric(Matrix{Float64}(I,p+q,p+q))
        cost = zeros(T)
        regret = zeros(T)
        avg_cost = zeros(T)
        gain = zeros(Float64,q,p)
        for t = 1:T
            if t == 1
                cost[t] = (x'*Q*x + u'*R*u)[1]
                regret[t] = (x'*Q*x + u'*R*u)[1] - j_optimal[1]
            else
                cost[t] = cost[t-1] + (x'*Q*x + u'*R*u)[1]
                regret[t] = regret[t-1] + (x'*Q*x + u'*R*u)[1] - j_optimal[1]
            end
            avg_cost[t] = cost[t] / t


            Â = reshape(θ̂[1:p,:],p,p)
            B̂ = reshape(θ̂[(p+1):(p+q),:],p,q)
            Ŝ = dare(Â,B̂,Q,R)
            gain = (R + B̂'*Ŝ*B̂)\(B̂'*Ŝ*Â)
            u = -gain * x
            w = randn(p,1)
            z = vcat(x,u)

            x = A * x + B * u + w

            normalize = 1 + (z' * Σ * z)[1]
            θ̂ = θ̂ + (Σ * z * (x - θ̂'*z)') / normalize
            Σ = Σ - Symmetric(Σ * z * z' * Σ) / normalize
        end
        global_gain[n] = gain
        global_cost[n] = cost
        global_avg_cost[n] = avg_cost
        global_regret[n] = regret
    end
    for n = 1:N
        sum_gain = sum_gain + global_gain[n]
    end
    avg_gain = N.\sum_gain
    # println("average estimated gain",avg_gain)

    return global_cost, global_avg_cost, global_regret
end


function plot_avg_regret_vs_t(data,T)
        data = reshape(hcat(data...),T,N)
        avg_regret = reshape(mean(data,dims=2),T,)
        pyplt.clf()
        X = [t for t = 100:T]
        Y = avg_regret[100:T]
        @. model(x,p) = p[1] * x^p[2]
        p0 = [1.0,1.0]
        fit = curve_fit(model,X,Y,p0)
        y_fit = [fit.param[1] * x^fit.param[2] for x in X]
        # pyplt.plot(X, Y, color ="blue")
        # pyplt.plot(X, y_fit, color ="red")
        # pyplt.xlabel("t")
        # pyplt.ylabel("average regret")
        # pyplt.title("average regret vs t for CE")
        # pyplt.savefig("average regret vs t CEa.png")
        # println(fit.param[1])
        # println(fit.param[2])

        return fit.param[1],fit.param[2]
end

function plot_avg_cost_vs_t(data,T,N)
    pyplt.clf()
    for n in 1:N
        pyplt.plot(reshape([t for t = 1:T],T,1),reshape(data[n],T,1))
        pyplt.axis([0,T,0,10])
        pyplt.xlabel("t")
        pyplt.ylabel("cost/t")
        pyplt.title("cost/t vs t ")
        pyplt.savefig("average cost vs t for CEa.png")
    end

end

function plot_log_avg_regret_vs_log_t(data,T)
    data = reshape(hcat(data...),T,N)
    avg_regret = reshape(mean(data,dims=2),T,)
    pyplt.clf()
    X = reshape(log.([t for t = 100:T]),(T-99),1)
    Y = reshape(log.(avg_regret[100:T]),(T-99),1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)

    # pyplt.plot(X, Y, color ="blue")
    # pyplt.plot(X, y_pred, color ="red")
    # pyplt.xlabel("logt")
    # pyplt.ylabel("log(average regret)")
    # pyplt.title("log(average regret) vs log(t) for CE(slope = $slope)")
    # pyplt.savefig("log average regret vs log t CEa.png")
    return slope

end

function plot_avg_regret_vs_sqrt_t(data,T)
    data = reshape(hcat(data...),T,N)
    avg_regret = reshape(mean(data,dims=2),T,)
    pyplt.clf()
    X = reshape([sqrt(t) for t = 100:T],(T-99),1)
    Y = reshape(avg_regret[100:T],(T-99),1)
    regr = LinearRegression()
    fit!(regr,X,Y)
    y_pred = predict(regr,X)
    slope = float(regr.coef_)
    intercept = float(regr.intercept_)
    # pyplt.fill_between([sqrt(t) for t = 100:T],bot[100:T],top[100:T],color="gray")
    # pyplt.plot(X, Y, color ="blue")
    # pyplt.plot(X, y_pred, color ="red")
    # pyplt.xlabel("sqrtt")
    # pyplt.ylabel("average regret")
    # pyplt.title("average regret vs sqrt t for CE(slope = $slope)")
    # pyplt.savefig("average regret vs sqrt t CEa.png")
    return slope
end

function test()
    I = 100
    T = 10000
    N = 10
    p = 3
    q = 3
    c_list = []
    α_list = []
    Q = [1.0 0.0 0.0;0.0 1.0 0.0;0.0 0.0 1.0]
    R = Q
    log_slope_list = []
    sqrt_slope_list = []
    for i in 1:I
        println(i)
        M = randn(p,p)
        A = M * M'
        B = A
        c = ctrb(A,B)
        if rank(c) == p
            data = sample(A,B,Q,R,T,N)
            push!(log_slope_list,plot_log_avg_regret_vs_log_t(data[3],T))
            push!(c_list,plot_avg_regret_vs_t(data[3],T)[1])
            push!(α_list,plot_avg_regret_vs_t(data[3],T)[2])
            push!(sqrt_slope_list,plot_avg_regret_vs_sqrt_t(data[3],T))
        end

    end

    return c_list,α_list,log_slope_list,sqrt_slope_list

end
