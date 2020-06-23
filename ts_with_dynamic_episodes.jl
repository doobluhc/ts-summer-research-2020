using PyPlot
using PyCall
using Distributions
using Statistics
using LinearAlgebra
using LinearAlgebra: inv
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: predict
using ControlSystems
@sk_import linear_model: LinearRegression
@pyimport matplotlib.pyplot as pyplt

function sample(a::Float64,b::Float64;J = 1000)
    cost = Float64[]
    avg_cost = Float64[]
    regret = Float64[]
    j_optimal = tr(dare(a, b, 1.0, 1.0))
    θ̃ = [0.0;0.0]
    θ̂ = [a;b]
    Σ = Symmetric([1.0 0.0;0.0 1.0])
    u = 0.0
    x = 0.0
    t::Int64 = 1
    Gⱼ = 0.0
    tⱼ = 0
    iterations = 1

    for j in 1:J
        print(t)
        Tⱼ₋₁ = t - tⱼ
        tⱼ = t

        θ̃ = rand(MvNormal(θ̂,Σ))

        S = float(dare(θ̃[1],θ̃[2],1.0,1.0))
        Gⱼ = -(1.0+b'*S[1]*b)^-1*b'*S[1]*a
        Σⱼ = Σ

        while t <= (tⱼ + Tⱼ₋₁) && det(Σ[t]) >= 0.5*det(Σⱼ)
            if iterations == 1
                push!(cost,x*x)
                push!(regret,cost[iterations] - j_optimal)
                push!(avg_cost,cost[iterations])
            else
                push!(cost,cost[iterations-1] + x*x)
                push!(regret,regret[iterations-1]+cost[iterations]-j_optimal)
                push!(avg_cost,cost[iterations]/iterations)


            u = Gⱼ * x
            w = randn()
            x = a*x+b*u+w
            z = [x;u]
            θ̂ = θ̂ + (Σ*z*(x-z'*θ̂))/(1+z'*Σ*z)
            Σ = Σ - Symmetric((Σ*z*z'*Σ))/(1+z'*Σ*z)
            iterations = iterations + 1
            t += 1
        end
    end
    return avg_cost,regret,iterations

end

function simulation(a::Float64,b::Float64;T = 500, N = 10)
    avg_cost  = [ Float64[] for n = 1:N ]
    regret = [ Float64[] for n = 1:N ]
    iterations = zeros(N)
    avg_regret = zeros(T)
    pyplt.clf()
    for n in 1:N
        avg_cost[n],regret[n],iterations[n]= sample(a,b)
    end


    #plot log(average regret) vs log t
    # for t = 1:T
    #     temp = zeros(N)
    #     for n = 1
    #         temp[n] = regret[n][t]
    #     end
    #     if mean(temp) < 0
    #         avg_regret[t] = 0
    #     else
    #         avg_regret[t] = log(10,mean(temp))
    #     end
    # end
    #
    # X = reshape([log(t) for t = 100:T],9901,1)
    # Y = reshape(avg_regret[100:T],9901,1)
    # regr = LinearRegression()
    # fit!(regr,X,Y)
    # y_pred = predict(regr,X)
    # slope = float(regr.coef_)
    # intercept = float(regr.intercept_)
    # pyplt.scatter(X, Y, color ="blue")
    # pyplt.plot(X, y_pred, color ="red")
    # pyplt.text(5,7.5,"slope = $slope")
    # pyplt.text(5,6.5,"intercept = $intercept")
    # pyplt.xlabel("logt")
    # pyplt.ylabel("log(average regret)")
    # pyplt.title("log(average regret) vs log(t) for TS")
    # pyplt.savefig("average  cost vs log t TSDE.png")


end
