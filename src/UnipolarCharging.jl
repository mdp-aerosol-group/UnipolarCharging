module UnipolarCharging

using QuadGK
using Interpolations
using DifferentialEquations
using Distributions
using StatsBase

export βe, βd, Tc, charges, analyticalDiffusion, analyticalField

const e = 1.60217662e-19     # Elementary charge [Coulombs]
const kb = 1.380649e-23      # Boltzmann constant [J K-1]
const KE = 9.0 * 1e9         # constant of proportionality [N.m2/C2]
const Zi = 1.4e-4            # average mobility of ion m2/Vs
const charges = 0:1:299 |> collect
const ncharges = 0:1:300 |> collect
const Dps = (2.0:1.0:1100.0) .* 1e-9 |> collect

function calculateFuchsBeta(Dp, n, T, p)
    # Dp: particle diameter in meter; 
    # The function calculates the charge distribution 
    # J = dNp,.n/dt = Bn-1*Np,n-1*Ni-Bn*Np,n*Ni (eqn. 8, Biskos et al. 2005)
    # --for unipolar, monodisperse particles 
    # Beta values are calculated from Fuchs limiting sphere theory

    ci = 462.5 * sqrt(T) / sqrt((273.15 + 21.0))  # Mean thermal speed of ions [m/s],
    # Fuchs used 462.5 for T = 21oC; 
    # converted by Eqn. 11 in Biskos for 300T
    diel = 1.00059         # Dielectric constant for air, Hinds 1999 pg 327
    dielv = 1              # Dielectric constant for vacuum, Hinds 1999 pg 327
  
    ki = Zi * p/101315.0 
    D_i = kb * T * ki / e  # m2/s Diffusion coeff Fuchs 1963

    a = Dp / 2             # Particle radius [m]
    r = 30 * a             # Distance from particle center
    γ = 1                  # ~ γ = α = 1  Table 1 Fuchs (1963) between 
    # from Biskos et al. 
    λ = 13.5 * 1e-9 * (101315.0/p) * (T/300.0) * (1 + 110.0/300.)/(1 + 111.0/T) 

    δ1 = ((1 + (λ / a))^5) / 5
    δ2 = ((1 + (λ^2 / a^2)) * (1 + (λ / a))^3) / 3
    δ3 = 2 / 15 * (1 + λ^2 / a^2)^2.5
    δ = (a^3 / λ^2) * (δ1 - δ2 + δ3)  # ~ equal or slightly smaller than a+λ 
    γ = a^2 / δ^2
   
    kappa = (diel - 1) * e^2 / (diel + 1)

    # ϕ(r) in Eqn. 1 in Biskos et al 2005
    potentialofion(n, r) = KE * ((n * e^2 / r) - (kappa * a^3 / (2 * r^2 * (r^2 - a^2))))

    integrald = Float64[]   # Integral from infinity to r in Eqn. 1 in Biskos et al 2005
    denominator0 = quadgk(
        r -> (1 / r^2) * exp(potentialofion(0, r) / (kb * T)),
        10000000,
        r,
        rtol = 1e-9,
    )
    push!(integrald, denominator0[1])

    for i = 1:length(n)-1
        denominator = quadgk(
            r -> (1 / r^2) * exp(potentialofion(n[i+1], r) / (kb * T)),
            10000000,
            r,
            rtol = 1e-9,
        )
        push!(integrald, denominator[1])
    end

    expo = Float64[]
    expo0 = exp(-potentialofion(0, δ) / (kb * T))
    push!(expo, expo0[1])
    for i = 1:length(n)-1
        expon = exp(-potentialofion(n[i+1], δ) / (kb * T))
        push!(expo, expon[1])
    end

    # This is the combination coeffiicent (betan) for ions carrying n elementary charges
    fuchsbeta = map(
        i ->
            (π * γ * ci * δ^2 * expo[i]) /
            (1 + expo[i] * (γ * ci * δ^2 / 4 * D_i) * integrald[i]),
        1:length(n),
    )
    ii = fuchsbeta .< 0
    fuchsbeta[ii] .= 0
    return fuchsbeta
end

function βe(E0, Dp, q)
    a = Dp / 2.0          # radius
    ki = Zi * p/101315.0  # nomenclature

    qs = 3.0 * ϵ / (ϵ + 2.0) * E0 * a^2 / (KE * e)
    if q >= qs
        return 0.0
    else
        return  3.0 * ϵ / (ϵ + 2.0) * π * a^2 * ki * E0 * (1.0 - q / qs)^2
    end
end

function Tc(E0, Dp, Ni, t)
    mβd = βd.(Dp, ncharges)
    mβe = βe.(E0, Dp, ncharges)

    u0 = zeros(length(ncharges), 2)
    u0[1,1] = 1.0
    u0[1,2] = 1.0
    tspan = [0.0, t]

    function f!(du, u, p, t)
        du[1,1] = -mβd[1] * u[1,1] * Ni 
        for i = 2:length(u[:,1])
            du[i, 1] = mβd[i-1] * u[i-1,1] * Ni  - mβd[i] * u[i,1] * Ni 
        end
        
        du[1,2] = -mβe[1] * u[1,2] * Ni 
        for i = 2:length(u[:,1])
            du[i, 2] = mβe[i-1] * u[i-1,2] * Ni  - mβe[i] * u[i,2] * Ni 
        end
    end
    problem = ODEProblem(f!, u0, tspan, Float64[])
    solution = solve(problem, RK4(), reltol = 1e-8, abstol = 1e-8)
    numerical = solution.u[end]

    ppx = numerical[:,1]
    ii = ppx .< 0.001
    ppx[ii] .= 0.0
    p1 = DiscreteNonParametric(ncharges, ppx/sum(ppx))

    ppy = numerical[:,2]
    ii = ppy .< 0.001
    ppy[ii] .= 0.0
    p2 = DiscreteNonParametric(ncharges, ppy/sum(ppy))
    N1 = rand(p1, 100000)
    N2 = rand(p2, 100000)

    N = N1 .+ N2
    A = fit(Histogram, N, ncharges .-0.5, closed=:right)

    return A.weights./sum(A.weights)
end

function analyticalDiffusion(Dd, Nit)
    ci = 462.5 * sqrt(T) / sqrt((273.15 + 21.0))  # Mean thermal speed of ions [m/s],
    return Dd*kb*T/(2*KE*e^2) * log(1.0 + π*KE*Dd*ci*e^2*Nit/(2.0*kb*T))
end

function analyticalField(Dd, Nit, E0)
    return (3*ϵ/(ϵ + 2)) * (E0*Dd^2/(4.0*KE*e)) * (π*KE*e*Zi*Nit)/(1 + π*KE*e*Zi*Nit)
end

function init(Tx, px, ϵx)
    global T = Tx
    global p = px
    global ϵ = ϵx
    
    out = map(Dps) do j
        calculateFuchsBeta(j, ncharges, T, p)
    end

    β = hcat(out...)'[:, :]
    global βd = extrapolate(interpolate((Dps, ncharges), β, Gridded(Linear())), Line())
end

end
