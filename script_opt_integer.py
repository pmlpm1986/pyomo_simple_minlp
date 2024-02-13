from __future__ import print_function, division
import scipy.io as sio
import numpy as np

""" Pyomo + Couenne based minimization (global-optimal) """
import pyomo.environ as pyo
from pyomo.core.expr.current import log
from pyomo.opt import SolverFactory

# Set the initial values:
K = 2
M = 3
L = 4
P = 100
sigma2 = 1

p={}
p[0,0] = 33.765777471068674
p[0,1] = 35.5742871072786
p[1,0] = 31.21117311348726
p[1,1] = 29.93557001930011
p[2,0] = 35.02304941544408
p[2,1] = 34.49014287342129


G={}
G[0,0] = 0.7862790964000568
G[0,1] = 0.020164266253024834
G[1,0] = 0.261341066254701
G[1,1] = 0.04353507505992858
G[2,0] = 68.7706562914202
G[2,1] = 0.05991469368813834
G[3,0] = 0.06185885418885737
G[3,1] = 0.5381422638807442
G[4,0] = 0.009355407747506612
G[4,1] = 0.13338736820139793
G[5,0] = 0.045406668801663455
G[5,1] = 0.339859899084291

# Solve an integer programming problem using Pyomo + COUENNE solver
# Create a concrete model
m = pyo.ConcreteModel()

# Mutable (settable) model parameters:
m.L = pyo.Param(initialize=L,within=pyo.NonNegativeIntegers)
m.P = pyo.Param(initialize=P,within=pyo.NonNegativeReals)
m.K = pyo.Param(initialize=K,within=pyo.NonNegativeReals)
m.M = pyo.Param(initialize=M,within=pyo.NonNegativeReals)
m.sigma2 = pyo.Param(initialize=sigma2,within=pyo.NonNegativeReals)

# Model parameter ranges:
m.k = pyo.RangeSet(0, m.K-1)
m.m = pyo.RangeSet(0, m.M-1,)
m.km = pyo.RangeSet(0, (m.K*m.M)-1)

# Multi-dimensional model parameters:
m.p = pyo.Param(m.m,m.k,initialize=p,within=pyo.NonNegativeReals)
m.G = pyo.Param(m.km,m.k,initialize=G,within=pyo.NonNegativeReals)

# Set initial values for the multi-dimensional optimization variable 
z={}
z[0,0] = 1
z[0,1] = 0
z[1,0] = 1
z[1,1] = 0
z[2,0] = 1
z[2,1] = 0
z[3,0] = 0
z[3,1] = 1
z[4,0] = 0
z[4,1] = 1
z[5,0] = 0
z[5,1] = 1

# Define and initialize the optimization variable
m.z = pyo.Var(m.km,m.k,initialize=z,within=pyo.NonNegativeIntegers,bounds=(0,L))

def objFun(model):
    # Initialize the objective function:
    fun = 0
    # Loop over all cells:
    for k in model.k:
        # Loop over all users:
        for m in model.m:
            interference = 0;
            for l in model.k:
                if l != k:
                    # Inter-cell interference:
                    interference = interference + (1-model.z[model.M*k+m,l]) * model.G[model.M*k+m,l] * model.P
            # Useful signal:
            useful_signal = model.z[model.M*k+m,k] * model.G[model.M*k+m,k] * model.p[m,k]
            
            # Add to the objective function (model.sigma2 is noise):
            fun = fun + log(1+useful_signal/(model.sigma2+interference))/log(2);
    return fun

m.obj = pyo.Objective(rule=objFun, sense=pyo.maximize)

def ax_constraint_rule(model,i):
    # return the expression for the constraint for i
    return sum(m.z[km,i] for km in m.km) == m.L

m.constr_sum = pyo.Constraint(m.k,rule=ax_constraint_rule)

# *****************************************************************************
# *****************************************************************************

m.pprint()

# opt = SolverFactory('couenne')
opt = SolverFactory('scip')

results = opt.solve(m,keepfiles=True,tee=True)

# m.write('mymodel-couenne-integer.nl')

# *****************************************************************************
# *****************************************************************************
