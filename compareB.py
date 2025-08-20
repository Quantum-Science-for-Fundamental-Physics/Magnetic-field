import numpy as np
import matplotlib.pyplot as plt
import ArbitraryPointBField
import jupyterMOTcode
import math
from scipy.optimize import minimize

x = 0
y = 0
z = 0
R = 17.95272*1e-3 #radius of innermost loop
I = 6.79188720783 #amps
d = (41.2877/2)*1e-3 #d/2 to closest layer, center of wire
num_layers = 4 #stacked vertically
num_loops = 8
Rwire = (1.27/2)*1e-3
plotGrad = True

#num_loops = 1
#num_layers = 1

if (False):
    R = ((4.225+3/16/2)*25.4)*1e-3 #radius of innermost loop
    #I = 89.11 #amps
    d = (((4.88+0.3*2+3/16)*25.4)/2)*1e-3 #d/2 to closest layer, center of wire
    num_layers = 6 #stacked vertically
    num_loops = 5
    Rwire = (((3/16+0.01)*25.4)/2)*1e-3

Rmin = 0.0286258
Rmax = 0.0494919
dMOT = 0.0292481

findOpt = True

"""
if (findOpt == False):
    I = optI
    R = optR
    d = optD
    Rwire = optRwire
"""

"""
R = Rmin
d = dMOT
"""

def compareZ(testI, testR, testd, testWire):
    print(testI)
    print(testR)
    print(testd)
    print(testWire)
    Z_list = np.linspace(-.01, .01, 1000)
    B1 = ArbitraryPointBField.plotZfield(x, y, testR, testI, testd, num_layers, num_loops, testWire, plotGrad)
    B2 = jupyterMOTcode.plotZ()
    #B1 = jupyterMOTcode.plotZ2(testI, testR, testd, testWire)

    #B2 = ArbitraryPointBField.plotZfield(x, y, R, I, d, num_layers, num_loops, Rwire, plotGrad)
    #B1 = ArbitraryPointBField.plotZfield(x, y, testR, testI, testd, num_layers, num_loops, testWire, plotGrad)

    B1_grad = np.gradient(B1, (Z_list[1] - Z_list[0]))
    B2_grad = np.gradient(B2, (Z_list[1] - Z_list[0]))
    total_error = 0.0
    total_grad_error = 0.0

    for i in range (len(B1_grad)):
        B1_grad[i] = B1_grad[i]/100
        B2_grad[i] = B2_grad[i]/100
        
        total_grad_error += 100*math.fabs((math.fabs(B1_grad[i]-B2_grad[i]))/B2_grad[i])
    
    for i in range (len(B1)):
        total_error += 100*math.fabs((math.fabs(B1[i]-B2[i]))/B2[i])

    total_error = total_error/1000
    total_grad_error = total_grad_error/1000
    print(f"Total % error, Bz: {total_error}")
    print(f"Total % error, grad z: {total_grad_error}")

    print(f"Grad adjustment: {B2_grad[500]/B1_grad[500]}")

    if (findOpt):
        return total_error

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(Z_list, B1, label="QSFP")
    axes[0].plot(Z_list, B2, label="DeMille")
    axes[0].set_xlabel("z")
    axes[0].set_title("B field VS z")

    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(Z_list, B1_grad, label="QSFP grad")
    axes[1].plot(Z_list, B2_grad, label="DeMille grad")
    axes[1].set_xlabel("z")
    plt.title("Gradient VS z")
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compareZinput(listParams):
    return compareZ(listParams[0], listParams[1], listParams[2], listParams[3])

def compareX():
    X_list = np.linspace(-1, 1, 1000)
    B1 = ArbitraryPointBField.plotXfield(x, z, R, I, d, num_layers, num_loops, Rwire, plotGrad)
    B2 = jupyterMOTcode.plotX()

    B1_grad = np.gradient(B1, (X_list[1] - X_list[0]))
    B2_grad = np.gradient(B2, (X_list[1] - X_list[0]))
    total_error = 0.0
    total_grad_error = 0.0

    for i in range (len(B1_grad)):
        B1_grad[i] = B1_grad[i]/100
        B2_grad[i] = B2_grad[i]/100
        
        total_grad_error += 100*math.fabs((math.fabs(B1_grad[i]-B2_grad[i]))/B2_grad[i])
    
    for i in range (len(B1)):
        total_error += 100*math.fabs((math.fabs(B1[i]-B2[i]))/B2[i])

    total_error = total_error/1000
    total_grad_error = total_grad_error/1000
    print(f"Total % error, Bx: {total_error}")
    print(f"Total % error, grad x: {total_grad_error}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(X_list, B1, label="B1")
    axes[0].plot(X_list, B2, label="B2")
    axes[0].set_xlabel("x")
    axes[0].set_title("B field VS x")

    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(X_list, B1_grad, label="B1 grad")
    axes[1].plot(X_list, B2_grad, label="B2 grad")
    axes[1].set_xlabel("x")
    plt.title("Gradient VS x")
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

regparams = [I, R, d, Rwire]
if (findOpt):
    testParams = [I * (3), R * (7/4), d * (7/4), Rwire]
    bounds = [(0, 200), (Rmin+0.001, Rmax-0.001), (dMOT+.01, dMOT+.1), (Rwire/2, Rwire*4)]
    result = minimize(compareZinput, x0=testParams, bounds=bounds)
    np.set_printoptions(precision=17, suppress=False)
    print(f"Result: {result.x}")
    print(f"% err: {result.fun}")

    findOpt = False
    compareZinput(result.x)
else:
    compareZinput(regparams)
    #compareX()