from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

c = Circuit('Test')
c.V('input', 'in', c.gnd, 5@u_V)
c.R(1, 'in', 'out', 1@u_kΩ)
c.R(2, 'out', c.gnd, 1@u_kΩ)

print('PySpice + Ngspice link: OK')
print(c)