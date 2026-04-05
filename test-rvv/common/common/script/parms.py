import math
from decimal import Decimal, getcontext
import struct

getcontext().prec = 80

# 计算 ln(2) 的值
ln2 = Decimal(2).ln()

def f32(x: float) -> float:
    return struct.unpack('!f', struct.pack('!f', float(x)))[0]

kLog2Inv = 1.0 / float(ln2)
kLog2Hi  = f32(float(ln2))
kLog2Lo  = float(ln2 - Decimal(str(kLog2Hi)))

print("kLog2Inv =", format(kLog2Inv, ".16g")+"f;")
print("kLog2Hi  =", format(kLog2Hi,  ".16g")+"f;")
print("kLog2Lo  =", format(kLog2Lo,  ".16g")+"f;")

# 计算 2^(-127) 的值
kTwoToMinus127 = 2**(-127)
print("kTwoToMinus127 =", format(kTwoToMinus127, ".16g")+"f;")

# 计算 PI 的值
pi = math.pi
print("pi =", format(pi, ".22g")+"f;")

# 计算 PI/2 的值
pi_2 = math.pi / 2
print("pi_2 =", format(pi_2, ".22g")+"f;")
