import numpy as np

xa = np.array([1,2,3])
ya = np.array([1,2,3])
X = [np.array([1,2]), np.array([5,6,7,1,2]), np.array([5,6,7,1,2,3,4,5]), np.array([7])]
Y = [np.array([1,2]), np.array([5,6,7,1,2]), np.array([5,6,7,1,2,3,4,5]), np.array([8])]

def line1_in_line2(l1x, l1y, l2x, l2y):
    result = False
    if len(l2x) < len(l1x):
        return False
    for i in range(len(l2x)-len(l1x)+1):
        print (l2x[i:i+len(l1x)], l1x, l2y[i:i+len(l1y)], l1y)
        if np.array_equal(l2x[i:i+len(l1x)], l1x) and np.array_equal(l2y[i:i+len(l1y)], l1y):
            result = True
            break
    return result

for i in range(len(X)):
    print (line1_in_line2(xa, ya, X[i], Y[i]))

def edge_of_line1_in_line2(l1x, l1y, l2x, l2y):
    result = False
    start = False
    end = False
    for i in range(len(l2x)):
        if l1x[0] == l2x[i] and l1y[0] == l2y[i]:
            result = True
            start = True
            break
    for i in range(len(l2x)):
        if l1x[-1] == l2x[i] and l1y[-1] == l2y[i]:
            result = True
            end = True
            break
    return result, start, end

for i in range(len(X)):
    print (edge_of_line1_in_line2(xa, ya, X[i], Y[i]))
    
def ary_in_arys(ary, arys):
    result = False
    for _ary in arys:
        if len(_ary) < len(ary):
            continue
        if result:
            break
        for i in range(len(_ary)-len(ary)+1):
            print (_ary[i:i+len(ary)], ary)
            if np.array_equal(_ary[i:i+len(ary)], ary):
                result = True
                break
    return result

ary_in_arys(a, [b,c])

u = np.array([3, 4])
v = np.array([-4, 3])
def calc_angle(u,v):
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return c


